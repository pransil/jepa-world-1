"""The full model: the JEPA core plus the auxiliary goal head and pixel decoder (SPEC §4.1, §4.6,
§4.7, §5.2-§5.5). See models/DESIGN-aux.md.

WorldModel subclasses JEPA, so the core's state_dict keys are unchanged (pre-M5 checkpoints load
strictly) and jepa.py stays independent of the aux heads (SPEC §10). The aux heads read only the
CONTEXT levels (frames t-C+1 .. t). Ground truth (ball_pos) is a label for the goal loss and never
reaches forward().
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence

import torch
from torch import nn

from jepa_world.models.goal import GoalHead, displacement_targets, goal_nll
from jepa_world.models.jepa import JEPA, losses_from_output
from jepa_world.models.pixel import PixelDecoder, pixel_l1

AUX_STAGES = ("goal", "pixel")


class WorldModel(JEPA):
    """JEPA + optional GoalHead (`goal`) + optional PixelDecoder (`pixel`).

    forward(frames, action, active, aux_active) -> JEPA.forward's dict, plus
        "goal":  (mean, logvar), each (B, K, 2)   if the goal head is built
        "pixel": (B, 3, S, S) in [0, 1]           if the pixel decoder is built
    An aux head that is built but not in `aux_active` still runs (it is logged every step), under
    no_grad, like an inactive horizon head (docs/decisions/2026-11-17-stage-schedule-within-run.md).
    """

    def __init__(self, cfg: dict, horizons: Sequence[int], *, goal: bool, pixel: bool):
        m = cfg["model"]
        levels = list(m["levels"])  # fine -> coarse
        goal_levels = [levels[-1], *([levels[-2]] if m["goal_l2"] else [])] if goal else []
        pixel_levels = list(levels) if pixel else []
        super().__init__(cfg, horizons, extra_levels=[*goal_levels, *pixel_levels])
        self.goal_levels, self.pixel_levels = goal_levels, pixel_levels  # L3 [, L2]; L1..L3
        ch = {name: lv["channels"] for name, lv in m["levels"].items()}
        self.goal = (
            GoalHead(
                ch[levels[-1]],
                l2_ch=ch[levels[-2]] if m["goal_l2"] else 0,
                horizons=m["goal_horizons"],
                hidden=m["goal_hidden"],
                embed_dim=m["horizon_embed_dim"],
                norm_groups=m["norm_groups"],
                logvar_bound=m["goal_logvar_bound"],
            )
            if goal
            else None
        )
        self.pixel = (
            PixelDecoder(
                [(lv["channels"], lv["size"]) for lv in m["levels"].values()],
                widths=m["pix_widths"],
                norm_groups=m["norm_groups"],
                skips=m["unet_skips"],
            )
            if pixel
            else None
        )

    @property
    def aux_stages(self) -> list[str]:
        """The aux stages this model has heads for, in AUX_STAGES order."""
        return [s for s in AUX_STAGES if getattr(self, s) is not None]

    def check_aux(self, aux_active: Collection[str] | None) -> set[str]:
        built = set(self.aux_stages)
        aux_active = built if aux_active is None else set(aux_active)
        if not aux_active <= built:
            raise ValueError(
                f"aux stages {sorted(aux_active - built)} are active but not built "
                f"(built: {sorted(built)}; add them to train.stages)"
            )
        return aux_active

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().trainable_parameters()
        for s in self.aux_stages:
            yield from getattr(self, s).parameters()

    def forward(
        self,
        frames: torch.Tensor,
        action: torch.Tensor | None = None,
        active: Collection[int] | None = None,
        aux_active: Collection[str] | None = None,
    ) -> dict:
        aux_active = self.check_aux(aux_active)
        grad_levels = [
            *(self.goal_levels if "goal" in aux_active else []),
            *(self.pixel_levels if "pixel" in aux_active else []),
        ]
        out = super().forward(frames, action, active=active, grad_levels=grad_levels)
        ctx = out["context"]  # context levels only: never the target encoder's output
        grad = torch.is_grad_enabled()
        if self.goal is not None:
            with torch.set_grad_enabled(grad and "goal" in aux_active):
                out["goal"] = self.goal(*(ctx[lv] for lv in self.goal_levels))
        if self.pixel is not None:
            with torch.set_grad_enabled(grad and "pixel" in aux_active):
                out["pixel"] = self.pixel([ctx[lv] for lv in self.pixel_levels])
        return out


def split_batch(batch: torch.Tensor | Mapping[str, torch.Tensor]):
    """A frames tensor (the M3 layout) or a with_gt dict -> (frames, ball_pos or None)."""
    if isinstance(batch, torch.Tensor):
        return batch, None
    return batch["frames"], batch.get("ball_pos")


def compute_world_losses(
    model: WorldModel,
    batch: torch.Tensor | Mapping[str, torch.Tensor],
    loss_cfg: dict,
    action: torch.Tensor | None = None,
    active: Collection[int] | None = None,
    aux_active: Collection[str] | None = None,
) -> dict[str, torch.Tensor]:
    """compute_losses' terms (jepa.py), plus, for each BUILT aux head, every step:
      loss/goal/k<k>   goal NLL of the displacement ball_pos[t+k] - ball_pos[t], per goal horizon
      goalerr/k<k>     mean |mean - displacement| (world m), detached
      goalratio/k<k>   goalerr / mean |displacement|: the error relative to persistence (which
                       predicts no motion), detached; < 1 beats persistence (SPEC §7.2, §11)
      loss/goal        sum of loss/goal/k<k> over the goal horizons (SPEC §5.2)
      loss/pix         mean |pixel decoder - frame t+1| (SPEC §5.3)
    loss/total adds w_goal * loss/goal and w_pix * loss/pix only for stages in `aux_active`
    (default: every built one). The model gets the frames only; ball_pos is a label."""
    frames, ball_pos = split_batch(batch)
    if model.goal is not None and ball_pos is None:
        raise ValueError(
            "the goal stage needs ball_pos: use batches with_gt=True (train.make_batches does "
            "this when train.stages has goal)"
        )
    active = model.check_active(active)
    aux_active = model.check_aux(aux_active)
    out = model(frames, action, active=active, aux_active=aux_active)
    losses = losses_from_output(model, out, loss_cfg, active)
    total = losses.pop("loss/total")
    if model.goal is not None:
        target = displacement_targets(ball_pos, model.context_frames, model.goal.horizons)
        mean, logvar = out["goal"]
        nll = goal_nll(mean, logvar, target)  # (K,)
        err = (mean.detach() - target).norm(dim=-1).mean(dim=0)  # (K,) metres
        persist = target.norm(dim=-1).mean(dim=0).clamp_min(1e-12)  # persistence's error
        for j, k in enumerate(model.goal.horizons):
            losses[f"loss/goal/k{k}"] = nll[j]
            losses[f"goalerr/k{k}"] = err[j]
            losses[f"goalratio/k{k}"] = err[j] / persist[j]
        losses["loss/goal"] = nll.sum()
        if "goal" in aux_active:
            total = total + loss_cfg["w_goal"] * losses["loss/goal"]
    if model.pixel is not None:
        losses["loss/pix"] = pixel_l1(out["pixel"], frames[:, model.context_frames])  # t+1
        if "pixel" in aux_active:
            total = total + loss_cfg["w_pix"] * losses["loss/pix"]
    losses["loss/total"] = total
    return losses
