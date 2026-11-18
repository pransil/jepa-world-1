"""JEPA core: context encoder, EMA target encoder, horizon heads, latent loss. SPEC §4-§5.1.

Independent of the goal head and the pixel decoder (SPEC §10). See models/DESIGN.md for the batch
contract and the decisions behind it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from jepa_world.models.heads import HorizonHead
from jepa_world.models.unet import encoder_from_cfg

# ---------------------------------------------------------------- batch slicing


def _stack(clip: torch.Tensor) -> torch.Tensor:
    """(B, C, 3, H, W) -> (B, 3C, H, W): RGB of each frame, frames in time order."""
    b, c, ch, h, w = clip.shape
    return clip.reshape(b, c * ch, h, w)


def context_stack(frames: torch.Tensor, context_frames: int) -> torch.Tensor:
    """Frames t-C+1 ... t, i.e. frames[:, :C]. Index C-1 is time t."""
    return _stack(frames[:, :context_frames])


def target_stack(frames: torch.Tensor, context_frames: int, k: int) -> torch.Tensor:
    """The C-frame stack ending at t+k: frames[:, k : k+C]
    (docs/decisions/2026-11-17-target-input-stack.md)."""
    return _stack(frames[:, k : k + context_frames])


# ---------------------------------------------------------------- losses


def latent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean over (batch, positions) of ||pred - sg(target)||^2 over channels (SPEC §5.1)."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    return (pred - target.detach()).square().sum(dim=1).mean()


def batch_std(z: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Std over the batch of every (channel, position) feature. correction=0, so B=1 gives 0."""
    return torch.sqrt(z.var(dim=0, correction=0) + eps)


def variance_guard(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    """VICReg-style floor: mean over features of relu(gamma - batch std)
    (docs/decisions/2026-11-17-collapse-guard-on.md)."""
    return F.relu(gamma - batch_std(z, eps)).mean()


# ---------------------------------------------------------------- model


class JEPA(nn.Module):
    """Context encoder + EMA target encoder + one head per horizon.

    forward(frames (B, C + k_max, 3, H, W)) -> {
        "context": {level: (B, ch, s, s)}   levels up to the deepest one a head reads,
        "pred":    {k: (B, ch, s, s)},
        "target":  {k: (B, ch, s, s)}       detached, from the target encoder,
    }
    """

    def __init__(self, cfg: dict, horizons: Sequence[int]):
        super().__init__()
        m = cfg["model"]
        far = sorted(set(horizons) & set(m["far_horizons"]))
        if far:
            raise NotImplementedError(f"far horizons {far}: Gaussian heads arrive in M4")
        self.context_frames = m["context_frames"]
        self.horizons = sorted(horizons)
        self.head_level = {k: m["head_level"][k] for k in self.horizons}
        self.context_encoder = encoder_from_cfg(cfg)
        if list(self.context_encoder.buffers()):
            raise RuntimeError("encoder has buffers; ema_update covers parameters only")
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
        names = self.context_encoder.names
        self.deepest = max(self.head_level.values(), key=names.index)
        channels = {name: lv["channels"] for name, lv in m["levels"].items()}
        heads = {}
        for k in self.horizons:
            level = self.head_level[k]
            if level != "L1":
                raise NotImplementedError(f"horizon {k} reads {level}: only L1 heads exist in M3")
            heads[str(k)] = HorizonHead(
                channels[level],
                channels[level],
                hidden=m["l1_head_hidden"],
                kernel=m["l1_head_kernel"],
                embed_dim=m["horizon_embed_dim"],
                action_dim=m["action_dim"],
                norm_groups=m["norm_groups"],
            )
        self.heads = nn.ModuleDict(heads)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Everything the optimizer may touch: the context encoder and the heads, never the target."""
        yield from self.context_encoder.parameters()
        yield from self.heads.parameters()

    def forward(self, frames: torch.Tensor, action: torch.Tensor | None = None) -> dict:
        need = self.context_frames + max(self.horizons)
        if frames.ndim != 5 or frames.shape[1] < need:
            raise ValueError(
                f"need frames (B, >= {need}, 3, H, W) for horizons {self.horizons}, "
                f"got {tuple(frames.shape)}"
            )
        context = self.context_encoder(
            context_stack(frames, self.context_frames), upto=self.deepest
        )
        pred = {k: self.heads[str(k)](context[self.head_level[k]], action) for k in self.horizons}
        return {"context": context, "pred": pred, "target": self.encode_targets(frames)}

    @torch.no_grad()
    def encode_targets(self, frames: torch.Tensor) -> dict[int, torch.Tensor]:
        """All horizons in one (B*K) batch through the target encoder, stopping at the deepest
        level a head reads. GroupNorm is per sample, so stacks do not influence each other."""
        b = frames.shape[0]
        stacks = torch.cat([target_stack(frames, self.context_frames, k) for k in self.horizons])
        levels = self.target_encoder(stacks, upto=self.deepest)
        return {
            k: levels[self.head_level[k]][i * b : (i + 1) * b].detach()
            for i, k in enumerate(self.horizons)
        }

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        """theta_target <- m * theta_target + (1 - m) * theta_context (SPEC §4.4).
        Call once per step, after optimizer.step()."""
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"EMA momentum must be in [0, 1], got {momentum}")
        for pt, pc in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            pt.mul_(momentum).add_(pc, alpha=1.0 - momentum)


def compute_losses(
    model: JEPA, frames: torch.Tensor, loss_cfg: dict, action: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Every term as a 0-d tensor: loss/k<k> for each horizon, loss/var (if the guard is on),
    loss/total, and emb_std/<level> (detached, so collapse is visible with the guard off)."""
    if loss_cfg["latent_distance"] != "mse":
        raise ValueError(
            f"loss.latent_distance={loss_cfg['latent_distance']!r}: only 'mse' (squared L2) "
            "is implemented (docs/decisions/2026-11-17-latent-distance-sq-l2.md)"
        )
    out = model(frames, action)
    losses: dict[str, torch.Tensor] = {}
    total = frames.new_zeros(())
    for k in model.horizons:
        lk = latent_loss(out["pred"][k], out["target"][k])
        losses[f"loss/k{k}"] = lk
        total = total + loss_cfg["w_k"] * lk
    read = sorted(set(model.head_level.values()))
    if loss_cfg["collapse_guard"]:
        var = torch.stack(
            [
                variance_guard(out["context"][lv], loss_cfg["var_gamma"], loss_cfg["var_eps"])
                for lv in read
            ]
        ).mean()
        losses["loss/var"] = var
        total = total + loss_cfg["w_var"] * var
    losses["loss/total"] = total
    for lv in read:
        losses[f"emb_std/{lv}"] = batch_std(out["context"][lv].detach()).mean()
    return losses
