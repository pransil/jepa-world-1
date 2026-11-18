"""JEPA core: context encoder, EMA target encoder, horizon heads, latent losses. SPEC §4-§5.1.

Near heads (k = 1..5) predict a mean on L1, scored by squared L2. Far heads (k = 10 on L2,
k = 20 on L3) predict a mean and a log-variance, scored by a Gaussian NLL. Independent of the goal
head and the pixel decoder (SPEC §10). See models/DESIGN.md for the batch contract and decisions.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Collection, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from jepa_world.models.heads import GaussianHead, HorizonHead
from jepa_world.models.unet import encoder_from_cfg

LOG_2PI = math.log(2 * math.pi)

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


def _check_same_shape(**tensors: torch.Tensor) -> None:
    shapes = {name: tuple(t.shape) for name, t in tensors.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"shape mismatch: {shapes}")


def latent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean over (batch, positions) of ||pred - sg(target)||^2 over channels (SPEC §5.1)."""
    _check_same_shape(pred=pred, target=target)
    return (pred - target.detach()).square().sum(dim=1).mean()


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Diagonal-Gaussian NLL of sg(target), summed over channels and averaged over
    (batch, positions), the same reduction as latent_loss (SPEC §5.1, far heads):

        mean_{b,h,w} sum_c 0.5 * [(t - mu)^2 * exp(-logvar) + logvar + log(2 pi)]

    The log(2 pi) term makes it a true NLL. With logvar = 0 it is 0.5 * latent_loss + const."""
    _check_same_shape(mean=mean, logvar=logvar, target=target)
    err2 = (target.detach() - mean).square()
    return (0.5 * (err2 * torch.exp(-logvar) + logvar + LOG_2PI)).sum(dim=1).mean()


def batch_var(z: torch.Tensor) -> torch.Tensor:
    """Variance over the batch of every (channel, position) feature: (B, C, H, W) -> (C, H, W).
    correction=0, so B = 1 gives 0, not NaN."""
    return z.var(dim=0, correction=0)


def _guard_from_var(var: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    return F.relu(gamma - torch.sqrt(var + eps)).mean()


def variance_guard(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    """VICReg-style floor: mean over features of relu(gamma - batch std)
    (docs/decisions/2026-11-17-collapse-guard-on.md)."""
    return _guard_from_var(batch_var(z), gamma, eps)


# ---------------------------------------------------------------- model


class JEPA(nn.Module):
    """Context encoder + EMA target encoder + one head per horizon.

    forward(frames (B, C + k_max, 3, H, W)) -> {
        "context": {level: (B, ch, s, s)}   levels up to the deepest one a head reads,
        "pred":    {k: (B, ch, s, s)}       the mean, for every horizon,
        "logvar":  {k: (B, ch, s, s)}       far horizons only, bounded (GaussianHead),
        "target":  {k: (B, ch, s, s)}       detached, from the target encoder,
    }
    """

    def __init__(self, cfg: dict, horizons: Sequence[int]):
        super().__init__()
        m = cfg["model"]
        self.context_frames = m["context_frames"]
        self.horizons = sorted(horizons)
        self.far_horizons = sorted(set(self.horizons) & set(m["far_horizons"]))
        self.head_level = {k: m["head_level"][k] for k in self.horizons}
        self.context_encoder = encoder_from_cfg(cfg)
        if list(self.context_encoder.buffers()):
            raise RuntimeError("encoder has buffers; ema_update covers parameters only")
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
        self.level_names = list(self.context_encoder.names)
        self.deepest = max(self.head_level.values(), key=self.level_names.index)
        channels = {name: lv["channels"] for name, lv in m["levels"].items()}
        heads: dict[str, nn.Module] = {}
        for k in self.horizons:
            level = self.head_level[k]
            key = level.lower()
            missing = [
                f"{key}_head_{p}" for p in ("hidden", "kernel") if f"{key}_head_{p}" not in m
            ]
            if missing:
                raise KeyError(f"horizon {k} reads {level}, but model.{missing} is not configured")
            common = dict(
                hidden=m[f"{key}_head_hidden"],
                kernel=m[f"{key}_head_kernel"],
                embed_dim=m["horizon_embed_dim"],
                action_dim=m["action_dim"],
                norm_groups=m["norm_groups"],
            )
            ch = channels[level]
            if k in self.far_horizons:
                heads[str(k)] = GaussianHead(ch, ch, logvar_bound=m["far_logvar_bound"], **common)
            else:
                heads[str(k)] = HorizonHead(ch, ch, **common)
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
        pred, logvar = {}, {}
        for k in self.horizons:
            out = self.heads[str(k)](context[self.head_level[k]], action)
            if k in self.far_horizons:
                pred[k], logvar[k] = out
            else:
                pred[k] = out
        return {
            "context": context,
            "pred": pred,
            "logvar": logvar,
            "target": self.encode_targets(frames),
        }

    @torch.no_grad()
    def encode_targets(self, frames: torch.Tensor) -> dict[int, torch.Tensor]:
        """One target-encoder call per level: the stacks of every head reading that level go in
        one batch, encoded only `upto` that level, so near stacks never pay for L2/L3. GroupNorm is
        per sample, so the grouping cannot change any result."""
        b = frames.shape[0]
        targets: dict[int, torch.Tensor] = {}
        for level in self.level_names:
            ks = [k for k in self.horizons if self.head_level[k] == level]
            if not ks:
                continue
            stacks = torch.cat([target_stack(frames, self.context_frames, k) for k in ks])
            z = self.target_encoder(stacks, upto=level)[level]
            for i, k in enumerate(ks):
                targets[k] = z[i * b : (i + 1) * b].detach()
        return targets

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
    model: JEPA,
    frames: torch.Tensor,
    loss_cfg: dict,
    action: torch.Tensor | None = None,
    active: Collection[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Every term as a 0-d tensor, for EVERY built horizon (SPEC §5.4):
      loss/k<k>    squared L2 (near) or Gaussian NLL (far)
      sqerr/k<k>   far only: the mean's squared L2, detached (is the prediction itself improving?)
      logvar/k<k>  far only: mean predicted log-variance, detached
      loss/var     variance guard over the levels read by ACTIVE heads (if the guard is on)
      loss/total   sum of w_k * loss/k over ACTIVE horizons, plus w_var * loss/var
      emb_std/<L>  batch std of every level any head reads, detached

    `active` (default: all built horizons) is the §5.5 stage schedule. Inactive horizons are
    still computed and logged, but stay out of loss/total, so their heads get no gradient."""
    if loss_cfg["latent_distance"] != "mse":
        raise ValueError(
            f"loss.latent_distance={loss_cfg['latent_distance']!r}: only 'mse' (squared L2) "
            "is implemented (docs/decisions/2026-11-17-latent-distance-sq-l2.md)"
        )
    active = set(model.horizons if active is None else active)
    if not active or not active <= set(model.horizons):
        raise ValueError(
            f"active horizons {sorted(active)} must be a non-empty subset of {model.horizons}"
        )
    out = model(frames, action)
    losses: dict[str, torch.Tensor] = {}
    total = frames.new_zeros(())
    for k in model.horizons:
        pred, target = out["pred"][k], out["target"][k]
        if k in model.far_horizons:
            lk = gaussian_nll(pred, out["logvar"][k], target)
            losses[f"sqerr/k{k}"] = latent_loss(pred.detach(), target)
            losses[f"logvar/k{k}"] = out["logvar"][k].detach().mean()
        else:
            lk = latent_loss(pred, target)
        losses[f"loss/k{k}"] = lk
        if k in active:
            total = total + loss_cfg["w_k"] * lk
    active_levels = {model.head_level[k] for k in active}
    guards, stds = [], {}
    for lv in sorted(set(model.head_level.values()), key=model.level_names.index):
        var = batch_var(out["context"][lv])  # one reduction, used for the guard and the log
        stds[f"emb_std/{lv}"] = var.detach().sqrt().mean()
        if loss_cfg["collapse_guard"] and lv in active_levels:
            guards.append(_guard_from_var(var, loss_cfg["var_gamma"], loss_cfg["var_eps"]))
    if guards:
        losses["loss/var"] = torch.stack(guards).mean()
        total = total + loss_cfg["w_var"] * losses["loss/var"]
    losses["loss/total"] = total
    losses.update(stds)
    return losses
