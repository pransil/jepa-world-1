"""JEPA core: context encoder, EMA target encoder, horizon heads, latent losses. SPEC §4-§5.1.

Near heads (k = 1..5) predict a mean on L1, scored by squared L2. Far heads (k = 10 on L2,
k = 20 on L3) predict a mean and a log-variance, scored by a Gaussian NLL. With model.top_down (M7,
off by default), each far head's level also predicts the next-finer target at its horizon (SPEC
§9.2). Independent of the goal head and the pixel decoder (SPEC §10). See models/DESIGN.md for the
batch contract and decisions.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from jepa_world.models.heads import GaussianHead, HorizonHead, TopDownHead
from jepa_world.models.losses import gaussian_nll_terms
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


def _check_same_shape(**tensors: torch.Tensor) -> None:
    shapes = {name: tuple(t.shape) for name, t in tensors.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"shape mismatch: {shapes}")


# The one place each loss is defined. latent_loss / gaussian_nll (public) and compute_losses
# (training) both go through these, so the two can never drift apart.

LATENT_REDUCE = ("sum", "mean")  # loss.latent_reduce


def _checked_reduce(reduce: str) -> str:
    if reduce not in LATENT_REDUCE:
        raise ValueError(f"loss.latent_reduce={reduce!r}: expected one of {LATENT_REDUCE}")
    return reduce


def latent_reduce(loss_cfg: dict) -> str:
    """loss.latent_reduce, validated: 'sum', or 'mean' (every per-level term divided by its
    level's channel count: docs/decisions/2026-11-19-latent-loss-normalization.md). A config
    from before 2026-11-19 has no key, and means 'sum'. The one place that default lives."""
    return _checked_reduce(loss_cfg.get("latent_reduce", "sum"))


def over_channels(x: torch.Tensor | float, channels: int, reduce: str) -> torch.Tensor | float:
    """A channel-summed term (or a weight on one) as it is ('sum'), or per channel ('mean':
    divided by `channels`). 'sum' returns `x` itself, so the sum path is bitwise what it was
    before the switch. world.py applies it to the fixed aux weights too."""
    return x if _checked_reduce(reduce) == "sum" else x / channels


def _err2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_same_shape(pred=pred, target=target)
    return (target.detach() - pred).square()


def _sq_from_err2(err2: torch.Tensor) -> torch.Tensor:
    return err2.sum(dim=1).mean()


def _nll_from_err2(err2: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return gaussian_nll_terms(err2, logvar).sum(dim=1).mean()  # the one NLL formula (losses.py)


def _relerr_from_sq(sq: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    if target.shape[0] < 2:  # a batch variance needs two samples
        return torch.full_like(sq, float("nan"))
    spread = batch_var(target.detach()).sum(dim=0).mean()  # = latent_loss(batch mean, target)
    return sq.detach() / spread.clamp_min(eps)


def latent_loss(pred: torch.Tensor, target: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """mean over (batch, positions) of ||pred - sg(target)||^2 over channels (SPEC §5.1);
    reduce='mean' divides by the channel count (loss.latent_reduce)."""
    return over_channels(_sq_from_err2(_err2(pred, target)), pred.shape[1], reduce)


def gaussian_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, reduce: str = "sum"
) -> torch.Tensor:
    """Diagonal-Gaussian NLL of sg(target), summed over channels and averaged over
    (batch, positions), the same reduction as latent_loss (SPEC §5.1, far heads):

        mean_{b,h,w} sum_c 0.5 * [(t - mu)^2 * exp(-logvar) + logvar + log(2 pi)]

    The log(2 pi) term makes it a true NLL. With logvar = 0 it is 0.5 * latent_loss + const.
    reduce='mean' divides by the channel count, as latent_loss does."""
    _check_same_shape(mean=mean, logvar=logvar)
    return over_channels(_nll_from_err2(_err2(mean, target), logvar), mean.shape[1], reduce)


def relative_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale-free error: latent_loss(pred, target) / latent_loss(batch mean of target, target).
    1.0 = no better than predicting the batch-mean target; 0 = exact. Shrinking the embeddings
    cannot lower it, so it shows real prediction quality when the raw losses are collapsing.
    NaN when the batch has one sample (no batch variance). Detached: a diagnostic only."""
    return _relerr_from_sq(latent_loss(pred.detach(), target), target, eps)


def normalize_target(target: torch.Tensor, mode: str) -> torch.Tensor:
    """loss.target_norm: 'none' (raw, the 2026-11-17 latent-distance decision) or 'layernorm'
    (zero mean, unit variance over channels at each position; no affine, as in I-JEPA)."""
    if mode == "none":
        return target
    if mode == "layernorm":
        return F.layer_norm(target.movedim(1, -1), (target.shape[1],)).movedim(-1, 1)
    raise ValueError(f"loss.target_norm={mode!r}: expected 'none' or 'layernorm'")


def batch_var(z: torch.Tensor) -> torch.Tensor:
    """Variance over the batch of every (channel, position) feature: (B, C, H, W) -> (C, H, W).
    correction=0, so B = 1 gives 0, not NaN."""
    return z.var(dim=0, correction=0)


def _guard_from_var(var: torch.Tensor, gamma: float, eps: float, reduce: str) -> torch.Tensor:
    g = F.relu(gamma - torch.sqrt(var + eps))  # (C, H, W)
    if reduce == "mean":
        return g.mean()
    if reduce == "sum":  # same scale as the latent losses: sum over channels, mean over positions
        return g.sum(dim=0).mean()
    raise ValueError(f"loss.var_reduce={reduce!r}: expected 'mean' or 'sum'")


def variance_guard(z: torch.Tensor, gamma: float, eps: float, reduce: str = "mean") -> torch.Tensor:
    """VICReg-style floor on relu(gamma - batch std) of every (channel, position) feature,
    averaged over features ('mean') or summed over channels ('sum')
    (docs/decisions/2026-11-17-collapse-guard-on.md)."""
    return _guard_from_var(batch_var(z), gamma, eps, reduce)


# ---------------------------------------------------------------- model


class JEPA(nn.Module):
    """Context encoder + EMA target encoder + one head per horizon.

    `extra_levels`: levels some consumer outside the JEPA core reads (the goal head and pixel
    decoder, models/world.py), so they are always encoded even if no horizon head reads them.
    This module knows nothing else about those consumers (SPEC §10).

    forward(frames (B, C + k_max, 3, H, W)) -> {
        "context": {level: (B, ch, s, s)}   levels up to the deepest one a head or an
                                            extra_levels consumer reads,
        "pred":    {k: (B, ch, s, s)}       the mean, for every horizon,
        "logvar":  {k: (B, ch, s, s)}       far horizons only, bounded (GaussianHead),
        "target":  {k: (B, ch, s, s)}       detached, from the target encoder, after loss.target_norm,
    } plus, only with model.top_down, for each far horizon k with a finer level below its own:
        "td_pred", "td_logvar": {k: (B, ch', s', s')}   the branch's Gaussian on the finer level,
        "td_target":            {k: (B, ch', s', s')}   that level of the target stack ending at t+k.
    """

    def __init__(self, cfg: dict, horizons: Sequence[int], extra_levels: Collection[str] = ()):
        super().__init__()
        m = cfg["model"]
        self.context_frames = m["context_frames"]
        self.horizons = sorted(horizons)
        self.far_horizons = sorted(set(self.horizons) & set(m["far_horizons"]))
        self.head_level = {k: m["head_level"][k] for k in self.horizons}
        # What the heads are scored against IS part of the target, so it is applied here (in
        # encode_targets), not in the loss: forward()["target"] is exactly what heads predict.
        self.target_norm = cfg["loss"]["target_norm"]
        normalize_target(torch.zeros(1, 2, 1, 1), self.target_norm)  # fail at build time if unknown
        self.context_encoder = encoder_from_cfg(cfg)
        if list(self.context_encoder.buffers()):
            raise RuntimeError("encoder has buffers; ema_update covers parameters only")
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
        self.level_names = list(self.context_encoder.names)
        unknown = set(extra_levels) - set(self.level_names)
        if unknown:
            raise ValueError(f"extra_levels {sorted(unknown)}: no such level ({self.level_names})")
        self.deepest = max([*self.head_level.values(), *extra_levels], key=self.level_names.index)
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
        self._build_top_down(m, channels)

    def _build_top_down(self, m: dict, channels: dict[str, int]) -> None:
        """M7 top-down branches (docs/decisions/2026-11-17-top-down-targets.md). Off, nothing is
        built, so the state-dict keys stay as before. A config saved before M7 has no
        model.top_down and reads as off. Built after the heads, so their init is unchanged."""
        top_down = m.get("top_down", False)
        if not isinstance(top_down, bool):  # a quoted "false" must not switch it on
            raise ValueError(f"model.top_down must be true or false, got {top_down!r}")
        self.top_down = top_down
        self.td_level: dict[int, str] = {}  # far horizon -> the finer level its branch predicts
        self.td_heads: nn.ModuleDict | None = None
        if not self.top_down:
            return
        for k in self.far_horizons:
            i = self.level_names.index(self.head_level[k])
            if i > 0:
                self.td_level[k] = self.level_names[i - 1]
        if not self.td_level:
            raise ValueError(
                "model.top_down needs a far head on a level with a finer level below it, and none "
                f"is built (far horizons {self.far_horizons}; add the far stage to train.stages)"
            )
        shape = (
            self.context_encoder.level_shape
        )  # {level: (channels, size)}, validated by the encoder
        td: dict[str, nn.Module] = {}
        for k, fine in self.td_level.items():
            coarse = self.head_level[k]
            scale = shape[fine][1] // shape[coarse][1]  # the encoder only builds power-of-2 steps
            key = fine.lower()  # the branch's output lives at the finer level: use its head shape
            td[str(k)] = TopDownHead(
                channels[coarse],
                channels[fine],
                scale=scale,
                logvar_bound=m["far_logvar_bound"],
                hidden=m[f"{key}_head_hidden"],
                kernel=m[f"{key}_head_kernel"],
                embed_dim=m["horizon_embed_dim"],
                action_dim=m["action_dim"],
                norm_groups=m["norm_groups"],
            )
        self.td_heads = nn.ModuleDict(td)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Everything the optimizer may touch: the context encoder and the heads, never the target."""
        yield from self.context_encoder.parameters()
        yield from self.heads.parameters()
        if self.td_heads is not None:
            yield from self.td_heads.parameters()

    def check_active(self, active: Collection[int] | None) -> set[int]:
        active = set(self.horizons if active is None else active)
        if not active or not active <= set(self.horizons):
            raise ValueError(
                f"active horizons {sorted(active)} must be a non-empty subset of {self.horizons}"
            )
        return active

    def forward(
        self,
        frames: torch.Tensor,
        action: torch.Tensor | None = None,
        active: Collection[int] | None = None,
        grad_levels: Collection[str] = (),
    ) -> dict:
        """`active` (default: all) are the horizons that will be trained. Every head still runs,
        but inactive heads, and context levels only they read, run without autograd: same values,
        no graph kept for a backward that would never use it. `grad_levels` are levels an active
        consumer outside the core trains on, so they keep autograd too."""
        active = self.check_active(active)
        need = self.context_frames + max(self.horizons)
        if frames.ndim != 5 or frames.shape[1] < need:
            raise ValueError(
                f"need frames (B, >= {need}, 3, H, W) for horizons {self.horizons}, "
                f"got {tuple(frames.shape)}"
            )
        deepest_active = max(
            [*(self.head_level[k] for k in active), *grad_levels], key=self.level_names.index
        )
        context = self.context_encoder(
            context_stack(frames, self.context_frames), upto=deepest_active
        )
        if deepest_active != self.deepest:
            with torch.no_grad():
                context.update(
                    self.context_encoder(
                        context[deepest_active], upto=self.deepest, after=deepest_active
                    )
                )
        pred, logvar = {}, {}
        for k in self.horizons:
            with torch.set_grad_enabled(torch.is_grad_enabled() and k in active):
                out = self.heads[str(k)](context[self.head_level[k]], action)
            if k in self.far_horizons:
                pred[k], logvar[k] = out
            else:
                pred[k] = out
        targets, td_targets = self._encode(frames)
        result = {"context": context, "pred": pred, "logvar": logvar, "target": targets}
        if self.td_heads is not None:
            td_pred, td_logvar = {}, {}
            for k in self.td_level:  # each branch trains exactly when its far horizon does
                with torch.set_grad_enabled(torch.is_grad_enabled() and k in active):
                    td_pred[k], td_logvar[k] = self.td_heads[str(k)](
                        context[self.head_level[k]], action
                    )
            result.update(td_pred=td_pred, td_logvar=td_logvar, td_target=td_targets)
        return result

    @torch.no_grad()
    def encode_targets(self, frames: torch.Tensor) -> dict[int, torch.Tensor]:
        """One target-encoder call per level: the stacks of every head reading that level go in
        one batch, encoded only `upto` that level, so near stacks never pay for L2/L3. GroupNorm is
        per sample, so the grouping cannot change any result. loss.target_norm is applied here."""
        return self._encode(frames)[0]

    @torch.no_grad()
    def _encode(
        self, frames: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """encode_targets, plus the top-down targets: for each branch, the finer level of its
        horizon's stack. That level is computed on the way to the far head's level anyway, so it
        costs no extra encoder pass."""
        b = frames.shape[0]
        targets: dict[int, torch.Tensor] = {}
        td: dict[int, torch.Tensor] = {}
        for level in self.level_names:
            ks = [k for k in self.horizons if self.head_level[k] == level]
            if not ks:
                continue
            stacks = torch.cat([target_stack(frames, self.context_frames, k) for k in ks])
            zs = self.target_encoder(stacks, upto=level)
            z = zs[level]
            fine = {self.td_level[k]: zs[self.td_level[k]] for k in ks if k in self.td_level}
            del zs  # free the other intermediate levels now, as before top-down (peak memory)
            z = normalize_target(z, self.target_norm)
            for i, k in enumerate(ks):
                targets[k] = z[i * b : (i + 1) * b].detach()
                if k in self.td_level:
                    zf = fine[self.td_level[k]][i * b : (i + 1) * b]
                    td[k] = normalize_target(zf, self.target_norm).detach()
        return targets, td

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
      loss/k<k>    squared L2 (near) or Gaussian NLL (far), against the (normalized) target.
                   Summed over channels, or, with loss.latent_reduce: mean, averaged over them;
                   so are sqerr/k, loss/td/k, sqerr/td/k and each level's guard (var/<L>), whose
                   own var_reduce is relative to that scale
      relerr/k<k>  scale-free error of the mean (relative_error), detached
      sqerr/k<k>   far only: the mean's squared L2, detached (is the prediction itself improving?)
      logvar/k<k>  far only: mean predicted log-variance, detached
      var/<L>      each read level's variance guard, detached (if the guard is on)
      loss/var     SUM of the guards of the levels ACTIVE heads read (if the guard is on)
      loss/total   sum of w_k * loss/k over ACTIVE horizons, plus w_var * loss/var
      emb_std/<L>  batch std of every level any head reads, detached
    and, with model.top_down, for each branch k (M7):
      loss/td/k<k>  Gaussian NLL of the finer target at t+k; w_td * it joins loss/total if k ACTIVE
      sqerr/td/k<k>, logvar/td/k<k>, relerr/td/k<k>   as for the far heads, detached

    `active` (default: all built horizons) is the §5.5 stage schedule. Inactive horizons are
    still computed and logged, but stay out of loss/total, so their heads get no gradient."""
    check_loss_cfg(model, loss_cfg)  # before the forward, so a bad config fails fast
    active = model.check_active(active)
    return losses_from_output(model, model(frames, action, active=active), loss_cfg, active)


def check_loss_cfg(model: JEPA, loss_cfg: dict) -> None:
    latent_reduce(loss_cfg)
    if loss_cfg["latent_distance"] != "mse":
        raise ValueError(
            f"loss.latent_distance={loss_cfg['latent_distance']!r}: only 'mse' (squared L2) "
            "is implemented (docs/decisions/2026-11-17-latent-distance-sq-l2.md)"
        )
    if model.td_heads is not None and "w_td" not in loss_cfg:
        raise ValueError("model.top_down is on, so loss.w_td must be set (base.yaml: 0.5)")
    if loss_cfg["target_norm"] != model.target_norm:
        raise ValueError(
            f"loss.target_norm={loss_cfg['target_norm']!r} but the model was built with "
            f"{model.target_norm!r}: the targets are normalized inside the model"
        )


def _gaussian_terms(
    losses: dict,
    tag: str,
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    reduce: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Gaussian head's NLL against its detached target (= gaussian_nll), and its detached
    diagnostics sqerr/<tag> and logvar/<tag>. The far heads and the top-down branches both go
    through it, so their terms can't drift apart. Returns (nll, the CHANNEL-SUMMED sqerr, which
    relerr divides by a channel-summed spread whatever the reduction)."""
    err2 = _err2(pred, target)  # computed once, shared by the loss and the diagnostics
    sqerr = _sq_from_err2(err2)  # = latent_loss(pred, target)
    channels = target.shape[1]
    losses[f"sqerr/{tag}"] = over_channels(sqerr, channels, reduce).detach()
    losses[f"logvar/{tag}"] = logvar.detach().mean()
    return over_channels(_nll_from_err2(err2, logvar), channels, reduce), sqerr


def add_term(
    total: torch.Tensor, terms: dict[str, torch.Tensor] | None, name: str, term: torch.Tensor
) -> torch.Tensor:
    """total + term, recording the weighted term under `name` when `terms` is given. The sum is
    the one it always was (the same operands, in the same order), so recording changes nothing."""
    if terms is not None:
        terms[name] = term
    return total + term


def losses_from_output(
    model: JEPA,
    out: dict,
    loss_cfg: dict,
    active: Collection[int],
    terms: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """compute_losses' terms from an existing forward() output, so a caller that also needs the
    output (models/world.py) runs the model once. The same terms and the same total.
    The caller validates first, before its forward: check_loss_cfg, and `active` from
    model.check_active (compute_losses and world.compute_world_losses both do).
    `terms`, if given, is filled with each weighted term exactly as it enters loss/total, with its
    graph ("k<k>", "td/k<k>", "var"): what train.encoder_grad_norms differentiates."""
    losses: dict[str, torch.Tensor] = {}
    reduce = latent_reduce(loss_cfg)
    total = out["context"][model.level_names[0]].new_zeros(())
    for k in model.horizons:
        pred, target = out["pred"][k], out["target"][k]  # target: detached, already normalized
        if k in model.far_horizons:
            lk, sqerr = _gaussian_terms(losses, f"k{k}", pred, out["logvar"][k], target, reduce)
        else:
            sqerr = _sq_from_err2(_err2(pred, target))  # = latent_loss(pred, target)
            lk = over_channels(sqerr, target.shape[1], reduce)
        losses[f"loss/k{k}"] = lk
        losses[f"relerr/k{k}"] = _relerr_from_sq(sqerr, target, eps=1e-12)
        if k in active:
            total = add_term(total, terms, f"k{k}", loss_cfg["w_k"] * lk)
    for k in model.td_level:  # empty unless model.top_down
        tag, target = f"td/k{k}", out["td_target"][k]
        ltd, sqerr = _gaussian_terms(
            losses, tag, out["td_pred"][k], out["td_logvar"][k], target, reduce
        )
        losses[f"loss/{tag}"] = ltd
        losses[f"relerr/{tag}"] = _relerr_from_sq(sqerr, target, eps=1e-12)
        if k in active:
            total = add_term(total, terms, tag, loss_cfg["w_td"] * ltd)
    active_levels = {model.head_level[k] for k in active}
    guards, extra = [], {}
    for lv in sorted(set(model.head_level.values()), key=model.level_names.index):
        var = batch_var(out["context"][lv])  # one reduction, used for the guard and the log
        extra[f"emb_std/{lv}"] = var.detach().sqrt().mean()
        if loss_cfg["collapse_guard"]:
            # var_reduce is relative to the latent terms' channel reduction ('sum': their scale),
            # so under either latent_reduce, w_var and var_reduce alone set each level's
            # guard-to-latent ratio (M4's collapse fix: far-collapse-guard-sum decision)
            g = over_channels(
                _guard_from_var(
                    var, loss_cfg["var_gamma"], loss_cfg["var_eps"], loss_cfg["var_reduce"]
                ),
                var.shape[0],
                reduce,
            )
            extra[f"var/{lv}"] = g.detach()
            if lv in active_levels:
                guards.append(g)
    if guards:  # summed, so a level's weight never changes when another level joins
        losses["loss/var"] = torch.stack(guards).sum()
        total = add_term(total, terms, "var", loss_cfg["w_var"] * losses["loss/var"])
    losses["loss/total"] = total
    losses.update(extra)
    return losses


@torch.no_grad()
def copy_relerr(
    model: JEPA, frames: torch.Tensor, targets: dict[int, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """relerr/copy/k<k>: the latent heads' persistence bar. The prediction is the target encoder's
    embedding of the stack ending at t, at the level head k reads (after loss.target_norm), and
    `targets` are forward()["target"]: so it is relerr/k with "nothing changes" for the head.
    One extra no-grad target-encoder pass, up to the deepest level a head reads."""
    levels = set(model.head_level.values())
    deepest = max(levels, key=model.level_names.index)
    zs = model.target_encoder(target_stack(frames, model.context_frames, 0), upto=deepest)
    z0 = {lv: normalize_target(zs[lv], model.target_norm) for lv in levels}
    out = {}
    for k in model.horizons:
        target = targets[k]
        sq = _sq_from_err2(_err2(z0[model.head_level[k]], target))
        out[f"relerr/copy/k{k}"] = _relerr_from_sq(sq, target, eps=1e-12)
    return out
