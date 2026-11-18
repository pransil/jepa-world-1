"""Evaluation metrics (SPEC §7.2) and split tags (SPEC §7.3). Pure numpy: arrays in, numbers out.

Positions are world (x, z) in metres. A horizon dict maps k to an (M, 2) array over the same M
windows at every k (M2's rule, data/DESIGN.md), which is the shape `baselines.predict` returns.
Every output is plain Python (str keys; float, int or None), so it serialises with `json.dumps`.
Design: jepa_world/DESIGN.md.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from jepa_world.data.splits import SPLITS

COVERAGE_LEVELS = (0.5, 0.9)  # SPEC §7.2: "the 50% and 90% predicted intervals"
EVENT_WINDOW = 5  # frames before and after an event onset or offset (1/3 s at 15 fps)
HEADLINE_SPLIT = "test"  # SPEC §7.3: headline numbers come from held-out terrain seeds only


# --- position error, stats, ratio ---------------------------------------------------------------


def _positions(a, what: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim < 1 or a.shape[-1] != 2:
        raise ValueError(f"{what}: expected world (x, z) positions (..., 2), got {a.shape}")
    return a


def position_error(pred, truth) -> np.ndarray:
    """(..., 2), (..., 2) → (...,) Euclidean distance in world units."""
    pred, truth = _positions(pred, "pred"), _positions(truth, "truth")
    if pred.shape != truth.shape:
        raise ValueError(f"pred {pred.shape} and truth {truth.shape} differ")
    return np.linalg.norm(pred - truth, axis=-1)


def error_stats(err) -> dict:
    """The same summary `baselines.evaluate` reports per horizon."""
    err = np.asarray(err, dtype=float)
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p90": float(np.quantile(err, 0.9)),
    }


def _int_keys(d: Mapping, what: str) -> dict[int, object]:
    """Horizon keys as ints. This module emits str(k) keys and JSON reloads give '10', so both
    1 and '1' are accepted, but not both in one dict."""
    out = {}
    for k, v in d.items():
        kk = int(k)
        if kk in out:
            raise ValueError(f"{what}: horizon {kk} appears twice")
        out[kk] = v
    return out


def _align(**named: Mapping) -> tuple[list[int], dict[str, dict[int, np.ndarray]]]:
    """The shared, sorted horizon set and the dicts re-keyed by int k. Raises unless every dict
    has the same horizons and every array covers the same M >= 1 windows, at every k."""
    out = {name: _int_keys(d, name) for name, d in named.items()}
    first = next(iter(out))
    ks = sorted(out[first])
    if not ks:
        raise ValueError(f"{first}: no horizons")
    windows = set()
    for name, d in out.items():
        if sorted(d) != ks:
            raise ValueError(f"horizons differ: {first} has {ks}, {name} has {sorted(d)}")
        for k, v in d.items():
            if np.ndim(v) == 0:
                raise ValueError(f"{name}[{k}] is a scalar; expected an array over the windows")
            windows.add(np.shape(v)[0])
    if len(windows) != 1:
        raise ValueError(
            f"every horizon must be scored on the same windows; got window counts {sorted(windows)}"
        )
    if windows == {0}:
        raise ValueError("no windows to score (M = 0)")
    return ks, out


def errors_by_horizon(pred: Mapping[int, np.ndarray], truth: Mapping[int, np.ndarray]) -> dict:
    """{k: (M,2)} ×2 → {str(k): {mean, median, p90}}."""
    ks, d = _align(pred=pred, truth=truth)
    return {str(k): error_stats(position_error(d["pred"][k], d["truth"][k])) for k in ks}


def ratio_to_persistence(
    model: Mapping[int, np.ndarray],
    persistence: Mapping[int, np.ndarray],
    truth: Mapping[int, np.ndarray],
) -> dict:
    """{str(k): mean model error / mean persistence error}, or None where persistence is exact.
    Below 1.0 means the model beats persistence (SPEC §7.2). Same rule as `baselines.evaluate`."""
    ks, d = _align(model=model, persistence=persistence, truth=truth)
    model, persistence, truth = d["model"], d["persistence"], d["truth"]
    out = {}
    for k in ks:
        p = float(position_error(persistence[k], truth[k]).mean())
        m = float(position_error(model[k], truth[k]).mean())
        out[str(k)] = m / p if p > 0 else None
    return out


# --- calibration --------------------------------------------------------------------------------


def _levels(levels: Iterable[float]) -> dict[str, float]:
    """{key: p}, keyed by p to 6 significant digits ('0.5', '0.9'). Raises on p outside (0, 1)
    and on two levels with the same key, which would otherwise overwrite each other."""
    out = {}
    for p in levels:
        p = float(p)
        if not 0.0 < p < 1.0:
            raise ValueError(f"coverage level {p} must lie in (0, 1)")
        key = f"{p:g}"
        if key in out:
            raise ValueError(f"coverage levels {out[key]} and {p} share the key {key!r}")
        out[key] = p
    if not out:
        raise ValueError("no coverage levels")
    return out


def coverage(mean, var, truth, levels: Iterable[float] = COVERAGE_LEVELS) -> dict:
    """Fraction of truths inside the central p-intervals of a diagonal Gaussian (M,2).

    Per axis: |y − μ| ≤ Φ⁻¹((1+p)/2)·σ. Joint: the 2-D p-region, Σ (y − μ)²/σ² ≤ −2 ln(1 − p),
    the closed-form χ² quantile with 2 degrees of freedom. A calibrated model gives ≈ p for each.
    """
    levels = _levels(levels)
    mean, truth = _positions(mean, "mean"), _positions(truth, "truth")
    var = _positions(var, "var")
    if not (mean.shape == var.shape == truth.shape):
        raise ValueError(f"mean {mean.shape}, var {var.shape}, truth {truth.shape} differ")
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        raise ValueError("var must be finite and > 0 (pass exp(log_var), not log_var)")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(truth))):
        raise ValueError("mean and truth must be finite")
    z = np.abs(truth - mean) / np.sqrt(var)  # (M, 2) standardised residuals
    r2 = np.sum(z**2, axis=-1)
    out = {}
    for key, p in levels.items():
        half = NormalDist().inv_cdf((1.0 + p) / 2.0)
        inside = z <= half
        out[key] = {
            "x": float(inside[..., 0].mean()),
            "z": float(inside[..., 1].mean()),
            "joint": float(np.mean(r2 <= -2.0 * math.log(1.0 - p))),
        }
    return out


# --- rollout drift ------------------------------------------------------------------------------


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc**2))


def rollout_drift(mean_error: Mapping[int, float]) -> dict:
    """How error grows across horizons.

    - growth[k] = e_k / e_kmin, relative to the smallest horizon (None if e_kmin is 0);
    - slope_m_per_frame: least-squares slope of e against k (None with one horizon);
    - exponent: least-squares slope of log e against log k, so e ∝ k^exponent
      (None with one horizon or any e ≤ 0).
    """
    mean_error = _int_keys(mean_error, "mean_error")
    ks = sorted(mean_error)
    if not ks:
        raise ValueError("no horizons")
    k = np.array(ks, dtype=float)
    e = np.array([float(mean_error[kk]) for kk in ks])
    base = e[0]
    multi = len(ks) > 1
    return {
        "horizons": ks,
        "error": {str(kk): float(v) for kk, v in zip(ks, e, strict=True)},
        "growth": {
            str(kk): (float(v / base) if base > 0 else None) for kk, v in zip(ks, e, strict=True)
        },
        "slope_m_per_frame": _ols_slope(k, e) if multi else None,
        "exponent": _ols_slope(np.log(k), np.log(e)) if multi and np.all(e > 0) else None,
    }


# --- uncertainty series and events --------------------------------------------------------------


def predictive_std(var) -> np.ndarray:
    """(..., 2) diagonal variances → (...,) RMS σ, √(mean of the variances), in metres."""
    var = _positions(var, "var")
    if not np.all(np.isfinite(var)) or np.any(var < 0):
        raise ValueError("var must be finite and >= 0")
    return np.sqrt(var.mean(axis=-1))


def to_series(values, ep, t, n: int, T: int) -> np.ndarray:  # noqa: N803 (T is the episode length)
    """Per-window values (M,) at windows (ep[i], t[i]) → an (n, T) series, NaN where unscored."""
    values = np.asarray(values, dtype=float).ravel()
    ep, t = np.asarray(ep, dtype=int).ravel(), np.asarray(t, dtype=int).ravel()
    if not (len(values) == len(ep) == len(t)):
        raise ValueError(f"values {len(values)}, ep {len(ep)}, t {len(t)} differ in length")
    if len(ep) and (ep.min() < 0 or ep.max() >= n or t.min() < 0 or t.max() >= T):
        raise ValueError(f"a window (ep, t) is out of range for n={n}, T={T}")
    flat = ep * T + t
    if len(np.unique(flat)) != len(flat):
        raise ValueError("duplicate (ep, t) windows")
    out = np.full((n, T), np.nan)
    out[ep, t] = values
    return out


def _nanmean(a: np.ndarray, axis=None):
    """Mean of the finite entries and their count, without numpy's empty-slice warnings."""
    ok = np.isfinite(a)
    count = ok.sum(axis=axis)
    total = np.where(ok, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count, count


def _locked(series: np.ndarray, at: np.ndarray, window: int) -> dict:
    """Event-locked summary over events at (episode, t) rows of `at` (E, 2)."""
    offsets = np.arange(-window, window)
    n_ep, T = series.shape  # noqa: N806
    if len(at):
        idx = at[:, 1:2] + offsets[None, :]  # (E, 2W)
        inside = (idx >= 0) & (idx < T)
        vals = np.where(inside, series[at[:, :1], np.clip(idx, 0, T - 1)], np.nan)
    else:
        vals = np.full((0, 2 * window), np.nan)
    curve, count = _nanmean(vals, axis=0)
    before, nb = _nanmean(vals[:, :window], axis=1)
    after, na = _nanmean(vals[:, window:], axis=1)
    paired = (nb > 0) & (na > 0)
    mb, cb = _nanmean(vals[:, :window])
    ma, ca = _nanmean(vals[:, window:])
    return {
        "n_events": int(len(at)),
        "offsets": offsets.tolist(),
        "curve": [float(c) if n else None for c, n in zip(curve, count, strict=True)],
        "n_per_offset": [int(n) for n in count],
        "mean_before": float(mb) if cb else None,
        "mean_after": float(ma) if ca else None,
        "delta": float(np.mean(after[paired] - before[paired])) if paired.any() else None,
        "n_paired": int(paired.sum()),
    }


def event_response(series, flag, window: int = EVENT_WINDOW) -> dict:
    """Uncertainty around the events of one flag (SPEC §7.2 "event behavior").

    series: (N, T) float, the predicted uncertainty indexed by the context's last frame t (NaN
    where no window was scored). flag: (N, T) bool, e.g. `out_of_view`.
    - onset: flag goes False → True at t ≥ 1 (the ball disappears). Expect delta > 0.
    - offset: flag goes True → False (the ball reappears). Expect delta < 0.
    "before" is offsets −W..−1 and "after" is 0..W−1 from the event frame. `delta` is the mean over
    events of each event's own after − before. With no events, everything but the counts is None.
    """
    series = np.asarray(series, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    if series.ndim != 2 or series.shape != flag.shape:
        raise ValueError(f"series {series.shape} and flag {flag.shape} must be the same (N, T)")
    if int(window) < 1:
        raise ValueError(f"window {window} must be >= 1")
    prev, cur = flag[:, :-1], flag[:, 1:]
    onset = np.argwhere(cur & ~prev) + [0, 1]
    offset = np.argwhere(~cur & prev) + [0, 1]
    return {
        "onset": _locked(series, onset, int(window)),
        "offset": _locked(series, offset, int(window)),
    }


def event_behaviour(series, flags: Mapping[str, np.ndarray], window: int = EVENT_WINDOW) -> dict:
    """`event_response` for each flag, e.g. {"out_of_view": ..., "occluded": ...}."""
    return {name: event_response(series, f, window) for name, f in flags.items()}


def by_visibility(mean, var, truth, hidden, levels: Iterable[float] = COVERAGE_LEVELS) -> dict:
    """Uncertainty, error and calibration on hidden versus visible windows, at one horizon.

    mean, var, truth: (M, 2). hidden: (M,) bool, e.g. `out_of_view | occluded` at the window's
    t. `std_ratio` = mean predictive_std on hidden windows / on visible ones: above 1 means the
    model knows it can't see the ball. A side with no windows (`occluded` until M7) gets None
    for its numbers, and `has_hidden` says so. It never raises or warns over it."""
    levels = tuple(_levels(levels).values())
    mean, var, truth = _positions(mean, "mean"), _positions(var, "var"), _positions(truth, "truth")
    if not (mean.shape == var.shape == truth.shape) or mean.ndim != 2:
        raise ValueError(f"mean {mean.shape}, var {var.shape}, truth {truth.shape}: need (M, 2)")
    hidden = np.asarray(hidden, dtype=bool)
    if hidden.shape != (len(mean),):
        raise ValueError(f"hidden {hidden.shape}: expected ({len(mean)},), one per window")
    std = predictive_std(var)
    err = position_error(mean, truth)
    out: dict = {"n_hidden": int(hidden.sum()), "n_visible": int((~hidden).sum())}
    out["has_hidden"] = out["n_hidden"] > 0
    for side, rows in (("hidden", hidden), ("visible", ~hidden)):
        any_rows = bool(rows.any())
        out[f"std_{side}"] = float(std[rows].mean()) if any_rows else None
        out[f"error_m_{side}"] = float(err[rows].mean()) if any_rows else None
        out[f"coverage_{side}"] = (
            coverage(mean[rows], var[rows], truth[rows], levels) if any_rows else None
        )
    sh, sv = out["std_hidden"], out["std_visible"]
    out["std_ratio"] = sh / sv if sh is not None and sv else None
    return out


def visibility(
    mean, var, truth, flags: Mapping[str, np.ndarray], levels: Iterable[float] = COVERAGE_LEVELS
) -> dict:
    """`by_visibility` for each flag, e.g. {"out_of_view": (M,), "occluded": (M,)}, and for
    "hidden", their union (the ball can't be seen for either reason)."""
    if "hidden" in flags:
        raise ValueError("'hidden' is the union of the flags; don't pass it as one")
    if not flags:
        raise ValueError("no flags")
    levels = tuple(_levels(levels).values())
    masks = {name: np.asarray(f, dtype=bool) for name, f in flags.items()}
    shapes = {m.shape for m in masks.values()}
    if len(shapes) != 1:
        raise ValueError(f"flags differ in shape: {sorted(shapes)}")
    out = {name: by_visibility(mean, var, truth, m, levels) for name, m in masks.items()}
    out["hidden"] = by_visibility(
        mean, var, truth, np.logical_or.reduce(list(masks.values())), levels
    )
    return out


# --- assembly -----------------------------------------------------------------------------------


def evaluate_positions(
    mean: Mapping[int, np.ndarray],
    truth: Mapping[int, np.ndarray],
    persistence: Mapping[int, np.ndarray],
    var: Mapping[int, np.ndarray] | None = None,
    out_of_view: Mapping[int, np.ndarray] | None = None,
    levels: Iterable[float] = COVERAGE_LEVELS,
) -> dict:
    """Every position metric of SPEC §7.2 for a goal head's mean (and variance, if it has one).

    All dicts map k (int, or str as this module emits) to arrays over the same M windows.
    `out_of_view[k]` (M,) bool is the flag at t + k, which gives the same split as
    `baselines.evaluate`. Report it as one family of a split's metrics, e.g.
    `SplitResult("test", {"goal_head": evaluate_positions(...)})`."""
    levels = tuple(_levels(levels).values())  # once: a generator would be spent after k_min
    named = {"mean": mean, "truth": truth, "persistence": persistence}
    if var is not None:
        named["var"] = var
    if out_of_view is not None:
        named["out_of_view"] = out_of_view
    ks, d = _align(**named)
    mean, truth, persistence = d["mean"], d["truth"], d["persistence"]
    m = int(np.shape(mean[ks[0]])[0])
    errors = errors_by_horizon(mean, truth)
    out = {
        "windows": m,
        "horizons": ks,
        "error_m": errors,
        "ratio_to_persistence": ratio_to_persistence(mean, persistence, truth),
        "drift": rollout_drift({k: errors[str(k)]["mean"] for k in ks}),
    }
    if var is not None:
        out["calibration"] = {str(k): coverage(mean[k], d["var"][k], truth[k], levels) for k in ks}
    if out_of_view is not None:
        by_view = {}
        for k in ks:
            e = position_error(mean[k], truth[k])
            gone = np.asarray(d["out_of_view"][k], dtype=bool)
            if gone.shape != (m,):
                raise ValueError(f"out_of_view[{k}] {gone.shape}: expected ({m},), one per window")
            by_view[str(k)] = {
                "in_view": float(e[~gone].mean()) if (~gone).any() else None,
                "out_of_view": float(e[gone].mean()) if gone.any() else None,
                "n_out_of_view": int(gone.sum()),
            }
        out["mean_error_m_by_view_at_t_plus_k"] = by_view
    return out


# --- split tags (SPEC §7.3) ---------------------------------------------------------------------


@dataclass(frozen=True)
class SplitResult:
    """Metrics tagged with the split they were measured on. Only the test split is a headline."""

    split: str
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split {self.split!r}: expected one of {list(SPLITS)}")

    @property
    def headline(self) -> bool:
        return self.split == HEADLINE_SPLIT


def report(results: Iterable[SplitResult]) -> dict:
    """{"headline": {"split": "test", "metrics": ...} or None, "development": {split: metrics}
    for every other split}. Dev numbers are reported separately and never as a headline
    (SPEC §7.3). Several results for one split (e.g. goal-head metrics and probes) are merged,
    and two results reporting the same top-level key raise rather than overwrite."""
    merged: dict[str, dict] = {}
    for r in results:
        into = merged.setdefault(r.split, {})
        clash = set(into) & set(r.metrics)
        if clash:
            raise ValueError(f"split {r.split!r}: two results both report {sorted(clash)}")
        into.update(r.metrics)
    headline = None
    if HEADLINE_SPLIT in merged:
        headline = {"split": HEADLINE_SPLIT, "metrics": merged.pop(HEADLINE_SPLIT)}
    return {"headline": headline, "development": merged}
