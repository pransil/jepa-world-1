"""Linear probes on frozen embeddings (SPEC §7.2): what each level linearly encodes about the ball's
position and its hidden parameters. Pure numpy + scikit-learn, CPU, no model code.

Probes are fit on trainable splits only (`data.splits.TRAINABLE`) and scored on held-out rows.
They never fit on the test split and never score on a terrain they were fit on (SPEC §3.5, §7.3).
Design: jepa_world/DESIGN.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from jepa_world.data.splits import SPLITS, TRAINABLE, LeakageError
from jepa_world.metrics import SplitResult

LEVELS = ("L1", "L2", "L3")
POOLED = "pooled"  # every fitted level's features, concatenated in name order
TARGETS = ("pos", "friction", "restitution", "radius")  # pos = ball_pos (x, z) at the window's t
DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-3, 3, 13))


def pool(features) -> np.ndarray:
    """(M, C, H, W) → (M, C) global spatial mean, accumulated in float64 without first copying the
    whole map to float64 (L1 on the test split is GBs). (M, C) passes through."""
    a = np.asarray(features)
    if a.ndim == 4:
        return a.mean(axis=(2, 3), dtype=np.float64)
    if a.ndim == 2:
        return a.astype(np.float64, copy=False)
    raise ValueError(f"features {a.shape}: expected (M, C, H, W) or (M, C)")


@dataclass(frozen=True, eq=False)
class ProbeData:
    """One split's rows: features {level: (M, D)}, targets {name: (M,) or (M, d)}, and each row's
    terrain seed (M,), used to refuse scoring on a terrain the probe was fit on."""

    split: str
    features: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    terrain_seed: np.ndarray

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split {self.split!r}: expected one of {list(SPLITS)}")
        if not self.features or not self.targets:
            raise ValueError("ProbeData needs at least one feature level and one target")
        if POOLED in self.features:
            raise ValueError(f"{POOLED!r} is built from the levels; don't pass it as one")
        feats = {}
        for name, f in self.features.items():
            f = np.asarray(f, dtype=float)
            if f.ndim != 2:
                raise ValueError(
                    f"features[{name!r}] {f.shape}: expected (M, D); pool() maps first"
                )
            feats[name] = f
        targets = {}
        for name, y in self.targets.items():
            y = np.asarray(y, dtype=float)
            targets[name] = y.reshape(len(y), -1)
        seeds = np.asarray(self.terrain_seed).ravel()
        rows = {f"features[{k!r}]": len(v) for k, v in feats.items()}
        rows |= {f"targets[{k!r}]": len(v) for k, v in targets.items()}
        rows["terrain_seed"] = len(seeds)
        if len(set(rows.values())) != 1:
            raise ValueError(f"every array needs the same rows (M); got {rows}")
        object.__setattr__(self, "features", feats)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "terrain_seed", seeds)

    @property
    def n(self) -> int:
        return len(self.terrain_seed)

    def levels(self, pooled_from: Sequence[str] = ()) -> dict[str, np.ndarray]:
        """The given levels, plus POOLED, the concatenation of `pooled_from`, when that is set."""
        out = dict(self.features)
        if pooled_from:
            out[POOLED] = np.concatenate([self.features[k] for k in pooled_from], axis=1)
        return out


@dataclass(frozen=True, eq=False)
class Probes:
    """Fitted probes, one StandardScaler + RidgeCV pipeline per (level, target), plus what they
    were fit on, so held-out rows are checked and pooled in exactly the same layout."""

    split: str
    n_train: int
    terrain_seeds: frozenset
    dims: Mapping[str, int]  # level → D, for the given (not pooled) levels
    target_dims: Mapping[str, int]  # target → d
    pooled_from: tuple[str, ...]  # the levels POOLED concatenates, in order; () with one level
    models: Mapping[tuple[str, str], Pipeline]

    def predict(self, level: str, target: str, features) -> np.ndarray:
        """(M, D) → (M, d) predicted target. Always 2-D: sklearn ravels a one-column target."""
        x = np.asarray(features, dtype=float)
        return self.models[(level, target)].predict(x).reshape(len(x), -1)


def fit_probes(train: ProbeData, alphas=DEFAULT_ALPHAS) -> Probes:
    """Fit every (level, target) probe on `train`. Alpha is picked by RidgeCV's leave-one-out on
    these rows only. Refuses any split outside TRAINABLE: nothing is ever fit on test (§7.3)."""
    if train.split not in TRAINABLE:
        raise LeakageError(
            f"refusing to fit probes on the {train.split!r} split; probes fit only on "
            f"{list(TRAINABLE)} and are scored on held-out terrains"
        )
    pooled_from = tuple(sorted(train.features)) if len(train.features) > 1 else ()
    models = {}
    for level, x in train.levels(pooled_from).items():
        for name, y in train.targets.items():
            pipe = make_pipeline(
                StandardScaler(), RidgeCV(alphas=tuple(alphas), alpha_per_target=True)
            )
            models[(level, name)] = pipe.fit(x, y)
    return Probes(
        split=train.split,
        n_train=train.n,
        terrain_seeds=frozenset(int(s) for s in train.terrain_seed),
        dims={k: v.shape[1] for k, v in train.features.items()},
        target_dims={k: v.shape[1] for k, v in train.targets.items()},
        pooled_from=pooled_from,
        models=models,
    )


def _check_layout(probes: Probes, held_out: ProbeData) -> None:
    for level, d in probes.dims.items():
        if level not in held_out.features:
            raise ValueError(f"held-out data lacks level {level!r}, which the probes were fit on")
        if held_out.features[level].shape[1] != d:
            raise ValueError(
                f"level {level!r}: held-out D={held_out.features[level].shape[1]}, "
                f"but the probes were fit on D={d}"
            )
    for name, d in probes.target_dims.items():
        if name not in held_out.targets:
            raise ValueError(f"held-out data lacks target {name!r}, which the probes were fit on")
        if held_out.targets[name].shape[1] != d:
            raise ValueError(
                f"target {name!r}: held-out has {held_out.targets[name].shape[1]} dims, "
                f"the probes {d}"
            )


def score_probes(probes: Probes, held_out: ProbeData) -> SplitResult:
    """Predict-only scoring of every fitted probe on held-out rows → SplitResult(held_out.split,
    {"probes": ...}).

    Per (level, target): r2 (mean over the target's dims), r2_per_dim, rmse_per_dim (target
    units). A held-out target with zero variance scores r2 = 0 unless predicted exactly
    (sklearn's force_finite). Held-out levels the probes weren't fit on are ignored, and POOLED
    is rebuilt from the levels the probes were fit on, in the same order."""
    if held_out.split == probes.split:
        raise LeakageError(f"held-out rows come from the probes' own split {probes.split!r}")
    shared = probes.terrain_seeds & {int(s) for s in held_out.terrain_seed}
    if shared:
        raise LeakageError(
            f"held-out rows share {len(shared)} terrain seed(s) with the probes' training rows, "
            f"e.g. {sorted(shared)[:5]}"
        )
    if held_out.n < 2:
        raise ValueError("scoring needs at least 2 held-out rows")
    _check_layout(probes, held_out)
    levels = held_out.levels(probes.pooled_from)
    by_level: dict[str, dict] = {}
    for level, name in probes.models:
        y = held_out.targets[name]
        pred = probes.predict(level, name, levels[level])
        r2 = r2_score(y, pred, multioutput="raw_values")
        rmse = np.sqrt(np.mean((y - pred) ** 2, axis=0))
        by_level.setdefault(level, {})[name] = {
            "r2": float(np.mean(r2)),
            "r2_per_dim": [float(v) for v in r2],
            "rmse_per_dim": [float(v) for v in rmse],
        }
    return SplitResult(
        held_out.split,
        {
            "probes": {
                "train_split": probes.split,
                "n_train": probes.n_train,
                "n_held_out": held_out.n,
                "pooled_from": list(probes.pooled_from),
                "by_level": by_level,
            }
        },
    )


def run_probes(train: ProbeData, held_out: ProbeData, alphas=DEFAULT_ALPHAS) -> SplitResult:
    """fit_probes on train, then score_probes on held_out."""
    return score_probes(fit_probes(train, alphas), held_out)
