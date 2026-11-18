"""Terrain-seed splits (SPEC §3.5): which (terrain_seed, ball_seed) episodes each split holds. Pure.

Config (`data.splits.<split>`): `terrain_seeds: [first, count]`, `ball_seeds: [first, count]`.
Every terrain of a split gets every ball seed of that split. Splits are by terrain, never by frame.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping

SPLITS = ("dev", "train", "test")
TRAINABLE = ("dev", "train")  # never test: its terrains are held out (SPEC §3.5, §7.3)


class LeakageError(ValueError):
    """Data from one split could reach another, or a context could see its own future."""


def _split_cfg(data_cfg: dict, split: str) -> dict:
    if split not in SPLITS:
        raise ValueError(f"split {split!r}: valid splits are {list(SPLITS)}")
    splits = data_cfg.get("splits") or {}
    if split not in splits:
        raise ValueError(f"data.splits has no {split!r} entry (needs all of {list(SPLITS)})")
    return splits[split]


def _seed_range(data_cfg: dict, split: str, key: str) -> list[int]:
    spec = _split_cfg(data_cfg, split).get(key)
    ok = (
        isinstance(spec, list | tuple)
        and len(spec) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in spec)
        and spec[0] >= 0
        and spec[1] >= 1
    )
    if not ok:
        raise ValueError(
            f"data.splits.{split}.{key} = {spec!r}: expected [first, count], "
            "integers with first >= 0 and count >= 1"
        )
    first, count = spec
    return list(range(first, first + count))


def terrain_seeds(data_cfg: dict, split: str) -> list[int]:
    return _seed_range(data_cfg, split, "terrain_seeds")


def ball_seeds(data_cfg: dict, split: str) -> list[int]:
    return _seed_range(data_cfg, split, "ball_seeds")


def split_episodes(data_cfg: dict, split: str) -> list[tuple[int, int]]:
    """(terrain_seed, ball_seed) for every episode of `split`, terrain-major."""
    return list(itertools.product(terrain_seeds(data_cfg, split), ball_seeds(data_cfg, split)))


def check_disjoint(seeds_by_split: Mapping[str, Iterable], what: str = "terrain seed") -> None:
    """Raise LeakageError naming the shared values and both splits if any two groups overlap."""
    sets = {name: set(values) for name, values in seeds_by_split.items()}
    for a, b in itertools.combinations(sets, 2):
        shared = sets[a] & sets[b]
        if shared:
            some = sorted(shared)[:10]
            raise LeakageError(
                f"splits {a!r} and {b!r} share {len(shared)} {what}(s): {some}"
                f"{' ...' if len(shared) > 10 else ''}"
            )


def check_split_config(data_cfg: dict) -> None:
    """The config-level leakage check: no terrain seed in two splits, and no ball seed shared by
    train and test (docs/decisions/2026-11-17-test-split-ball-params.md)."""
    check_disjoint({s: terrain_seeds(data_cfg, s) for s in SPLITS}, "terrain seed")
    check_disjoint({s: ball_seeds(data_cfg, s) for s in ("train", "test")}, "ball seed")
