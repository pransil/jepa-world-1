"""YAML experiment configs with single inheritance and dotted command-line overrides.

    # configs/near_heads.yaml
    base: base.yaml            # path relative to this file; merged first, then this file on top
    train: {batch_size: 64}

    load_config("configs/near_heads.yaml", ["train.lr=3e-4", "model.ema_momentum=0.999"])

Values in files and overrides are parsed as YAML, so `3e-4`, `true` and `[1, 2]` get their
natural types. Plain PyYAML (YAML 1.1) reads `3e-4` as the STRING "3e-4" because it has no
decimal point. That would make AdamW raise, or worse, compare as text. `_Loader` fixes it.
Unknown keys in an override raise an error: a typo should fail loudly, not start a run that
silently ignored it.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads 1e-3 / 3E+4 (no decimal point) as floats."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def parse_yaml(text: str):
    return yaml.load(text, Loader=_Loader)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config inheritance cycle: {' -> '.join(map(str, (*seen, path)))}")
    cfg = parse_yaml(path.read_text()) or {}
    parent = cfg.pop("base", None)
    if parent is None:
        return cfg
    return deep_merge(_load(path.parent / parent, (*seen, path)), cfg)


def apply_override(cfg: dict, dotted: str) -> None:
    key, sep, raw = dotted.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got {dotted!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        if not isinstance(node.get(p), dict):
            raise KeyError(f"override {key!r}: no section {p!r}")
        node = node[p]
    if leaf not in node:
        raise KeyError(f"override {key!r}: unknown key {leaf!r} (known: {sorted(node)})")
    node[leaf] = parse_yaml(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = _load(Path(path))
    for o in overrides or []:
        apply_override(cfg, o)
    return cfg
