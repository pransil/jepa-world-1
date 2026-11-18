"""Episode files: one compressed .npz per episode, loadable without pickle (SPEC §3.4, §3.5).

Arrays: every GT field, `frames` and `seg` when present, `terrain_x`, `terrain_heights`.
`meta_json` is a 0-d string holding: ball (hidden params + initial state), terrain (params, zmin,
elevation), and meta (seeds, generator_version, sim cfg and its hash, view).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jepa_world.sim.ball import BallParams
from jepa_world.sim.episode import GT_FIELDS, Episode
from jepa_world.sim.terrain import Terrain


def save_episode(path: str | Path, ep: Episode) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    arrays = {name: getattr(ep, name) for name in GT_FIELDS}
    if ep.frames is not None:
        arrays["frames"] = ep.frames
    if ep.seg is not None:
        arrays["seg"] = ep.seg
    meta = {
        "ball": ep.ball.to_dict(),
        "terrain": {
            "params": ep.terrain.params,
            "zmin": ep.terrain.zmin,
            "elevation": ep.terrain.elevation,
        },
        "meta": ep.meta,
    }
    np.savez_compressed(
        path,
        terrain_x=ep.terrain.x,
        terrain_heights=ep.terrain.heights,
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
        **arrays,
    )
    return path


def load_episode(path: str | Path) -> Episode:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        t = meta["terrain"]
        terrain = Terrain(
            x=z["terrain_x"],
            heights=z["terrain_heights"],
            params=t["params"],
            zmin=t["zmin"],
            elevation=t["elevation"],
        )
        return Episode(
            **{name: z[name] for name in GT_FIELDS},
            terrain=terrain,
            ball=BallParams(**meta["ball"]),
            meta=meta["meta"],
            frames=z["frames"] if "frames" in z else None,
            seg=z["seg"] if "seg" in z else None,
        )
