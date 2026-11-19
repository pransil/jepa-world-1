"""The pred/actual viewer: one self-contained HTML page that plays held-out episodes frame by
frame, with each model's goal-head predictions beside the ball's true motion and its pixel
decoder's frame beside the real one. Design: jepa_world/DESIGN.md, "The pred/actual viewer".

  uv run python -m jepa_world.viewer --model "24k headline=<run dir>@24000" \\
      --model "48k headline=<run dir>" [--split test] [--episodes 24] [--device cpu] [--out f.html]

Only dev or test, never a split any of the models trained on (eval.check_eval_split). The page is
written under $JEPA_ARTIFACTS/viewers/ unless --out says otherwise, and never into the repo.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import math
import secrets
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jepa_world import eval as eval_cli
from jepa_world import metrics
from jepa_world.data.build import split_dir
from jepa_world.data.dataset import load_split, read_manifest, valid_starts
from jepa_world.data.splits import terrain_seeds
from jepa_world.sim import camera
from jepa_world.sim.io import load_episode
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import artifacts_root, write_new
from jepa_world.utils.repro import git_info

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("viewer_template.html")
PAYLOAD_SLOT = "__PAYLOAD__"
MAX_BYTES = 16 * 1024 * 1024  # the Artifact page limit, data URIs included
DEFAULT_EPISODES = 24
EVENTS = ("contact", "bounce", "out_of_view", "occluded")


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    label: str  # what the page calls the model
    run_dir: Path
    step: int | None  # None: the run's latest checkpoint


def parse_model_spec(text: str) -> ModelSpec:
    """'LABEL=RUN_DIR' or 'LABEL=RUN_DIR@STEP'."""
    label, sep, rest = text.partition("=")
    step = None
    if sep and "@" in rest:
        rest, _, step = rest.rpartition("@")
        step = int(step) if step.isdigit() else -1
    if not sep or not label.strip() or not rest or step == -1:
        raise ValueError(f"--model {text!r}: expected LABEL=RUN_DIR or LABEL=RUN_DIR@STEP")
    return ModelSpec(label.strip(), Path(rest), step)


def pick_episodes(n_total: int, n: int) -> np.ndarray:
    """n episode indices evenly spaced over [0, n_total), first and last included. Chosen by
    position in the manifest only, never by anything a model output (no cherry-picking)."""
    if n < 1:
        raise ValueError(f"--episodes must be >= 1, got {n}")
    if n >= n_total:
        return np.arange(n_total)
    return np.round(np.linspace(0, n_total - 1, n)).astype(np.int64)


def goal_track(pos, t, mean, logvar) -> tuple[np.ndarray, np.ndarray]:
    """One episode's positions pos (T, 2), windows t (W,), the goal head's displacement mean and
    log-variance (W, K, 2) -> predicted positions pred (W, K, 2) = pos[t] + mean, and the
    per-axis standard deviation std (W, K, 2) = exp(logvar / 2). eval.goal_positions' numbers."""
    pos, t = np.asarray(pos, dtype=np.float64), np.asarray(t)
    mean, logvar = np.asarray(mean, dtype=np.float64), np.asarray(logvar, dtype=np.float64)
    if mean.shape != logvar.shape or mean.ndim != 3 or mean.shape[2] != 2 or len(t) != len(mean):
        raise ValueError(
            f"shape mismatch: t {t.shape}, mean {mean.shape}, logvar {logvar.shape} "
            "(expected (W,), (W, K, 2), (W, K, 2))"
        )
    return pos[t][:, None, :] + mean, np.exp(0.5 * logvar)


def region_scales(levels: Sequence[float]) -> list[dict]:
    """For each coverage level p, the factor s with which the ellipse (s * std_x, s * std_z) is
    the joint region metrics.coverage counts: sum of squared z-scores <= -2 ln(1 - p)."""
    out = []
    for p in levels:
        if not 0 < p < 1:
            raise ValueError(f"coverage level {p!r} is not in (0, 1)")
        out.append({"p": float(p), "scale": math.sqrt(-2.0 * math.log(1.0 - p))})
    return out


def to_uint8(images) -> np.ndarray:
    """(W, 3, S, S) float in [0, 1] -> (W, S, S, 3) uint8: clipped, rounded half up."""
    x = np.clip(np.asarray(images, dtype=np.float64), 0.0, 1.0)
    return np.floor(x * 255.0 + 0.5).astype(np.uint8).transpose(0, 2, 3, 1)


def png_strip(frames) -> str:
    """(n, S, S, 3) uint8 -> one lossless PNG, the frames side by side, as a data URI."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"expected (n, S, S, 3) uint8 frames, got {frames.dtype} {frames.shape}")
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(list(frames), axis=1)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def eval_summary(run_dir, checkpoint, split: str) -> dict | None:
    """The newest `eval --run` report in run_dir/eval/ for exactly this checkpoint (by file name;
    eval only scores a run's own checkpoints) that scored `split`, reduced to what the page shows.
    None if there is none: the page then says so instead of showing another checkpoint's numbers.

    Same test set by construction: `eval --run` and the viewer both refuse an eval config whose
    split seeds differ from the run's own (eval.check_eval_split). The coverage columns carry the
    levels that eval used, as their labels, whatever the page's own region levels are."""
    evals = sorted((Path(run_dir) / "eval").glob("*.json"))
    for path in reversed(evals):  # names start with the UTC time: newest first
        doc = json.loads(path.read_text())
        if Path(doc["meta"]["checkpoint"]).name != Path(checkpoint).name:
            continue
        report = doc["report"]
        if split == "test":
            head = report.get("headline")
            found = head["metrics"] if head and head["split"] == "test" else None
        else:
            found = report.get("development", {}).get(split)
        if not found or "goal_head" not in found:
            continue
        goal, base = found["goal_head"], found["baselines"]
        others = goal.get("ratio_to_baseline", {})
        by_k = {}
        for k in goal["horizons"]:
            ks = str(k)
            by_k[ks] = {
                "error_m": goal["error_m"][ks]["mean"],
                "ratio_to_persistence": goal["ratio_to_persistence"][ks],
                "ratio_to_const_velocity": others.get("const_velocity", {}).get(ks),
                "ratio_to_simulator_mean": others.get("simulator_mean", {}).get(ks),
                "coverage": {p: c["joint"] for p, c in goal["calibration"][ks].items()},
            }
        pooled = found.get("probes", {}).get("by_level", {}).get("pooled", {})
        probe_r2 = {t: pooled[t]["r2"] for t in ("pos", "pos_cam") if t in pooled} or None
        return {
            "file": path.name,
            "split": split,
            "headline": split == "test",
            "episodes": base["episodes"],
            "windows": base["windows"],
            "starts": base["starts"],
            "by_k": by_k,
            "probe_r2": probe_r2,
        }
    return None


def render_html(payload: dict) -> str:
    """The page: the template with `payload` as JSON in its application/json script. Every '<' is
    escaped (as \\u003c, still valid JSON), so no string in the data can close the script."""
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PAYLOAD_SLOT) != 1:
        raise ValueError(f"{TEMPLATE} must hold {PAYLOAD_SLOT} exactly once")
    return template.replace(PAYLOAD_SLOT, text)


# --- the exporter -------------------------------------------------------------------------------


def _round(a, decimals: int) -> list:
    """Nested lists of floats rounded to `decimals`, NaN as None (JSON null)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isnan(a), None, np.round(a, decimals)).tolist()


def _sig(a, digits: int) -> list:
    """Nested lists of floats to `digits` significant digits, NaN as None."""
    a = np.asarray(a, dtype=np.float64)
    flat = [None if math.isnan(x) else float(f"{x:.{digits}g}") for x in a.ravel().tolist()]
    return np.array(flat, dtype=object).reshape(a.shape).tolist()


def _seed_note(cfg: dict, split: str, n: int) -> str:
    seeds = terrain_seeds(cfg["data"], split)
    return (
        f"{split} split: {n} episodes on terrain seeds {min(seeds)}–{max(seeds)}, "
        "held out from every model's training split (checked)"
    )


def _episode_payload(es, i: int, row: dict, path: Path) -> tuple[dict, dict]:
    """What the page needs from episode i, and the episode file's sim config."""
    ep = load_episode(path / row["file"])
    if not np.array_equal(ep.ball_pos, es.ball_pos[i]):
        raise ValueError(f"{path / row['file']}: positions differ from the loaded split's row {i}")
    sim = ep.meta["sim_cfg"]
    return {
        "id": Path(row["file"]).stem,
        "terrain_seed": int(es.terrain_seed[i]),
        "ball_seed": int(es.ball_seed[i]),
        "T": int(es.T),
        "radius": float(ep.ball.radius),
        "pos": _round(es.ball_pos[i], 5),
        "cam": _round(es.cam_pos[i], 5),
        "px": _round(camera.project(es.ball_pos[i], es.cam_pos[i], sim), 3),
        "events": {name: np.flatnonzero(getattr(ep, name)).tolist() for name in EVENTS},
        "terrain": {"x": _round(ep.terrain.x, 4), "z": _round(ep.terrain.heights, 5)},
        "frames": png_strip(es.frames[i].permute(0, 2, 3, 1).numpy()),
    }, sim


def _model_episode(run, es, i: int, t, out: dict, sim: dict, scored_hi: int) -> dict:
    """One model's outputs on episode i's windows t (every t >= C-1), for the page."""
    e, T = {}, es.T
    if out["goal_mean"] is not None:
        ks = list(run.cfg["model"]["goal_horizons"])
        pos = es.ball_pos[i]
        pred, std = goal_track(pos, t, out["goal_mean"], out["goal_logvar"])
        cam_t = es.cam_pos[i][t][:, None]  # frame t's camera, for every horizon
        idx = t[:, None] + np.array(ks)[None]
        inside = idx <= T - 1
        truth = np.full(pred.shape, np.nan)
        truth[inside] = pos[idx[inside]]
        scored = t <= scored_hi  # the windows eval scores (t >= C-1 already)
        g = eval_cli.goal_positions(
            es.ball_pos, np.full(scored.sum(), i), t[scored], out["goal_mean"][scored],
            out["goal_logvar"][scored], ks,
        )  # fmt: skip
        e["pred"] = _round(pred, 5)
        e["std"] = _sig(std, 4)
        e["pred_px"] = _round(camera.project(pred, cam_t, sim), 3)
        e["true_px"] = _round(camera.project(truth, cam_t, sim), 3)
        e["errors"] = {
            str(k): {
                "model": float(metrics.position_error(g["pred"][k], g["truth"][k]).mean()),
                "persistence": float(
                    metrics.position_error(g["persistence"][k], g["truth"][k]).mean()
                ),
            }
            for k in ks
        }
    if out["pixel"] is not None:
        pix = out["pixel"]  # (W, 3, S, S): the decoder's frame t+1
        real = es.frames[i].float().div(255.0).numpy()
        nxt = t + 1 <= T - 1
        tail = [None] * int((~nxt).sum())
        mae = np.abs(pix[nxt] - real[t[nxt] + 1]).mean(axis=(1, 2, 3))
        copy = np.abs(real[t[nxt]] - real[t[nxt] + 1]).mean(axis=(1, 2, 3))
        e["pixel"] = png_strip(to_uint8(pix))
        e["pixel_mae"] = _sig(mae, 4) + tail
        e["copy_mae"] = _sig(copy, 4) + tail
    return e


def _in_git_work_tree(path: Path) -> bool:
    """Whether `path` (which may not exist yet) lies in any git work tree: this repo, the main
    clone of a lane worktree, or another repository."""
    if path.resolve().is_relative_to(REPO_ROOT):
        return True
    d = path.resolve().parent
    while not d.exists():
        d = d.parent
    try:
        r = subprocess.run(
            ["git", "-C", str(d), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return False  # no git to ask: nothing to commit the page into either
    return r.stdout.strip() == "true"


def export(
    specs: Sequence[ModelSpec],
    split: str,
    n_episodes: int,
    eval_cfg: dict,
    *,
    out: str | Path | None = None,
    device: str = "cpu",
    max_bytes: int = MAX_BYTES,
) -> tuple[Path, dict]:
    """Every model's goal-head and pixel-decoder outputs on n_episodes episodes of a held-out
    split, as one HTML page. eval_cfg supplies the canonical split seeds (as for `eval --run`),
    the coverage levels and the batch size. Returns the page's path and its payload."""
    if split not in eval_cli.HELD_OUT:  # before any checkpoint is read
        raise ValueError(f"split {split!r}: the viewer shows dev or test (held out) only")
    if out is not None and _in_git_work_tree(Path(out)):
        raise ValueError(f"--out {out}: the page holds data, so it goes outside the git work tree")
    if not specs:
        raise ValueError("no --model given")
    dev = pick_device(device)
    runs = []
    for s in specs:
        ckpt = None if s.step is None else f"step_{s.step:07d}.pt"
        runs.append(eval_cli.load_run(s.run_dir, ckpt, torch.device("cpu")))
    for s, r in zip(specs, runs, strict=True):
        if r.cfg["data"]["source"] != "episodes":
            raise ValueError(f"{s.run_dir} trained on data.source={r.cfg['data']['source']!r}")
        if getattr(r.model, "goal", None) is None and getattr(r.model, "pixel", None) is None:
            raise ValueError(
                f"{s.run_dir} has neither a goal head nor a pixel decoder: the viewer has nothing "
                "of this model's to show (train.stages lacks goal and pixel)"
            )
        eval_cli.check_eval_split(r.cfg, eval_cfg, split)  # LeakageError on the training split
    dirs = {str(split_dir(r.cfg, split)) for r in runs}
    if len(dirs) != 1:
        raise ValueError(
            f"the models must read the same dataset for split {split!r}, got {sorted(dirs)}"
        )
    path = Path(dirs.pop())
    es = load_split(runs[0].cfg, split)
    rows = read_manifest(path)["episodes"]
    picked = pick_episodes(es.n, n_episodes)
    episodes, sims = [], []
    for i in picked:
        e, sim = _episode_payload(es, int(i), rows[i], path)
        episodes.append(e)
        sims.append(sim)
    batch = eval_cfg["eval"]["batch_size"]
    models = []
    with gpu_lock(f"viewer {split}", enabled=dev.type == "mps"):
        for s, r in zip(specs, runs, strict=True):
            model, c = r.model.to(dev), r.model.context_frames
            has_goal = getattr(model, "goal", None) is not None
            has_pixel = getattr(model, "pixel", None) is not None
            ks = list(r.cfg["model"]["goal_horizons"]) if has_goal else []
            k_eval = max([*r.cfg["model"]["horizons"], *ks])
            scored = valid_starts(es.T, c, k_eval)
            t = np.arange(c - 1, es.T)
            per_ep = []
            for i, sim in zip(picked, sims, strict=True):
                o = eval_cli.model_outputs(
                    model, es, np.full(len(t), i), t, batch, dev, goal=has_goal,
                    features=False, pixel=has_pixel,
                )  # fmt: skip
                per_ep.append(_model_episode(r, es, int(i), t, o, sim, int(scored[-1])))
            model.to("cpu")
            models.append(
                {
                    "label": s.label,
                    "run": r.run_dir.name,
                    "checkpoint": r.checkpoint.name,
                    "step": r.step,
                    "trained_on": r.cfg["data"]["split"],
                    "goal": has_goal,
                    "pixel": has_pixel,
                    "t0": c - 1,
                    "horizons": ks,
                    "k_eval": int(k_eval),  # sets the scored windows: max latent or goal k
                    "scored": [int(scored[0]), int(scored[-1])],
                    "eval": eval_summary(r.run_dir, r.checkpoint, split),
                    "episodes": per_ep,
                }
            )
    sim0 = sims[0]
    payload = {
        "title": "Ball Prediction Viewer",
        "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git": git_info(),
        "split": split,
        "split_note": _seed_note(eval_cfg, split, es.n),
        "dataset": path.name,
        "picked_by": f"{len(picked)} of {es.n} episodes, evenly spaced through the split's "
        "manifest, not chosen by any model's output",
        "sim": {
            "frame_size": int(sim0["frame_size"]),
            "view_height": float(sim0["camera"]["view_height"]),
            "px_per_m": camera.px_per_m(sim0),
            "fps": float(sim0["fps"]),
        },
        "regions": region_scales(eval_cfg["eval"]["coverage_levels"]),
        "episodes": episodes,
        "models": models,
    }
    html = render_html(payload)
    size = len(html.encode())
    if size > max_bytes:
        raise ValueError(
            f"the page would be {size / 2**20:.1f} MB, over the {max_bytes / 2**20:.3g} MB "
            "limit: use fewer --episodes or models"
        )
    if out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = artifacts_root() / "viewers" / f"{stamp}-{split}-{secrets.token_hex(3)}.html"
    return write_new(out, html), payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR[@STEP]",
        help="repeatable; the page's model picker lists them in this order",
    )
    p.add_argument("--split", default="test", choices=list(eval_cli.HELD_OUT))
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    p.add_argument("--device", default="cpu", help="cpu (default) or mps, under the GPU lock")
    p.add_argument(
        "--config",
        default=str(eval_cli.BASE_CONFIG),
        help="the eval config: canonical split seeds, coverage levels, batch size",
    )
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", help="the page's path (default: $JEPA_ARTIFACTS/viewers/...)")
    args = p.parse_args(argv)
    specs = [parse_model_spec(m) for m in args.model]
    cfg = load_config(args.config, args.set)
    path, payload = export(specs, args.split, args.episodes, cfg, out=args.out, device=args.device)
    size = path.stat().st_size / 2**20
    print(
        f"viewer: {path}\n{size:.1f} MB: {len(payload['episodes'])} {args.split} episodes x "
        f"{len(payload['models'])} models"
    )
    for m in payload["models"]:
        ev = m["eval"]["file"] if m["eval"] else "none for this checkpoint"
        print(f"  {m['label']}: {m['run']} step {m['step']} (eval: {ev})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
