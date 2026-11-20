"""A: the latent heads' "copy the current embedding" baseline (relerr/copy/k) and per-head encoder
gradient norms (gradnorm/enc/<term>/<level>). Both are diagnostics: neither may change training
(jepa_world/models/DESIGN.md, "Diagnostics")."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tiny import random_batch, tiny_aux_cfg, tiny_cfg

from jepa_world import train
from jepa_world.models.jepa import target_stack
from jepa_world.models.world import compute_world_losses, loss_term_names

FULL = [
    "train.stages=[near, far, goal, pixel]",
    "train.stage_start={near: 0, far: 0, goal: 0, pixel: 0}",
]
TOP_DOWN = ["model.top_down=true"]


def full_cfg(*overrides: str) -> dict:
    """Every term there is: near, far, guard, goal, pixel (and top-down with TOP_DOWN)."""
    return tiny_aux_cfg(*FULL, *overrides)


def model_of(cfg: dict, seed: int = 0) -> train.WorldModel:
    torch.manual_seed(seed)
    return train.build_model(cfg)


# ---- the weighted terms of loss/total ----


@pytest.mark.parametrize("extra", [[], TOP_DOWN], ids=["plain", "top_down"])
@pytest.mark.parametrize(
    ("active", "aux"), [(None, None), ([1, 2, 3, 4, 5], ()), ([1, 2, 3, 4, 5, 10], ["goal"])]
)
def test_weighted_terms_sum_to_loss_total_bitwise(extra, active, aux):
    """terms[name] is each weighted term exactly as it enters loss/total, and only the active ones:
    their sum, in order, is loss/total bit for bit."""
    cfg = full_cfg(*extra)
    model = model_of(cfg)
    terms: dict[str, torch.Tensor] = {}
    losses = compute_world_losses(
        model, random_batch(cfg), cfg["loss"], active=active, aux_active=aux, terms=terms
    )
    total = losses["loss/total"].new_zeros(())
    for t in terms.values():
        total = total + t
    assert torch.equal(total, losses["loss/total"])
    act = set(model.horizons if active is None else active)
    want = {f"k{k}" for k in act} | {"var"}
    want |= {f"td/k{k}" for k in model.td_level if k in act}
    want |= {
        {"goal": "goal", "pixel": "pix"}[s] for s in (model.aux_stages if aux is None else aux)
    }
    assert set(terms) == want


def test_asking_for_terms_or_the_copy_baseline_changes_no_other_value():
    cfg = full_cfg()
    batch = random_batch(cfg)
    plain = compute_world_losses(model_of(cfg), batch, cfg["loss"])
    both = compute_world_losses(model_of(cfg), batch, cfg["loss"], terms={}, copy_baseline=True)
    assert set(both) - set(plain) == {f"relerr/copy/k{k}" for k in model_of(cfg).horizons}
    # the same columns the run logs for it
    assert set(both) - set(plain) == {
        k for k in train.diagnostic_keys(model_of(cfg), cfg) if k.startswith("relerr/copy/")
    }
    for k, v in plain.items():
        assert torch.equal(v, both[k]) or (v.isnan() and both[k].isnan()), k


# ---- the copy baseline ----


def test_copy_baseline_is_zero_when_nothing_moves():
    """A static clip: the stack ending at t is the stack ending at t+k, so copying is exact."""
    cfg = full_cfg()
    batch = random_batch(cfg)
    batch["frames"] = batch["frames"][:, :1].expand_as(batch["frames"]).contiguous()
    losses = compute_world_losses(model_of(cfg), batch, cfg["loss"], copy_baseline=True)
    for k in model_of(cfg).horizons:
        assert losses[f"relerr/copy/k{k}"].item() == 0.0, k


def test_copy_baseline_is_the_relerr_of_the_target_embedding_at_t():
    """relerr/copy/k = mean_{b,pos} ||z_t - z_{t+k}||^2 over channels / the target's batch spread,
    the same denominator as relerr/k, with z from the target encoder at the head's level."""
    cfg = full_cfg()
    model = model_of(cfg)
    batch = random_batch(cfg, batch=3)
    losses = compute_world_losses(model, batch, cfg["loss"], copy_baseline=True)
    frames, c = batch["frames"], model.context_frames
    for k in model.horizons:
        lv = model.head_level[k]
        with torch.no_grad():
            z0 = model.target_encoder(target_stack(frames, c, 0), upto=lv)[lv]
            zk = model.target_encoder(target_stack(frames, c, k), upto=lv)[lv]
        sq = (z0 - zk).pow(2).sum(1).mean()
        spread = zk.var(dim=0, unbiased=False).sum(0).mean()
        torch.testing.assert_close(losses[f"relerr/copy/k{k}"], sq / spread, rtol=1e-5, atol=0)


def test_top_down_branches_get_their_own_copy_baseline():
    """relerr/copy/td/k<k>: the stack ending at t at the branch's finer level, against the
    branch's target, as relerr/td/k<k> is scored."""
    cfg = full_cfg(*TOP_DOWN)
    model = model_of(cfg)
    batch = random_batch(cfg, batch=3)
    losses = compute_world_losses(model, batch, cfg["loss"], copy_baseline=True)
    frames, c = batch["frames"], model.context_frames
    assert model.td_level
    for k, lv in model.td_level.items():
        with torch.no_grad():
            z0 = model.target_encoder(target_stack(frames, c, 0), upto=lv)[lv]
            zk = model.target_encoder(target_stack(frames, c, k), upto=lv)[lv]
        sq = (z0 - zk).pow(2).sum(1).mean()
        spread = zk.var(dim=0, unbiased=False).sum(0).mean()
        torch.testing.assert_close(losses[f"relerr/copy/td/k{k}"], sq / spread, rtol=1e-5, atol=0)


def test_copy_baseline_takes_no_gradient():
    cfg = full_cfg()
    model = model_of(cfg)
    losses = compute_world_losses(model, random_batch(cfg), cfg["loss"], copy_baseline=True)
    assert all(not losses[f"relerr/copy/k{k}"].requires_grad for k in model.horizons)


# ---- encoder gradient norms ----


def test_grad_norms_match_autograd_per_term_and_level():
    cfg = full_cfg(*TOP_DOWN)
    model = model_of(cfg)
    terms: dict[str, torch.Tensor] = {}
    compute_world_losses(model, random_batch(cfg), cfg["loss"], terms=terms)
    norms = train.encoder_grad_norms(model, terms)
    enc = model.context_encoder
    for name, t in terms.items():
        for lv, stage in zip(enc.names, enc.stages, strict=True):
            ps = list(stage.parameters())
            gs = torch.autograd.grad(t, ps, retain_graph=True, allow_unused=True)
            want = math.sqrt(sum(float(g.pow(2).sum()) for g in gs if g is not None))
            got = norms[f"gradnorm/enc/{name}/{lv}"]
            assert math.isclose(float(got), want, rel_tol=1e-5, abs_tol=1e-12), (name, lv)


def test_heads_pull_only_on_the_levels_below_what_they_read():
    """Near heads read L1: no gradient reaches the L2 and L3 stages. The k = 20 head and the goal
    head read L3, so they reach every stage."""
    cfg = full_cfg()
    model = model_of(cfg)
    terms: dict[str, torch.Tensor] = {}
    compute_world_losses(model, random_batch(cfg), cfg["loss"], terms=terms)
    n = {k: float(v) for k, v in train.encoder_grad_norms(model, terms).items()}
    for k in (1, 5):
        assert n[f"gradnorm/enc/k{k}/L1"] > 0
        assert n[f"gradnorm/enc/k{k}/L2"] == 0 and n[f"gradnorm/enc/k{k}/L3"] == 0
    for name in ("k20", "goal"):
        assert all(n[f"gradnorm/enc/{name}/{lv}"] > 0 for lv in ("L1", "L2", "L3")), name


# ---- training is unchanged ----


def _params_after(cfg: dict, tmp_path, name: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    batches = (random_batch(cfg, seed=i) for i in itertools.count())
    train.run(cfg, tmp_path / name, batches=batches, device=torch.device("cpu"))
    (ckpt,) = (tmp_path / name / "checkpoints").iterdir()
    return torch.load(ckpt, weights_only=False)["model"]


@pytest.mark.parametrize("extra", [[], TOP_DOWN], ids=["plain", "top_down"])
def test_diagnostics_on_or_off_train_bitwise_the_same(tmp_path, extra):
    """The M7 ablations reuse the 48k reference only while training is bitwise unchanged: params
    after a few steps are equal with both diagnostics every step and with both off."""
    base = ["train.steps=4", *extra]
    off = full_cfg(*base, "train.grad_norm_every=0", "train.copy_every=0")
    on = full_cfg(*base, "train.grad_norm_every=1", "train.copy_every=1")
    a, b = _params_after(off, tmp_path, "off"), _params_after(on, tmp_path, "on")
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _rows(run_dir) -> list[dict]:
    return [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("extra", [[], TOP_DOWN], ids=["plain", "top_down"])
def test_diagnostics_are_logged_only_on_their_steps(tmp_path, extra):
    """A diagnostic column is in a step's line exactly when it was measured that step, finite;
    every other column is in every line. And diagnostic_keys names every one of them."""
    cfg = full_cfg("train.steps=7", "train.grad_norm_every=2", "train.copy_every=3", *extra)
    torch.manual_seed(0)
    batches = (random_batch(cfg, seed=i) for i in itertools.count())
    final = train.run(cfg, tmp_path, batches=batches, device=torch.device("cpu"))
    rows = _rows(tmp_path)
    keys = train.diagnostic_keys(train.build_model(cfg), cfg)
    assert set(keys) == {k for r in rows for k in r if k.startswith(train.DIAGNOSTIC_PREFIXES)}
    assert not set(keys) & set(final)  # run() returns the step's terms, not its diagnostics
    plain = {k for k in rows[0] if k not in keys}
    for r in rows:
        assert plain <= set(r)
        done = r["step"] - 1  # steps completed before this one, as the stage schedule counts
        for key in keys:
            every = 3 if key.startswith("relerr/copy/") else 2
            assert (key in r) == (done % every == 0), (r["step"], key)
            assert key not in r or math.isfinite(r[key]), (r["step"], key)


def test_bench_and_other_stepless_calls_measure_no_diagnostics():
    """train_step(step=None) is a step outside a run (bench times it): no diagnostic runs."""
    cfg = full_cfg()
    model = model_of(cfg)
    out = train.train_step(model, train.build_optimizer(model, cfg), random_batch(cfg), cfg)
    assert not [k for k in out if k.startswith(train.DIAGNOSTIC_PREFIXES)]


def test_every_recorded_term_has_a_name_in_loss_term_names():
    """diagnostic_keys builds its columns from loss_term_names, so a term recorded under another
    name would be lost: with every stage active the two are the same set."""
    for extra in ([], TOP_DOWN):
        cfg = full_cfg(*extra)
        model = model_of(cfg)
        terms: dict[str, torch.Tensor] = {}
        compute_world_losses(model, random_batch(cfg), cfg["loss"], terms=terms)
        assert set(terms) == set(loss_term_names(model, cfg["loss"]))


def test_an_unnamed_term_fails_the_step_instead_of_vanishing(monkeypatch):
    import jepa_world.train as t

    monkeypatch.setattr(t, "loss_term_names", lambda model, loss_cfg: ["k1"])
    cfg = full_cfg()
    model = model_of(cfg)
    with pytest.raises(RuntimeError, match="loss_term_names"):
        t.train_step(model, t.build_optimizer(model, cfg), random_batch(cfg), cfg, step=0)


def test_a_term_not_yet_in_the_loss_has_no_grad_norm(tmp_path):
    """Before its stage starts a head isn't trained, so it pulls on nothing: no column that step,
    not a norm."""
    cfg = full_cfg(
        "train.steps=2",
        "train.stage_start={near: 0, far: 1, goal: 1, pixel: 1}",
        "train.grad_norm_every=1",
    )
    torch.manual_seed(0)
    batches = (random_batch(cfg, seed=i) for i in itertools.count())
    train.run(cfg, tmp_path, batches=batches, device=torch.device("cpu"))
    first, second = _rows(tmp_path)
    for name in ("k10", "k20", "goal", "pix"):
        assert f"gradnorm/enc/{name}/L1" not in first, name
        assert math.isfinite(second[f"gradnorm/enc/{name}/L1"]), name
    assert math.isfinite(first["gradnorm/enc/k1/L1"])
    # the optimizer's own gradient: L2 and L3 get none before a head reading them trains
    assert first["gradnorm/enc/total/L2"] == 0 and first["gradnorm/enc/total/L3"] == 0
    assert second["gradnorm/enc/total/L3"] > 0


def test_total_grad_norm_is_the_encoder_gradient_the_optimizer_gets(tmp_path):
    cfg = full_cfg("train.steps=1", "train.grad_norm_every=1")
    model = model_of(cfg)
    batch = random_batch(cfg)
    ref = model_of(cfg)
    compute_world_losses(ref, batch, cfg["loss"])["loss/total"].backward()
    out = train.train_step(model, train.build_optimizer(model, cfg), batch, cfg, step=0)
    enc = ref.context_encoder
    for lv, stage in zip(enc.names, enc.stages, strict=True):
        want = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in stage.parameters()))
        assert math.isclose(float(out[f"gradnorm/enc/total/{lv}"]), want, rel_tol=1e-5), lv


# ---- config ----


@pytest.mark.parametrize("key", ["grad_norm_every", "copy_every"])
@pytest.mark.parametrize("value", ["-1", "1.5", "true", "null"])
def test_diagnostic_intervals_must_be_ints_ge_0(key, value):
    with pytest.raises(ValueError, match=f"train.{key}"):
        train.check_config(tiny_cfg(f"train.{key}={value}"))


def test_diagnostics_are_on_by_default():
    """So the M7 ablation runs, which inherit base.yaml, log them."""
    t = tiny_cfg()["train"]
    assert t["grad_norm_every"] > 0 and t["copy_every"] > 0


MPS_BITWISE = """
import json, sys
sys.path.insert(0, "tests")
import jepa_world  # sets PYTORCH_ENABLE_MPS_FALLBACK before torch
import torch
from tiny import random_batch, spec_cfg
from jepa_world import train
from jepa_world.bench import record_mps_fallbacks

def params_after(every):
    cfg = spec_cfg(
        "train.batch_size=2",
        "train.stages=[near, far, goal, pixel]",
        "train.stage_start={near: 0, far: 0, goal: 0, pixel: 0}",
        f"train.grad_norm_every={every}",
        f"train.copy_every={every}",
    )
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = train.build_model(cfg).to(dev)
    opt = train.build_optimizer(model, cfg)
    diag = {}
    for step in range(3):
        batch = train.to_device(random_batch(cfg, seed=step), dev)
        out = train.train_step(model, opt, batch, cfg, step=step)
        diag.update({k: v for k, v in out.items() if k.startswith(train.DIAGNOSTIC_PREFIXES)})
    torch.mps.synchronize()
    finite = all(bool(torch.isfinite(v).item()) for v in diag.values())
    return {k: v.cpu() for k, v in model.state_dict().items()}, len(diag), finite

with record_mps_fallbacks() as fallbacks:
    off, n_off, _ = params_after(0)
    on, n_on, finite = params_after(1)
same = sorted(k for k in off if torch.equal(off[k], on[k]))
print(json.dumps({"n": len(off), "same": len(same), "diag_off": n_off, "diag_on": n_on,
                  "finite": finite, "fallbacks": fallbacks}))
"""


@pytest.mark.mps
def test_diagnostics_leave_mps_training_bitwise_the_same():
    """The claim M7 relies on, on the device the reference trained on: the SPEC-shape model, every
    stage, 3 steps on MPS, params bitwise equal with both diagnostics every step and with both off.
    A fresh interpreter, so every fallback op's first use is seen."""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-c", MPS_BITWISE], cwd=root, capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["same"] == res["n"], res
    assert res["diag_off"] == 0 and res["diag_on"] > 0 and res["finite"], res
    assert res["fallbacks"] == [], res["fallbacks"]


# ---- experiments/compare_metrics.py: the re-run check the M7 reference reuse relies on ----


def _compare_metrics():
    spec = importlib.util.spec_from_file_location(
        "compare_metrics", Path(__file__).resolve().parents[1] / "experiments/compare_metrics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def _write_rows(run_dir: Path, rows: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


def test_a_rerun_with_diagnostics_matches_a_run_without_them(tmp_path):
    """The reference predates the diagnostics: a re-run that logs them still compares clean, both
    ways, because every key both have is bitwise equal and the rest are diagnostics."""
    compare = _compare_metrics()
    base = ["train.steps=4", "train.log_every=1"]
    for name, every in (("off", 0), ("on", 1)):
        cfg = full_cfg(*base, f"train.grad_norm_every={every}", f"train.copy_every={every}")
        torch.manual_seed(0)
        batches = (random_batch(cfg, seed=i) for i in itertools.count())
        train.run(cfg, tmp_path / name, batches=batches, device=torch.device("cpu"))
    on, off = str(tmp_path / "on"), str(tmp_path / "off")
    assert compare(["--run", on, "--ref", off]) == 0
    assert compare(["--run", off, "--ref", on]) == 0


def test_compare_metrics_still_refuses_real_differences(tmp_path):
    compare = _compare_metrics()
    ref = _write_rows(tmp_path / "ref", [{"step": 1, "loss/total": 1.0, "relerr/k1": 0.5}])
    cases = {  # name: (rows, expected exit status)
        "value": ([{"step": 1, "loss/total": 1.5, "relerr/k1": 0.5}], 1),
        "missing key": ([{"step": 1, "loss/total": 1.0}], 1),
        "new plain key": ([{"step": 1, "loss/total": 1.0, "relerr/k1": 0.5, "x": 1.0}], 1),
        "new diagnostic": (
            [{"step": 1, "loss/total": 1.0, "relerr/k1": 0.5, "gradnorm/enc/k1/L1": 2.0}],
            0,
        ),
    }
    for name, (rows, want) in cases.items():
        run = _write_rows(tmp_path / name.replace(" ", "_"), rows)
        assert compare(["--run", str(run), "--ref", str(ref)]) == want, name
    a = _write_rows(tmp_path / "a", [{"step": 1, "gradnorm/enc/k1/L1": 2.0}])
    b = _write_rows(tmp_path / "b", [{"step": 1, "gradnorm/enc/k1/L1": 3.0}])
    assert compare(["--run", str(a), "--ref", str(b)]) == 1  # shared diagnostics must match too
