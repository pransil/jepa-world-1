"""MPS cost of A's diagnostics at the 48k reference config (all four stages, batch 32, synthetic
frames + ball_pos). Times plain steps, steps with the copy baseline due, and steps with both due,
under the GPU lock. Prints the ms per step and the fractions of training time at
the given intervals. Run from the repo root: uv run python experiments/diag_overhead.py"""

import json
import time

import torch

from jepa_world import train
from jepa_world.utils.config import load_config
from jepa_world.utils.locks import gpu_lock

cfg = load_config("configs/exp_headline_48k.yaml", ["data.source=synthetic"])
dev = torch.device("mps")


def timed(model, opt, batches, steps):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for s in steps:
        train.train_step(model, opt, train.to_device(next(batches), dev), cfg, step=s)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000 / len(steps)


with gpu_lock("A diagnostics overhead timing"):
    torch.manual_seed(0)
    model = train.build_model(cfg).to(dev)
    opt = train.build_optimizer(model, cfg)
    batches = train.make_batches(cfg, model.horizons)
    timed(model, opt, batches, [7001 + i for i in range(10)])  # warm-up, all stages active
    plain = timed(model, opt, batches, [7011 + i for i in range(40) if (7011 + i) % 10])
    copy = timed(
        model, opt, batches, [7010 + 10 * i for i in range(1, 15) if (7010 + 10 * i) % 500]
    )
    both = timed(model, opt, batches, [7500 + 500 * i for i in range(8)])
out = {
    "ms_plain": round(plain, 1),
    "ms_copy_due": round(copy, 1),
    "ms_both_due": round(both, 1),
    "overhead_copy_every_10": round((copy - plain) / 10 / plain, 4),
    "overhead_grad_every_500": round((both - copy) / 500 / plain, 4),
}
out["overhead_total"] = round(out["overhead_copy_every_10"] + out["overhead_grad_every_500"], 4)
print(json.dumps(out))
