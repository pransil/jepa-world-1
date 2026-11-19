# SPEC: JEPA world model on rolling-ball episodes

Status: v0.1, design handoff. Several choices are defaults that still need confirming. They are marked **[OPEN]**.

## 1. Goal

Build a small, controllable testbed for a JEPA-style world model with multi-horizon prediction, where:

- Ground truth is exact (simulator state, not labels).
- The embedding hierarchy is tested directly: fine levels should predict near futures, coarse levels far futures.
- Uncertainty is explicit and testable, because the hidden physical parameters are known.
- The model can later be extended with goal-weighted losses and action conditioning.

Version one is a predictive model only. It has no policy, no actions, and no goals beyond predicting ball position.

## 2. Environment

- Hardware: MacBook Pro, M4, 36 GB unified memory. Development runs locally.
- Language: Python 3.11 or newer.
- Core libraries (verify current versions and APIs before pinning):
  - `torch` with the MPS backend. Set `PYTORCH_ENABLE_MPS_FALLBACK=1`, and log which ops fall back to CPU during profiling.
  - `mujoco` (native Apple Silicon build), for physics and rendering.
  - `numpy`, `scikit-learn` (linear probes and metrics), `pyyaml` (configs), `pytest` (tests).
  - Optional: `tensorboard` for logging.
- Do not assume any version or API. Check the installed package before writing code against it.
- Sandbox note: code may be written and tested in a Linux sandbox first. MPS-specific behavior must be verified on the Mac.

## 3. Simulation and data

### 3.1 Physics and rendering

- Engine: MuJoCo.
- Terrain: a heightfield `h(x)` that varies along x and is constant along y, so the terrain is effectively a 2D profile extruded in depth.
- Ball: one sphere. Motion is restricted to the x-z plane (y fixed), using a joint constraint or a narrow channel. **[OPEN]** Confirm the constraint method against the installed MuJoCo version.
- Hidden per-episode parameters (not given to the model): ball radius, friction, restitution (bounciness). Sample them per episode.
- Gravity: standard, fixed across episodes.

### 3.2 Camera

- Side-on, fixed orientation, looking along the y axis.
- Tracks the ball horizontally with smoothing (lerp toward the ball's x position).
- **[OPEN]** Camera tracking mode. Pick one:
  - `track_full`: the camera follows x and y, so the ball is always in frame. Off-screen terrain is the main hidden information.
  - `track_lagged` (default proposal): smoothed tracking with a deadzone in y. The ball can leave the frame during high bounces or deep drops, which gives out-of-view events.
  - `track_lagged_obstacles`: as above, plus fixed foreground obstacles that occlude the ball. Gives true occlusion events.
- The camera pose is recorded as ground truth every frame.

### 3.3 Rendering

- Frame size: 64 × 64 RGB.
- Shading: flat and simple. Visual detail should not dominate the loss.
- Rate: 15 frames per second.
- Episode length: 60 frames (4 seconds). Long enough for horizon k = 20 to exist for most start times.

### 3.4 Ground truth per frame

Saved alongside each frame:

- Ball world position (x, z), velocity (vx, vz).
- Ball hidden parameters (radius, friction, restitution). Stored once per episode.
- Terrain seed and heightfield parameters (stored once per episode).
- Camera pose.
- Event flags: `bounce`, `out_of_view`, `occluded`, `contact`.

### 3.5 Dataset splits

- Split by **terrain seed**, not by frame. Test terrains are never seen in training.
- Development set: 10 terrain seeds, fixed. Used for debugging only.
- Training: 500 terrain seeds, each with several ball-parameter samples. Ball parameters vary in training from the start.
- Test: 50 held-out terrain seeds, with a separate set of ball parameters.
- Store the generator config and seeds, so episodes can be regenerated exactly.

## 4. Model

### 4.1 Overview

Four parts:

1. **Context encoder** (U-Net encoder, trainable). Produces embeddings at three levels.
2. **Target encoder** (EMA copy of the context encoder, no gradients). Encodes future frames into target embeddings.
3. **Predictor** (multi-horizon heads, trainable). Predicts future target embeddings from context embeddings and horizon k.
4. **Auxiliary heads** (trainable, not part of the JEPA loop): goal head (ball position) and pixel decoder.

### 4.2 Inputs and shapes (starting values)

- Context input: the last 4 frames, channel-stacked. Shape (B, 12, 64, 64).
- Levels (context encoder and target encoder outputs):
  - L1: (B, 64, 64, 64), fine.
  - L2: (B, 128, 32, 32), mid.
  - L3: (B, 256, 8, 8), coarse.
- Horizons: k in {1, 2, 3, 4, 5, 10, 20} frames.

### 4.3 Context encoder

- U-Net with three resolution levels as above, each with two 3×3 convolutions, normalization, and a nonlinearity.
- Downsampling by stride-2 convolutions or pooling. Choose one and keep it fixed.
- Outputs the three feature maps. The decoder path is used only by the pixel decoder.

### 4.4 Target encoder

- Same architecture as the context encoder. Parameters are an exponential moving average of the context encoder's parameters:

  θ_target ← m · θ_target + (1 − m) · θ_context

- Momentum m: default 0.996. **[OPEN]** Tune, possibly toward 0.999 later.
- Run under `torch.no_grad()`. Targets are detached before the loss is computed.
- Encodes the frame at t + k with the same architecture, producing targets at the same level the head reads.

### 4.5 Predictor (multi-horizon heads)

- One head per horizon k. Each head reads from one level:
  - k in {1..5}: reads L1, outputs a map shaped like the L1 target.
  - k = 10: reads L2.
  - k = 20: reads L3.
- Input to each head: the level's context features, concatenated with a learned embedding of k, broadcast over spatial positions.
- Head structure: two 3×3 convolutions with normalization and nonlinearity, then a 1×1 output convolution. Start with a 32-channel hidden width for the L1 heads to control compute.
- Output form:
  - Near heads (k = 1..5): deterministic mean.
  - Far heads (k = 10, 20): mean and log-variance per location (diagonal Gaussian).
- Optional action input: a conditioning vector, zeros in version one. Keep the input slot in the code.

### 4.6 Goal head (auxiliary)

- Reads pooled L3 features (plus optionally pooled L2) and outputs a 2D position distribution for the ball's world (x, z) at horizon k.
- Output: mean and covariance (or diagonal variance for version one).
- Loss: Gaussian negative log-likelihood against the ground-truth position.

### 4.7 Pixel decoder (auxiliary, optional grounding)

- Reads L1 plus the skip from the U-Net decoder path, and predicts the RGB frame at t + 1.
- Loss: L1, with a small weight. Keep it separate from the JEPA loss.
- Purpose: keep the scene grounded. It must not drive the embedding toward blur.

## 5. Losses

### 5.1 JEPA latent loss

For each horizon k, the loss is a distance between the predicted and stopped target embeddings:

L_k = mean over positions of ‖ predictor_k(z_ctx) − stop_grad(z_tgt,k) ‖²

- Use the same distance for all horizons. Normalize the embeddings per level if needed.
- Since 2026-11-19 the default averages each level's latent terms (and its collapse guard) over channels instead of
  summing them (`loss.latent_reduce: mean`; docs/decisions/2026-11-19-latent-loss-normalization.md).
- Cosine distance is an alternative. Pick one and report it.
- Far heads: replace the squared distance with the Gaussian NLL of the target under the predicted distribution.
- **Collapse guard (optional, on by default in tests):** a VICReg-style variance term on the context embeddings, penalizing any dimension whose batch standard deviation falls below a threshold.

### 5.2 Goal loss

L_goal = Gaussian NLL of the true ball (x, z) under the goal head's prediction, for each horizon.

### 5.3 Pixel loss

L_pix = L1(predicted frame, true frame), at low weight.

### 5.4 Total

L = Σ_k w_k · L_k + w_goal · L_goal + w_pix · L_pix

Starting weights (placeholders, tune with ablations):

- w_k = 1.0 for all horizons.
- w_goal = 0.5. With the per-channel latent losses (§5.1, 2026-11-19) the default is 0.5 / 256, the same ratio
  to the L3 latent terms (same decision).
- w_pix = 0.05. Since 2026-11-19 the default instead sets w_pix each step so that w_pix · L_pix is 10% of
  the rest of L (`loss.pix_share`; docs/decisions/2026-11-19-pixel-share-default.md).

Log every term separately, every step.

### 5.5 Schedule

1. Train the near heads and the context and target encoders first.
2. Add the far heads.
3. Add the goal head.
4. Add the pixel decoder last, at low weight.

## 6. Training

- Batch size: start at 32. Scale to 64 if memory and speed allow.
- Optimizer: AdamW. Learning rate and weight decay: start with defaults, then tune.
- Mixed precision: avoid assuming bf16 works on MPS. Start with fp32 and test any lower-precision path explicitly.
- Checkpoints: save model, EMA target, optimizer, step, and config.
- Seeds: fix random seeds for data, initialization, and training. Log library versions.

## 7. Evaluation

### 7.1 Baselines (required, reported alongside every result)

- **Persistence:** predict the current embedding (or current position) for every horizon.
- **Physics baseline:** constant-velocity or simulator-aware prediction, using known terrain. This is the bar the learned model must beat on known terrain.

### 7.2 Metrics

- Position error (world units) by horizon, for the goal head's mean.
- Ratio of model error to persistence error, by horizon. Below 1.0 means the model beats persistence.
- Calibration: the fraction of truth values inside the 50% and 90% predicted intervals. Should match the nominal levels.
- Rollout drift: how error grows across horizons.
- Probe accuracy (linear probes on frozen embeddings, using scikit-learn): ball position, and the hidden parameters (friction, restitution, radius), at each level.
- Event behavior: uncertainty before and after out-of-view and occlusion events. It should rise when the ball disappears and fall when it reappears.

### 7.3 Generalization

- All headline numbers are on held-out terrain seeds.
- Report development-set numbers separately, and never use them to claim generalization.

## 8. Milestones

Each milestone ends with passing tests and a short written summary.

**M1. Generator.** MuJoCo scene, heightfield, ball, camera (default `track_lagged`), rendering, ground-truth export. Tests: deterministic regeneration from seed; per-frame ground truth matches rendered geometry; event flags are consistent with ground truth.

**M2. Data pipeline and baselines.** Dataset loader with the leakage test. Persistence and physics baselines on the test split. Tests: context never includes frames at or after t + k; splits never share terrain seeds.

**M3. Near-horizon JEPA.** Context encoder, target encoder (EMA), near heads only, latent loss only. Tests: no gradient reaches the target encoder; EMA update matches the formula; loss decreases on the development set.

**M4. Multi-horizon and far heads.** Add L2 and L3 heads and the Gaussian NLL on far heads. Tests: head output shapes match targets; far-head variance is positive.

**M5. Goal and pixel heads.** Goal head with NLL, pixel decoder at low weight. Tests: goal NLL is finite; pixel term does not dominate the logged total.

**M6. Evaluation and probes.** All metrics in section 7, with the held-out test split.

**M7. Ablations.** Skip connections on and off. Top-down prediction (see section 9). Camera modes and obstacles. Pixel weight at 0, low, and high.

## 9. Open design questions

1. **Camera mode** (section 3.2). Default: `track_lagged`. Needs confirming.
2. **Top-down prediction.** Not in version one. As an ablation, coarse heads also predict lower-level future targets, and errors propagate upward. Define the exact targets before implementing.
3. **Collapse guard.** Whether to use the VICReg-style term by default, or only if collapse appears.
4. **Target size at L1.** Compute cost may be high. Benchmark before fixing the head width. Reduce L1 head width or use 1×1 convolutions if needed.
5. **Distance function** for the latent loss: squared distance or cosine.
6. **Horizon ordering.** Whether far heads read the same context as near heads, or a summary of it.

## 10. Repo layout (proposed)

```
jepa-ball/
  SPEC.md
  configs/            # yaml, one per experiment
  sim/                # MuJoCo scene, terrain, camera, rendering, ground truth
  data/               # dataset, splits, leakage checks
  models/             # unet.py, heads.py, jepa.py (EMA, losses), goal.py, pixel.py
  train.py
  eval.py             # metrics, baselines, probes
  tests/              # pytest
```

Keep the JEPA core (`jepa.py`) separate from the pixel decoder and goal head, so each can be removed or changed without touching the others.

## 11. Working notes for the implementer

- Test before claiming something works. Every milestone needs tests that pass, not just code that runs.
- Verify MuJoCo, PyTorch, and MPS APIs against the installed versions. Do not rely on memory for API details.
- Benchmark the L1 heads on the Mac early, before building the rest on top of them.
- Log per-term losses, per-horizon errors, and persistence ratios from the first run.
- Ask before changing any item in section 9.
