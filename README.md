# wmdrift

A diagnostic suite for long-horizon failure modes in autoregressive video
world models. You give it a rollout video and the camera trajectory the model
was conditioned on, and it measures four things: camera drift, temporal
jitter, memory attenuation, and quality decay. Inference-only, runs on CPU,
uses CUDA for LPIPS/CLIP when available.

## Why I built this

I have been reading through [minWM](https://github.com/shengshu-ai/minWM)
(a minimal training/inference repo for AR video world models). Its
debug-world-model skill describes the long-horizon failure modes of
autoregressive rollouts qualitatively -- the camera drifts from the commanded
path, frames jitter, the model forgets places it has already seen, and image
quality decays over long rollouts. Issue
[#11](https://github.com/shengshu-ai/minWM/issues/11) discusses long-horizon
training and issue [#18](https://github.com/shengshu-ai/minWM/issues/18) the
camera translation scale ambiguity. But nowhere were these failures measured
with numbers I could compare across runs. So I built the thing I wanted to
have: point it at a rollout, get a report.

## Install

```
python -m venv .venv
.venv\Scripts\activate        # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Tested with Python 3.10, torch 2.14 (cu126), opencv-python 4.11 on Windows.
CPU-only torch works too; only LPIPS and optional CLIP alignment use the GPU.

## Quickstart

The repo ships four small synthetic videos with known ground-truth poses
(rendered by `scripts/make_fixtures.py`, see below):

```
python -m wmdrift analyze tests/fixtures/synthetic_pan.mp4 --pose "w*30" --output reports/pan
```

This writes `reports/pan/report.md`, `metrics.csv`, `per_frame.csv`, and four
PNG plots. Other commands:

```
python -m wmdrift batch reports_dir_with_videos --output reports/batch --pose "w*30"
python -m wmdrift compare reports/pan reports/jittered
```

Pose strings use the minWM convention: `w/s` forward/back, `a/d` strafe,
`u/dn` up/down, `j/l` yaw, `i/k` pitch, each step 0.08 units or 3 degrees,
chained with commas (`"w*10,a*5"`). The parser is a reimplementation of
`minwm/processors/camera.py` so trajectories match what the model saw.

The fixtures are synthetic -- rendered from a random 3D scene through the
exact commanded poses -- so they validate the pipeline, not any model. To
regenerate them:

```
python scripts/make_fixtures.py
```

To analyze a real rollout, pass its video and the pose string (or JSON
trajectory) it was generated with.

## The four metrics

**Camera drift** (`wmdrift/metrics/drift.py`). ORB features + essential
matrix + `recoverPose` give a visual-odometry trajectory from the video.
Umeyama Sim(3) aligns it to the commanded trajectory, and the report gives
per-frame ATE plus rotation geodesic error. The fitted scale factor is
reported explicitly: monocular VO cannot recover metric scale (this is
minWM issue #18 in measurement form), so the scale is part of the answer,
not something to hide.

**Temporal jitter** (`wmdrift/metrics/jitter.py`). Mean Farneback flow
magnitude per frame pair, linearly detrended, then `rfft`. The score is the
share of spectral power above fs/4. I also report the high-frequency RMS in
flow units, because the ratio alone barely separated my clean and jittered
fixtures -- a nearly flat residual still has a moderate ratio.

**Memory attenuation** (`wmdrift/metrics/memory.py`). If the commanded
trajectory returns to its start (or `--force-loop`), compare the first and
last frame with LPIPS. A model with real spatial memory reproduces the
starting view; a model that forgot it does not. Falls back to SSIM + MSE
when torch/lpips are unavailable.

**Quality decay** (`wmdrift/metrics/quality.py`). Per-frame Laplacian
variance, linear fit for the decay slope, and a collapse heuristic (first
frame below mean(first 25%) - 2 std). With `--prompt` and the `[clip]` extra,
per-frame CLIP similarity to the generation prompt is added.

## Example output

A real run on the pan fixture: [examples/sample_report/report.md](examples/sample_report/report.md).

![drift curve](examples/sample_report/drift_curve.png)

## Validation on Synthetic Fixtures

Each fixture is rendered from a random 3D scene through exactly the commanded
poses, with one known failure injected — so every metric has a ground truth to
check against. I ran the full pipeline on all four fixtures; each metric
separates its injected failure from the clean baseline.

| Fixture | Injected failure | Key evidence |
|---------|------------------|--------------|
| `synthetic_pan.mp4` | none — clean forward dolly, 30 steps | ATE mean **0.0257**; fitted VO scale 0.0834 vs commanded 0.08/step; jitter HF RMS 0.0259; quality slope −14.60/frame |
| `synthetic_jittered.mp4` | random ±3 px per-frame shifts | jitter HF RMS **0.4138** — 16× the clean fixture; ATE mean rises to 0.0542; 2 unreliable frame pairs (RANSAC found too few matches) |
| `synthetic_loop.mp4` | none — square loop (w→a→s→d, 10 each) exercising memory | loop-closure LPIPS **0.0097** (starting view reproduced); SSIM 0.9626, MSE 8.0; quality slope +0.82/frame (flat); ATE mean 0.2120 with 19 unreliable pairs — sharp turns stress the VO |
| `synthetic_decay.mp4` | progressive blur over the last 60% of frames | quality slope **−79.27/frame** — 5.4× steeper than the clean pan; collapse heuristic fires at frame 13; jitter stays low (HF RMS 0.0283), confirming blur and jitter are measured independently |

Two honest caveats from these runs:

- The clean pan's quality slope (−14.60/frame) comes from the forward dolly
  magnifying the scene, not from blur — Laplacian variance conflates zoom with
  sharpness loss (see Limitations).
- The loop fixture shows the ORB VO degrades on sharp turns, which is why
  unreliable frame pairs are counted and reported explicitly rather than
  silently dropped.

The test suite — 24 tests covering all four metrics and their edge cases —
passes in ≈9.5 s.

## Limitations

- Monocular scale ambiguity: ATE is measured after Sim(3) alignment, so it
  says the recovered path matches the commanded one up to a global scale --
  not metric truth.
- Laplacian variance is a crude no-reference quality proxy. A forward dolly
  legitimately loses sharpness as the scene magnifies, so the slope conflates
  zoom with blur. BRISQUE/NIQE would be better; both are fragile to install
  on Windows, so they are future work.
- The loop-closure check trusts the pose string: it assumes the model really
  was conditioned to return home.
- The VO assumes fx = fy = frame width and a centered principal point. Fine
  for square synthetic fixtures and most square model outputs, wrong for
  anything else.
- Not real-time. Farneback flow on a 31-frame 256px clip takes a few seconds;
  a full-length rollout takes minutes.
- LPIPS and CLIP were trained on natural images; on synthetic or heavily
  stylized rollouts their absolute values mean less.
- COLMAP or ORB-SLAM3 would give better trajectories than my 100-line ORB VO.
  Future work.

## Technology Stack

`Python` · `OpenCV` (ORB features, essential matrix, Farneback flow, Laplacian) · `NumPy` / `SciPy` · `PyTorch` · `lpips` (perceptual memory metric) · `openai-clip` (optional prompt alignment) · `scikit-image` (SSIM fallback) · `pytest` (24 tests)

## References

- minWM: https://github.com/shengshu-ai/minWM
- Causal Forcing (AR video diffusion): https://arxiv.org/abs/2504.01907
- RIFLEx (long-horizon AR video): https://arxiv.org/abs/2506.14340
- LPIPS: https://arxiv.org/abs/1801.03924
- ORB: https://ieeexplore.ieee.org/document/6126544
- Umeyama, "Least-squares estimation of transformation parameters between
  two point patterns": https://ieeexplore.ieee.org/document/88573

---

Most of the code here was written with Claude Code; the design, experiments,
fixture generation, and testing were done by me.
