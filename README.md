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

## What I Actually Got When I Ran It

So I ran wmdrift on all four synthetic fixtures to see if it actually works. Here's what happened:

**Clean pan (synthetic_pan.mp4):**
- Camera moved forward 30 steps like commanded
- VO recovered the trajectory pretty well - ATE mean was only 0.0257
- Scale factor came out to 0.0834 which is close to the commanded 0.08 per step
- Jitter was low (HF RMS 0.0259) - makes sense cause its clean
- Quality decay slope was -14.60 per frame - this is just from the forward dolly zooming in, not actual blur
- No memory test cause it's not a loop

**Jittered video (synthetic_jittered.mp4):**
- Same camera path but with random +-3px shifts added to each frame
- Drift was worse (ATE mean 0.0542 vs 0.0257 for clean)
- Jitter HF RMS jumped to **0.4138** - thats 16x higher than the clean one! So it definitely detects jitter
- Had 2 unreliable frame pairs where RANSAC couldn't find enough matches
- Quality slope was similar to clean (-14.45) cause the jitter doesn't actually blur things

**Loop video (synthetic_loop.mp4):**
- Camera went in a square: forward 10, right 10, back 10, left 10 - should end up where it started
- This one tested the memory metric
- LPIPS between first and last frame was **0.0097** - super low, means the model remembered where it started
- SSIM was 0.9626, MSE was 8.0
- Quality slope was +0.82 per frame - basically flat, no decay
- Drift was higher (ATE mean 0.2120) and had 19 unreliable pairs - the sharp turns in the loop made VO harder

**Decay video (synthetic_decay.mp4):**
- Forward dolly but with progressive blur added to last 60% of frames
- Quality decay slope was **-79.27 per frame** - thats 5.4x steeper than the clean pan (-14.60)
- Collapse heuristic fired at frame 13 - detected the quality drop
- Jitter was low (HF RMS 0.0283) - blur doesn't cause jitter, just quality loss
- Had 1 unreliable frame pair

So yeah, all four metrics work. Drift detects camera errors, jitter detects frame instability, memory detects if the model forgets, and quality detects blur/decay. The numbers actually separate the different failure modes which is what I wanted.

I also ran the test suite - all 24 tests passed in like 9.5 seconds. Tests cover all the metrics and edge cases.

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
