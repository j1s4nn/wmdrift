"""Generate the synthetic test fixtures in tests/fixtures/.

Renders a random 3D scene (smooth depth + high-contrast texture) through the
minWM pose-string camera convention, so every video has a known ground-truth
trajectory. The scene has strong depth variation on purpose: a planar or
low-texture scene makes the essential-matrix VO in wmdrift.pose.vo degenerate.

Usage:  python scripts/make_fixtures.py
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wmdrift.pose.parser import c2w_trajectory  # noqa: E402

W = H = 256
FX = FY = 256.0
CX = CY = 128.0
FPS = 8
SEED = 0

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")


def make_scene(rng):
    """Random depth map + color texture for the frame-0 camera.

    Depth is blurred noise mapped to [3.0, 7.5] meters-ish: far enough that
    the w*30 dolly (2.4 units) never crosses geometry, varied enough that
    consecutive frames have real parallax.
    """
    noise = rng.random((H, W, 3))
    texture = cv2.GaussianBlur(noise, (31, 31), 0)
    # sharp random shapes give ORB something to grab onto
    for _ in range(40):
        x0, y0 = rng.integers(0, W - 30, 2)
        size = int(rng.integers(8, 30))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(texture, (x0, y0), (x0 + size, y0 + size), color, -1)
        else:
            cv2.circle(texture, (x0, y0), size // 2, color, -1)
    texture = np.clip(texture * 255, 0, 255).astype(np.uint8)

    dnoise = rng.random((H, W))
    dnoise = cv2.GaussianBlur(dnoise, (51, 51), 0)
    dnoise = (dnoise - dnoise.min()) / (dnoise.max() - dnoise.min())
    depth = 3.0 + 4.5 * dnoise  # strong variation: 3.0 .. 7.5
    return depth, texture


def world_points(depth, texture):
    """Back-project the frame-0 depth map into world points (frame 0 = identity)."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    z = depth
    x = (u - CX) / FX * z
    y = (v - CY) / FY * z
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    cols = texture.reshape(-1, 3)
    return pts, cols


def render_frame(pts, cols, w2c):
    """Project world points into a camera and z-buffer splat them."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    cam = pts @ R.T + t
    z = cam[:, 2]
    valid = z > 0.05
    u = np.round(FX * cam[:, 0] / np.maximum(z, 1e-6) + CX).astype(np.int64)
    v = np.round(FY * cam[:, 1] / np.maximum(z, 1e-6) + CY).astype(np.int64)
    valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # painter's algorithm: draw far points first, near points overwrite
    order = np.argsort(-z[valid])
    uu, vv = u[valid][order], v[valid][order]
    cc = cols[valid][order]

    img = np.zeros((H, W, 3), np.uint8)
    hit = np.zeros((H, W), bool)
    img[vv, uu] = cc
    hit[vv, uu] = True
    if not hit.all():
        # disocclusion holes at depth edges: cheap inpaint
        hole = (~hit).astype(np.uint8) * 255
        img = cv2.inpaint(img, hole, 3, cv2.INPAINT_TELEA)
    return img


def render_video(pose_str, rng):
    c2w = c2w_trajectory(pose_str)
    depth, texture = make_scene(rng)
    pts, cols = world_points(depth, texture)
    frames = []
    for T in c2w:
        frames.append(render_frame(pts, cols, np.linalg.inv(T)))
    return frames


def add_jitter(frames, rng, max_px=3):
    """Random per-frame translation, like an unstable autoregressive rollout."""
    out = []
    for f in frames:
        dx, dy = rng.integers(-max_px, max_px + 1, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
    return out


def add_decay(frames):
    """Progressive blur on the last 60% of frames, like quality collapse."""
    out = []
    n = len(frames)
    for i, f in enumerate(frames):
        frac = (i - 0.4 * n) / (0.6 * n)
        if frac > 0:
            sigma = 0.3 + 3.5 * frac
            k = int(2 * round(2 * sigma) + 1)
            f = cv2.GaussianBlur(f, (k, k), sigma)
        out.append(f)
    return out


def write_video(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed for {path} (mp4v codec unavailable?)")
    for f in frames:
        writer.write(f)
    writer.release()
    size_kb = os.path.getsize(path) / 1024
    print(f"wrote {path} ({len(frames)} frames, {size_kb:.0f} KB)")


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    specs = {
        "synthetic_pan.mp4": {
            "pose": "w*30",
            "notes": "clean forward dolly, 0.08 units per frame",
            "transform": None,
        },
        "synthetic_jittered.mp4": {
            "pose": "w*30",
            "notes": "same dolly with random +-3px per-frame shifts",
            "transform": add_jitter,
        },
        "synthetic_loop.mp4": {
            "pose": "w*10,a*10,s*10,d*10",
            "notes": "square loop, last frame returns to the first viewpoint",
            "transform": None,
        },
        "synthetic_decay.mp4": {
            "pose": "w*30",
            "notes": "dolly with progressive blur on the last 60% of frames",
            "transform": add_decay,
        },
    }

    gt = {"seed": SEED, "fps": FPS, "resolution": [W, H],
          "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
          "translation_step": 0.08, "videos": {}}

    for name, spec in specs.items():
        frames = render_video(spec["pose"], np.random.default_rng(SEED))
        if spec["transform"] is not None:
            extra = rng if spec["transform"] is add_jitter else None
            frames = spec["transform"](frames, extra) if extra is not None \
                else spec["transform"](frames)
        write_video(os.path.join(FIXTURE_DIR, name), frames)
        gt["videos"][name] = {"pose": spec["pose"], "notes": spec["notes"]}

    with open(os.path.join(FIXTURE_DIR, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)
    print("wrote ground_truth.json")


if __name__ == "__main__":
    main()
