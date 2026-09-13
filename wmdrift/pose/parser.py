"""minWM pose-string parsing.

Conventions copied from minWM/minwm/processors/camera.py so trajectories match
what the model was conditioned on. Each step is 0.08 units of translation or
3 degrees of rotation, applied in the camera-local frame; the first frame is
identity and later frames chain from it.

String format: comma-separated ``key*N`` segments, e.g. ``"w*10,a*5"``.
Keys: w/s forward-back, a/d left-right, u/dn up-down, j/l yaw, i/k pitch.
"""

import re

import numpy as np

_STEP = 0.08
_ROT_STEP = np.radians(3.0)

_MOTIONS = {
    "w": {"forward": _STEP},
    "s": {"forward": -_STEP},
    "d": {"right": _STEP},
    "a": {"right": -_STEP},
    "u": {"up": _STEP},
    "dn": {"up": -_STEP},
    "j": {"yaw": -_ROT_STEP},
    "l": {"yaw": _ROT_STEP},
    "i": {"pitch": _ROT_STEP},
    "k": {"pitch": -_ROT_STEP},
}


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def parse_motions(traj_str: str) -> list:
    """Parse a trajectory string into a flat list of per-step motion dicts."""
    segments = traj_str.strip().split(",")
    motions = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        m = re.fullmatch(r"([a-z]+)\*(\d+)", seg)
        if m is None:
            raise ValueError(f"Cannot parse trajectory segment: '{seg}'. Expected 'w*19'.")
        key, n = m.group(1), int(m.group(2))
        if key not in _MOTIONS:
            raise ValueError(f"Unknown direction '{key}'. Valid: {list(_MOTIONS.keys())}")
        motions.extend([_MOTIONS[key]] * n)
    return motions


def c2w_trajectory(traj_str: str) -> np.ndarray:
    """Build ``(T, 4, 4)`` camera-to-world poses, first frame identity.

    Same chaining as minWM's ``_generate_c2w_trajectory``: rotations
    premultiply in the local frame, translations are rotated into world
    before being added.
    """
    motions = parse_motions(traj_str)
    T = np.eye(4)
    poses = [T.copy()]
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward:
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, forward])
        right = move.get("right", 0.0)
        if right:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0, 0])
        up = move.get("up", 0.0)
        if up:
            # up in camera frame = -Y (OpenCV Y-down)
            T[:3, 3] += T[:3, :3] @ np.array([0, -up, 0])
        poses.append(T.copy())
    return np.stack(poses)


def parse_trajectory(traj_str: str) -> np.ndarray:
    """Parse a trajectory string into ``(T, 4, 4)`` w2c view matrices.

    Mirrors minWM's ``parse_trajectory``: build c2w, invert each frame.
    """
    c2w = c2w_trajectory(traj_str)
    w2c = np.zeros_like(c2w)
    for i in range(len(c2w)):
        w2c[i] = np.linalg.inv(c2w[i])
    return w2c
