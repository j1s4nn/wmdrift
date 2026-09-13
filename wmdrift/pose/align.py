"""Sim(3) alignment (Umeyama) + trajectory errors.

Monocular VO recovers trajectory shape but not metric scale (this is exactly
minWM issue #18: the model's translation units are arbitrary). Umeyama fits
rotation + scale + translation between recovered and commanded positions, and
the fitted scale is reported alongside the errors instead of being hidden.
"""

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Fit s, R, t minimizing ||dst - (s R src + t)||^2.

    Args:
        src: (N, 3) recovered positions.
        dst: (N, 3) commanded positions.

    Returns:
        (scale, R (3,3), t (3,)).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    var_s = (src_c ** 2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def align_trajectory(rec_c2w: np.ndarray, cmd_c2w: np.ndarray) -> dict:
    """Align recovered poses to commanded poses with Sim(3).

    Positions are aligned with the fitted s, R, t. Rotation errors compare
    the raw recovered and commanded rotations directly: both trajectories
    start at identity in the frame-0 camera frame, so they are already in
    the same frame, and for straight-line trajectories the Umeyama rotation
    is degenerate (arbitrary spin around the line) which would poison the
    rotation error.

    Returns dict with fitted scale/R/t, aligned recovered c2w, per-frame
    translation error (ATE) and rotation geodesic error in degrees.
    """
    n = min(len(rec_c2w), len(cmd_c2w))
    rec = rec_c2w[:n]
    cmd = cmd_c2w[:n]

    scale, R, t = umeyama(rec[:, :3, 3], cmd[:, :3, 3])

    aligned = np.array(rec)
    aligned[:, :3, 3] = scale * (rec[:, :3, 3] @ R.T) + t

    ate = np.linalg.norm(aligned[:, :3, 3] - cmd[:, :3, 3], axis=1)

    rot_err = np.zeros(n)
    for i in range(n):
        dR = cmd[i, :3, :3].T @ rec[i, :3, :3]
        cos = (np.trace(dR) - 1) / 2
        rot_err[i] = np.degrees(np.arccos(np.clip(cos, -1, 1)))

    return {
        "scale": scale,
        "R": R,
        "t": t,
        "aligned_c2w": aligned,
        "ate": ate,
        "rot_err_deg": rot_err,
        "n_frames": n,
    }
