"""Camera drift: commanded vs VO-recovered trajectory.

Pipeline: pose string -> commanded c2w (minWM convention), video -> ORB VO
recovered c2w, then Sim(3)-align and measure ATE + rotation error. The
fitted scale is part of the result: without it the comparison is meaningless
for a monocular model output.
"""

from ..pose.align import align_trajectory
from ..pose.parser import c2w_trajectory
from ..pose.vo import run_vo


def drift_metrics(frames: list, pose_str: str,
                  min_inlier_ratio: float = 0.3) -> dict:
    """Run VO on frames and compare against the commanded trajectory.

    Returns None if the video has too few frames or VO finds nothing.
    """
    cmd = c2w_trajectory(pose_str)
    if len(frames) < 3:
        return None

    vo = run_vo(frames, min_inlier_ratio=min_inlier_ratio)
    n = min(len(vo["c2w"]), len(cmd))
    if n < 3:
        return None

    aligned = align_trajectory(vo["c2w"], cmd)
    ate = aligned["ate"]

    return {
        "scale": aligned["scale"],
        "ate_mean": float(ate.mean()),
        "ate_max": float(ate.max()),
        "ate_final": float(ate[-1]),
        "ate": ate,
        "rot_err_deg": aligned["rot_err_deg"],
        "rot_err_mean_deg": float(aligned["rot_err_deg"].mean()),
        "rot_err_max_deg": float(aligned["rot_err_deg"].max()),
        "cmd_positions": cmd[:n, :3, 3],
        "rec_positions_aligned": aligned["aligned_c2w"][:n, :3, 3],
        "unreliable_pairs": vo["unreliable_pairs"],
        "inlier_counts": vo["inlier_counts"],
        "n_frames": n,
    }
