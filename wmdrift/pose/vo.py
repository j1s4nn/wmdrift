"""Minimal visual odometry: ORB + essential matrix + pose recovery.

Assumes a pinhole camera with fx = fy = frame width and a centered principal
point. That matches the synthetic fixtures (fx=fy=256 at 256x256) and is a
reasonable guess for square model outputs; for anything else the recovered
trajectory is only qualitatively right.
"""

import cv2
import numpy as np
from tqdm import tqdm

# TODO: handle low-texture scenes better than just flagging them


def intrinsics(width: int, height: int) -> np.ndarray:
    f = float(width)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]])


def _match_pair(gray_a, gray_b, orb, matcher, K):
    """Relative pose between two frames. Returns (4x4 transform, n_inliers, n_matches).

    The transform maps camera-1 coords to camera-2 coords (p2 = M[:3,:3] p1 + M[:3,3]).
    """
    kpa, des_a = orb.detectAndCompute(gray_a, None)
    kpb, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kpa) < 8 or len(kpb) < 8:
        return None, 0, 0

    pairs = matcher.knnMatch(des_a, des_b, k=2)
    good = []
    for p in pairs:
        if len(p) == 2 and p[0].distance < 0.75 * p[1].distance:
            good.append(p[0])
    if len(good) < 8:
        return None, 0, len(good)

    pts1 = np.float64([kpa[m.queryIdx].pt for m in good])
    pts2 = np.float64([kpb[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        # findEssentialMat can return stacked candidates for degenerate input
        return None, 0, len(good)
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    inliers = int(mask.sum()) if mask is not None else 0
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t.flatten()
    return M, inliers, len(good)


def run_vo(frames: list, min_inlier_ratio: float = 0.3,
           show_progress: bool = True) -> dict:
    """Chain pairwise relative poses into a recovered c2w trajectory.

    Pairs whose RANSAC inlier ratio is below min_inlier_ratio are treated as
    unreliable: the previous pose is carried forward and the pair index is
    recorded so the report can mention the gap.

    Returns:
        dict with c2w (T,4,4), per-pair inlier counts, unreliable pair indices.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape
    K = intrinsics(w, h)
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    c2w = [np.eye(4)]
    inlier_counts = []
    unreliable = []

    pairs = range(len(grays) - 1)
    if show_progress:
        pairs = tqdm(pairs, desc="visual odometry", leave=False)
    for i in pairs:
        M, inliers, n_matches = _match_pair(grays[i], grays[i + 1], orb, matcher, K)
        ratio = inliers / n_matches if n_matches else 0.0
        inlier_counts.append({"pair": i, "inliers": inliers,
                              "matches": n_matches, "ratio": ratio})
        if M is None or ratio < min_inlier_ratio:
            unreliable.append(i)
            c2w.append(c2w[-1].copy())
            continue
        # M maps cam_i -> cam_{i+1}, so c2w_{i+1} = c2w_i @ inv(M)
        c2w.append(c2w[-1] @ np.linalg.inv(M))

    return {
        "c2w": np.stack(c2w),
        "inlier_counts": inlier_counts,
        "unreliable_pairs": unreliable,
        "K": K,
    }
