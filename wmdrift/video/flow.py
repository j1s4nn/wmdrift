"""Farneback dense optical flow, wrapped for the jitter metric."""

import cv2
import numpy as np
from tqdm import tqdm

from .io import resize_frames, to_gray


def dense_flow(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    """Farneback flow from gray_a to gray_b, returns (H, W, 2) float32."""
    return cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_magnitudes(frames: list, max_width: int = 256, show_progress: bool = True):
    """Mean flow magnitude per consecutive frame pair.

    Frames are downscaled to max_width first because Farneback is slow on
    full-res video and the jitter metric only needs the magnitude series.

    Returns:
        (mags, full_mags): mags is (T-1,) mean magnitude per pair;
        full_mags is a list of (H, W) magnitude arrays, one per pair.
    """
    small = resize_frames(frames, max_width)
    grays = to_gray(small)
    mags = np.zeros(len(grays) - 1, dtype=np.float64)
    full_mags = []
    pairs = range(len(grays) - 1)
    if show_progress:
        pairs = tqdm(pairs, desc="optical flow", leave=False)
    for i in pairs:
        flow = dense_flow(grays[i], grays[i + 1])
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags[i] = mag.mean()
        full_mags.append(mag)
    return mags, full_mags
