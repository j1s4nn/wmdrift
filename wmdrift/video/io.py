"""Video loading with OpenCV."""

import cv2
import numpy as np


def load_video(path: str) -> tuple[list, float]:
    """Read a video file into a list of BGR uint8 frames plus its fps.

    Raises:
        IOError: if the file cannot be opened or contains no frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # zero or nan
        fps = 16.0  # minWM demo default, only used for plotting
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise IOError(f"Video has no decodable frames: {path}")
    return frames, float(fps)


def to_gray(frames: list) -> list:
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]


def resize_frames(frames: list, max_width: int) -> list:
    """Downscale frames so width <= max_width, keeping aspect ratio."""
    out = []
    for f in frames:
        h, w = f.shape[:2]
        if w > max_width:
            scale = max_width / w
            out.append(cv2.resize(f, (max_width, int(round(h * scale)))))
        else:
            out.append(f)
    return out


def frames_to_array(frames: list) -> np.ndarray:
    return np.stack(frames)
