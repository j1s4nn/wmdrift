"""Temporal jitter from optical-flow magnitudes.

A stable rollout has a smooth flow-magnitude curve; frame-to-frame camera
jitter shows up as high-frequency energy. Score = share of spectral power
above fs/4 (half of Nyquist), after removing the linear trend so a steady
acceleration does not count as jitter.
"""

import numpy as np


def jitter_score(mags: np.ndarray, fps: float) -> dict:
    """Compute the jitter score for a flow-magnitude series.

    score is the spec's ratio: power above fs/4 over total power, on the
    linearly detrended series. The ratio alone turned out to be a weak
    discriminator (a nearly flat residual still has a moderate ratio), so
    hf_rms -- the high-frequency amplitude in flow-magnitude units -- is
    reported alongside it.

    Args:
        mags: (T-1,) mean flow magnitude per frame pair.
        fps: video fps, used for the frequency axis.

    Returns:
        dict with score, hf_rms, freqs, power, detrended series.
    """
    mags = np.asarray(mags, dtype=np.float64)
    n = len(mags)
    if n < 8:
        return {"score": float("nan"), "hf_rms": float("nan"),
                "freqs": np.array([]), "power": np.array([]),
                "detrended": mags, "note": "series too short for spectral analysis"}

    # remove linear trend
    x = np.arange(n)
    slope, intercept = np.polyfit(x, mags, 1)
    detrended = mags - (slope * x + intercept)

    spectrum = np.fft.rfft(detrended)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    total = power.sum()
    high_mask = freqs > fps / 4
    hf_rms = float(np.sqrt(power[high_mask].sum() * 2 / n ** 2))
    if total <= 0:
        return {"score": 0.0, "hf_rms": hf_rms, "freqs": freqs, "power": power,
                "detrended": detrended, "note": None}
    high = power[high_mask].sum()
    return {"score": float(high / total), "hf_rms": hf_rms,
            "freqs": freqs, "power": power, "detrended": detrended, "note": None}
