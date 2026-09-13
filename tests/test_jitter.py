"""Jitter metric must separate the jittered fixture from the clean pan."""

import os

import pytest

from wmdrift.metrics.jitter import jitter_score
from wmdrift.video.flow import flow_magnitudes
from wmdrift.video.io import load_video

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _score(name):
    frames, fps = load_video(os.path.join(FIXTURES, name))
    mags, _ = flow_magnitudes(frames, show_progress=False)
    return jitter_score(mags, fps)


@pytest.fixture(scope="module")
def scores():
    return _score("synthetic_pan.mp4"), _score("synthetic_jittered.mp4")


def test_jittered_hf_rms_far_above_clean(scores):
    clean, jittered = scores
    # observed: clean ~0.026, jittered ~0.41 flow units
    assert jittered["hf_rms"] > 5 * clean["hf_rms"]


def test_jittered_ratio_score_above_clean(scores):
    clean, jittered = scores
    assert jittered["score"] > clean["score"]


def test_short_series_is_nan():
    import numpy as np
    r = jitter_score(np.array([1.0, 2.0, 3.0]), 8.0)
    assert r["score"] != r["score"]  # nan
    assert r["note"] is not None
