"""Drift metric on the synthetic pan fixture.

Thresholds come from the actual fixture behavior (see scripts/make_fixtures.py),
not from wishful numbers: the fixture is rendered through the exact commanded
poses, so a working VO + Sim(3) pipeline should land close.
"""

import os

import pytest

from wmdrift.metrics.drift import drift_metrics
from wmdrift.video.io import load_video

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def pan_drift():
    frames, _ = load_video(os.path.join(FIXTURES, "synthetic_pan.mp4"))
    return drift_metrics(frames, "w*30")


def test_scale_is_near_commanded_step(pan_drift):
    # VO recovers unit-baseline steps; commanded steps are 0.08, so the
    # fitted Sim(3) scale should land near 0.08 (monocular scale ambiguity)
    assert 0.05 < pan_drift["scale"] < 0.12


def test_ate_small_after_alignment(pan_drift):
    # total trajectory is 2.4 units long; observed mean ATE ~0.04
    assert pan_drift["ate_mean"] < 0.15
    assert pan_drift["ate_final"] < 0.2


def test_rotation_error_near_zero_for_pure_translation(pan_drift):
    assert pan_drift["rot_err_mean_deg"] < 5.0
    assert pan_drift["rot_err_max_deg"] < 10.0


def test_no_unreliable_pairs_on_clean_fixture(pan_drift):
    assert pan_drift["unreliable_pairs"] == []


def test_too_few_frames_returns_none():
    frames, _ = load_video(os.path.join(FIXTURES, "synthetic_pan.mp4"))
    assert drift_metrics(frames[:2], "w*1") is None
