"""Loop-closure memory metric.

LPIPS downloads AlexNet weights on first use; if that fails (offline CI,
proxy) the test falls back to asserting the SSIM backend result.
"""

import os

import pytest

from wmdrift.metrics.memory import is_loop, loop_closure_error
from wmdrift.video.io import load_video

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_loop_detection_from_pose_string():
    assert is_loop("w*10,a*10,s*10,d*10")
    assert not is_loop("w*30")


def test_loop_fixture_closes():
    frames, _ = load_video(os.path.join(FIXTURES, "synthetic_loop.mp4"))
    r = loop_closure_error(frames[0], frames[-1])
    # first and last frame are the same viewpoint; only mp4 compression differs
    assert r["ssim"] > 0.9
    if r["backend"] == "lpips-alex":
        assert r["lpips"] < 0.05
    else:
        pytest.skip(f"lpips unavailable: {r.get('note')}")


def test_pan_is_not_a_loop():
    assert not is_loop("w*30")
