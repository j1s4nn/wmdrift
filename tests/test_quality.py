"""Quality decay on the blur fixture.

Note: even the clean pan loses sharpness over time (~-14/frame) because a
forward dolly magnifies the scene -- the slope conflates zoom with blur.
The test therefore compares decay against the pan instead of expecting a
flat clean slope. The loop fixture (net zero motion) is the flat control.
"""

import os

import pytest

from wmdrift.metrics.quality import quality_decay, sharpness_series
from wmdrift.video.io import load_video

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _quality(name):
    frames, _ = load_video(os.path.join(FIXTURES, name))
    return quality_decay(sharpness_series(frames))


@pytest.fixture(scope="module")
def qualities():
    return {n: _quality(f"synthetic_{n}.mp4") for n in ["pan", "decay", "loop"]}


def test_decay_slope_strongly_negative(qualities):
    assert qualities["decay"]["slope"] < -40


def test_decay_much_steeper_than_pan(qualities):
    # observed: decay ~-79/frame vs pan ~-14/frame (pan loss is zoom, not blur)
    assert qualities["decay"]["slope"] < 3 * qualities["pan"]["slope"]


def test_decay_ends_blurry(qualities):
    assert qualities["decay"]["sharpness"][-1] < 0.05 * qualities["decay"]["sharpness"][0]


def test_loop_slope_flat(qualities):
    # net-zero motion: no zoom artifact, sharpness stays put
    assert abs(qualities["loop"]["slope"]) < 5


def test_decay_collapse_fires(qualities):
    assert qualities["decay"]["collapse_frame"] is not None
