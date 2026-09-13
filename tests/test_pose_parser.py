"""Pose parser must match the minWM convention exactly."""

import numpy as np
import pytest

from wmdrift.pose.parser import c2w_trajectory, parse_trajectory


def test_w19_gives_20_poses_first_identity():
    # same expectation as minWM's own test_wan21_inference.py
    vm = parse_trajectory("w*19")
    assert vm.shape == (20, 4, 4)
    assert np.allclose(vm[0], np.eye(4))


def test_forward_is_plus_z_at_008_per_step():
    c2w = c2w_trajectory("w*3")
    assert c2w.shape == (4, 4, 4)
    np.testing.assert_allclose(c2w[1][:3, 3], [0, 0, 0.08], atol=1e-9)
    np.testing.assert_allclose(c2w[3][:3, 3], [0, 0, 0.24], atol=1e-9)


def test_right_and_up_signs():
    c2w = c2w_trajectory("d*1")
    np.testing.assert_allclose(c2w[1][:3, 3], [0.08, 0, 0], atol=1e-9)
    c2w = c2w_trajectory("u*1")
    # up is -Y in OpenCV camera coords
    np.testing.assert_allclose(c2w[1][:3, 3], [0, -0.08, 0], atol=1e-9)


def test_chained_segments():
    c2w = c2w_trajectory("w*10,d*9")
    assert c2w.shape == (20, 4, 4)
    # after w*10 the camera is at z=0.8, then strafes right in world frame
    np.testing.assert_allclose(c2w[10][:3, 3], [0, 0, 0.8], atol=1e-9)
    np.testing.assert_allclose(c2w[-1][:3, 3], [0.72, 0, 0.8], atol=1e-9)


def test_yaw_rotates_in_place():
    c2w = c2w_trajectory("j*5")
    np.testing.assert_allclose(c2w[-1][:3, 3], [0, 0, 0], atol=1e-9)
    # 5 steps of 3 degrees = 15 degrees of yaw, translation untouched
    angle = np.arctan2(c2w[-1][0, 2], c2w[-1][2, 2])
    assert abs(angle) > 0


def test_loop_returns_home():
    c2w = c2w_trajectory("w*10,a*10,s*10,d*10")
    np.testing.assert_allclose(c2w[-1], np.eye(4), atol=1e-9)


def test_w2c_is_inverse_of_c2w():
    c2w = c2w_trajectory("w*4,a*3,l*2")
    w2c = parse_trajectory("w*4,a*3,l*2")
    for i in range(len(c2w)):
        np.testing.assert_allclose(w2c[i] @ c2w[i], np.eye(4), atol=1e-5)


def test_malformed_segment_raises():
    with pytest.raises(ValueError):
        parse_trajectory("forward*5")
    with pytest.raises(ValueError):
        parse_trajectory("w*x")
    with pytest.raises(ValueError):
        parse_trajectory("w5")
