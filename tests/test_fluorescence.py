from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_allclose

from markovids.pcl import fluo


def test_should_pack_named_parameters_when_gaussian_parameters_are_provided():
    values = [1, 2, 3, 4, 5, 6, 7]
    result = fluo.pack_params(*values)
    assert result == dict(zip(["x0", "y0", "sigma_x", "sigma_y", "theta", "amplitude", "offset"], values))


def test_should_peak_at_center_when_evaluating_rotated_gaussian():
    coords = (np.array([2., 3]), np.array([4., 4]))
    result = fluo.gaussian_2d(coords, 2, 4, 1, 2, np.pi / 4, 10, 3)
    assert result[0] == 13
    assert 3 < result[1] < 13


def test_should_recover_centroid_and_covariance_when_estimating_image_moments():
    image = np.zeros((5, 5))
    image[2, 1:4] = [1, 2, 1]
    center, covariance = fluo.estimate_gaussian_moments(image)
    assert_allclose(center, [2, 2])
    assert_allclose(covariance, [[.5, 0], [0, 0]])


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_should_select_closest_large_blob_when_multiple_candidates_exist(monkeypatch, dtype):
    image = np.arange(1600).reshape(40, 40).astype(dtype)
    far = np.array([[[1, 1]], [[1, 7]], [[7, 7]], [[7, 1]]], np.int32)
    near = far + 16
    small = np.array([[[0, 0]], [[1, 0]], [[0, 1]]], np.int32)
    monkeypatch.setattr(fluo.cv2, "findContours", Mock(return_value=([far, near, small], None)))
    mask = fluo.get_closest_blob(image)
    assert mask.dtype == bool
    assert mask[20, 20]
    assert not mask[3, 3]


@pytest.mark.parametrize("contours", [[], [np.array([[[1, 1]], [[2, 2]], [[3, 3]]], np.int32)]])
def test_should_return_none_when_no_valid_blob_is_available(monkeypatch, contours):
    image = np.zeros((20, 20), np.uint8)
    monkeypatch.setattr(fluo.cv2, "findContours", Mock(return_value=(contours, None)))
    result = fluo.get_closest_blob(image, min_size=0)
    assert result is None


@pytest.mark.parametrize("outcome", ["success", "unsuccessful", "runtime", "value"])
def test_should_return_fit_or_initial_guess_when_gaussian_optimizer_completes_or_fails(monkeypatch, outcome):
    import scipy.optimize
    y, x = np.indices((11, 11))
    image = np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 8) * 20 + 1
    values = np.array([5., 5, 2, 2, 0, 20, 1])

    def solve(residuals, **kwargs):
        assert residuals(values).shape == (121,)
        if outcome == "runtime":
            raise RuntimeError("no convergence")
        if outcome == "value":
            raise ValueError("invalid fit")
        return SimpleNamespace(success=outcome == "success", x=values)

    monkeypatch.setattr(scipy.optimize, "least_squares", Mock(side_effect=solve))
    result, initial = fluo.fit_2d_gaussian_with_moments(image, loss="huber")
    assert initial["x0"] == pytest.approx(5)
    assert initial["y0"] == pytest.approx(5)
    assert (result is not None) == (outcome == "success")
    if result is not None:
        assert_allclose(list(result.values()), values)
