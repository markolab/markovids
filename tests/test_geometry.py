from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from markovids.pcl import io, registration as reg


@pytest.fixture
def intrinsics():
    return np.array([[2., 0, 1], [0, 4, 1], [0, 0, 1]])


@pytest.fixture
def cloud_points():
    return np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 2, 3]])


@pytest.mark.parametrize("project", [True, False])
def test_should_project_valid_depth_and_apply_scale_when_creating_cloud(intrinsics, project):
    image = np.array([[4., np.nan], [0, 8]])
    expected = [[-.5, -.25, 1], [0, 0, 2]] if project else [[0, 0, 1], [1, 1, 2]]
    result = io.pcl_from_depth(image, intrinsics, project_xy=project, post_scale=2)
    assert_allclose(result, np.array(expected) / 2)
    assert_allclose(image, [[4, np.nan], [0, 8]], equal_nan=True)


def test_should_use_raw_depth_for_xy_when_shifting_z(intrinsics):
    image = np.array([[4., 8], [12, 20]])
    result = io.pcl_from_depth(image, intrinsics, post_z_shift=4)
    assert_allclose(result, [[-.5, -.25, 3], [0, -.5, 2], [-1.5, 0, 1]])


@pytest.mark.parametrize("tensor", [False, True])
def test_should_rasterize_point_cloud_when_given_legacy_or_tensor_container(intrinsics, tensor):
    points = np.array([[0., 0, 2], [-.5, -.25, 1]])
    cloud = SimpleNamespace(point=SimpleNamespace(positions=SimpleNamespace(numpy=lambda: points))) if tensor else SimpleNamespace(points=points)
    result = io.depth_from_pcl(cloud, intrinsics, width=3, height=3, buffer=2)
    assert result.shape == (5, 5)
    assert result[2, 2] == 8
    assert result[1, 1] == 4
    assert np.isnan(result[0, 0])


def test_should_spread_points_and_adjust_depth_when_transform_correction_is_enabled(intrinsics):
    cloud = SimpleNamespace(points=np.array([[1., 1, 2]]))
    result = io.depth_from_pcl(cloud, intrinsics, width=3, height=3, buffer=4, project_xy=False, post_scale=2, transform_correct=True, z_adjust=20)
    assert_allclose(result[3:6, 3:6], 4)
    assert result[0, 0] == 20


def test_should_convert_pixel_coordinates_when_projecting_with_floor_distance():
    uvz = np.array([[2., 4, 8]])
    result = io.project_world_coordinates(uvz, floor_distance=16, cx=0, cy=0, fx=2, fy=4)
    assert_allclose(result, [[2, 2, 2]])


def test_should_round_trip_coordinates_when_converting_between_pixels_and_cloud(intrinsics):
    uvz = np.array([[0., 0, 4], [1, 2, 8]])
    xyz = io.pxl_to_pcl_coords(uvz, intrinsics)
    u, v, z = io.pcl_to_pxl_coords(xyz.copy(), intrinsics)
    projected = io.project_world_coordinates(uvz, cx=1, cy=1, fx=2, fy=4)
    assert_allclose(np.column_stack((u, v, z)), uvz)
    assert_allclose(projected, xyz)


def test_should_filter_nonpositive_shifted_depth_when_projecting_keypoints(intrinsics):
    uvz = np.array([[1., 2, 4], [3, 4, 20], [1, 1, np.nan]])
    result = io.pxl_to_pcl_coords(uvz, intrinsics, project_xy=False, post_z_shift=4)
    assert_allclose(result, [[1, 2, 3]])


def test_should_reverse_height_shift_when_converting_cloud_to_pixels(intrinsics):
    points = np.array([[2., 3, 1]])
    result = io.pcl_to_pxl_coords(points, intrinsics, project_xy=False, post_z_shift=16)
    assert_allclose(result, ([2], [3], [12]))


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=UnboundLocalError, reason='pcl_to_pxl_coords scales undefined points instead of its input coordinates')
def test_should_scale_cloud_coordinates_when_post_scale_is_supplied(intrinsics):
    points = np.array([[1., 1, 2]])
    original = points.copy()
    u, v, z = io.pcl_to_pxl_coords(points, intrinsics, post_scale=2)
    assert_allclose(z, [16])
    assert_array_equal(points, original)


@pytest.mark.parametrize("buffer", [0, (0, 0)])
def test_should_interpolate_depth_and_mask_distant_pixels_when_cloud_is_sparse(intrinsics, buffer):
    cloud = SimpleNamespace(points=np.array([[1., 1, 10], [4, 1, 10], [1, 4, 10], [4, 4, 10], [np.nan, 0, 0]]))
    result = io.depth_from_pcl_interpolate(cloud, intrinsics, width=6, height=6, project_xy=False, z_scale=1, buffer=buffer, distance_threshold=1.1, z_adjust=20)
    assert result.shape == (6, 6)
    assert result[1, 1] == 10
    assert np.isnan(result[2, 2])
    assert np.isnan(result[0, 0])


def test_should_return_empty_depth_when_interpolation_backend_raises(intrinsics, monkeypatch):
    import scipy.interpolate
    cloud = SimpleNamespace(points=np.array([[1., 1, 2], [3, 3, 2]]))
    monkeypatch.setattr(scipy.interpolate, "griddata", Mock(side_effect=ValueError("degenerate cloud")))
    result = io.depth_from_pcl_interpolate(cloud, intrinsics, width=5, height=5, buffer=0, project_xy=False)
    assert np.isnan(result).all()


def test_should_remove_distant_outliers_when_trimming_cloud():
    points = np.array([[-1., 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [100, 100, 100]])
    result = io.trim_outliers(points)
    assert_array_equal(result, points[:4])


@pytest.mark.parametrize("similarity", [False, True])
def test_should_recover_known_transform_when_points_are_rotated_and_translated(cloud_points, similarity):
    rotation = Rotation.from_euler("z", .5).as_matrix()
    translation = np.array([2., -4, 1])
    scale = 2 if similarity else 1
    target = scale * cloud_points @ rotation.T + translation
    if similarity:
        result_scale, result_rotation, result_translation = reg.estimate_similarity_transform(cloud_points, target)
    else:
        result_rotation, result_translation = reg.estimate_rigid_transform(cloud_points, target)
        result_scale = 1
    assert_allclose(result_scale, scale)
    assert_allclose(result_rotation, rotation, atol=1e-14)
    assert_allclose(result_translation, translation)


@pytest.mark.parametrize("estimator", [reg.estimate_rigid_transform, reg.estimate_similarity_transform])
def test_should_correct_reflection_when_target_has_negative_determinant(cloud_points, estimator):
    target = cloud_points * [-1, 1, 1]
    result = estimator(cloud_points, target)
    assert np.linalg.det(result[-2]) == pytest.approx(1)


def test_should_reject_mismatched_shapes_when_estimating_similarity(cloud_points):
    target = cloud_points[:-1]
    with pytest.raises(AssertionError) as error:
        reg.estimate_similarity_transform(cloud_points, target)
    assert error.value is not None


@pytest.mark.parametrize("similarity", [False, True])
def test_should_apply_weights_to_residuals_when_computing_registration_loss(cloud_points, similarity):
    weights = np.array([0., 1, 2, 3, 4])
    parameters = np.zeros(14 if similarity else 12)
    if similarity:
        parameters[[6, 13]] = 1
    residual = reg.residuals_similarity_fixed if similarity else reg.residuals_rigid
    result = residual(parameters, cloud_points, cloud_points + 1, cloud_points - 2, weights, weights)
    assert_allclose(result[:15], np.repeat(-weights, 3))
    assert_allclose(result[15:], np.repeat(2 * weights, 3))


@pytest.mark.parametrize("method", ["estimate_transform", "bundle_adjust_rigid_fixed_structure", "bundle_adjust_fixed_structure_similarity"])
def test_should_return_inverse_camera_transforms_when_registering_three_clouds(cloud_points, monkeypatch, method):
    b, c = cloud_points + [3, 0, 0], cloud_points + [0, 4, 0]
    x = np.zeros(14 if "similarity" in method else 12)
    if len(x) == 14:
        x[[6, 13]] = 1
        x[3] = 3
        x[11] = 4
    else:
        x[3] = 3
        x[10] = 4
    solver = Mock(return_value=SimpleNamespace(x=x))
    monkeypatch.setattr(reg, "least_squares", solver)
    result = getattr(reg, method)(cloud_points, b, c)
    assert result["points_3d"] is cloud_points
    assert_allclose(result["B_to_A"]["t"], [-3, 0, 0], atol=1e-12)
    assert_allclose(result["C_to_A"]["t"], [0, -4, 0], atol=1e-12)
    assert_allclose(result["B_to_A"]["R"], np.eye(3), atol=1e-12)
    assert solver.call_count == (method != "estimate_transform")
    if solver.called:
        assert solver.call_args.kwargs["loss"] == "huber"


def test_should_round_trip_similarity_when_inverting_transform():
    rotation = Rotation.from_euler("x", .4).as_matrix()
    translation = np.array([1., 2, 3])
    point = np.array([4., 5, 6])
    inverse_rotation, inverse_translation, inverse_scale = reg.invert_similarity_transform(rotation, translation, 2)
    result = inverse_scale * inverse_rotation @ (2 * rotation @ point + translation) + inverse_translation
    assert_allclose(result, point)
