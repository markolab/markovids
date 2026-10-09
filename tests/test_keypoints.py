from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.pcl import kpoints as kp


@pytest.fixture
def constraints():
    return kp.BoneConstraints([(0, 1), (1, 2)], np.array([10., 10]), np.array([1., 1]), ["a", "b", "c"])


def test_should_map_names_and_connections_when_bone_constraints_are_defined(constraints):
    expected = ([0, 1], [1, 2])
    index = constraints.get_keypoint_index("b")
    endpoints = constraints.get_connection_indices()
    assert index == 1
    assert_array_equal(endpoints, expected)


def test_should_reject_unknown_name_when_looking_up_bone_keypoint(constraints):
    name = "missing"
    with pytest.raises(ValueError, match="missing") as error:
        constraints.get_keypoint_index(name)
    assert error.value is not None


def test_should_correct_only_violating_bones_when_optimizing_sequence(constraints):
    optimizer = kp.BoneConstraintOptimizer(constraints, iterations=3, correction_rate=1)
    points = np.array([[[0., 0, 0], [20, 0, 0], [30, 0, 0]], [[0, 0, 0], [10, 0, 0], [np.nan] * 3]])
    original = points.copy()
    confidence = np.ones((2, 3))
    result, scores = optimizer.process_sequence(points, confidence)
    lengths = optimizer.compute_bone_lengths(result)
    assert abs(lengths[0, 0] - 10) < abs(20 - 10)
    assert_allclose(lengths[1, 0], 10)
    assert np.isnan(result[1, 2]).all()
    assert scores[1, 2] == 0
    assert np.all(scores <= confidence)
    assert_allclose(points, original, equal_nan=True)


@pytest.mark.parametrize("robust", [True, False])
def test_should_estimate_lengths_and_ignore_invalid_samples_when_bone_data_is_noisy(robust):
    lengths = np.concatenate((np.linspace(9, 11, 30), [150, 0, 200, np.nan]))
    points = np.zeros((len(lengths), 2, 3))
    points[:, 1, 0] = lengths
    confidence = np.ones((len(points), 2))
    confidence[0] = .1
    result = kp.estimate_bone_lengths_from_data(points, ["a", "b"], [("a", "b"), ("a", "missing")], confidence, min_samples=10, use_robust=robust)
    assert list(result) == [("a", "b")]
    mean, deviation = result[("a", "b")]
    assert 9 < mean < 11
    assert 0 < deviation < 2


@pytest.mark.parametrize("count, expected", [(9, {}), (10, {("a", "b"): (10., 1.)})])
def test_should_require_ten_samples_when_estimating_constant_bone_lengths(count, expected):
    points = np.zeros((count, 2, 3))
    points[:, 1, 0] = 10
    result = kp.estimate_bone_lengths_from_data(points, ["a", "b"], [("a", "b")], use_robust=False)
    assert result == expected


def test_should_build_only_measured_connections_when_creating_bone_constraints(monkeypatch):
    estimate = Mock(return_value={("b", "a"): (12., 2.)})
    monkeypatch.setattr(kp, "estimate_bone_lengths_from_data", estimate)
    result = kp.create_bone_constraints_from_data(np.zeros((1, 3, 3)), ["a", "b", "c"], [("b", "a"), ("b", "c")], method="mean")
    assert result.connections == [(1, 0)]
    assert_array_equal(result.target_lengths, [12])
    assert_array_equal(result.length_stds, [2])
    assert estimate.call_args.kwargs["use_robust"] is False


@pytest.mark.parametrize("frames", [1, 2, 3, 4, 5])
def test_should_build_derivatives_with_correct_dimensions_when_trajectory_is_short(frames):
    regularizer = kp.TemporalRegularization(fps=2)
    matrices = regularizer.build_difference_matrices(frames)
    for order in range(1, 5):
        matrix = matrices[f"D{order}"]
        assert matrix.shape == (max(0, frames - order), frames)
        assert_allclose(matrix @ np.ones(frames), 0)
    if frames > 1:
        assert_allclose(matrices["D1"] @ np.arange(frames), -2)


@pytest.mark.parametrize("mask, gaps", [([], []), ([True, True], []), ([False, False], [(0, 2)]), ([False, True, False, False, True, False], [(0, 1), (2, 4), (5, 6)])])
def test_should_find_half_open_gaps_when_mask_contains_missing_frames(mask, gaps):
    regularizer = kp.TemporalRegularization()
    result = regularizer._find_gaps(np.array(mask, dtype=bool))
    assert result == gaps


def test_should_return_missing_data_and_zero_confidence_when_no_observation_is_valid():
    regularizer = kp.TemporalRegularization()
    points = np.full((4, 3), np.nan)
    result, confidence = regularizer.optimize_trajectory(points, np.ones(4))
    assert np.isnan(result).all()
    assert result is not points
    assert_array_equal(confidence, 0)


@pytest.mark.parametrize("gaps", [[0], [3], [7], [2, 3, 4, 5]])
def test_should_fill_short_gaps_and_preserve_long_gaps_when_regularizing(gaps):
    regularizer = kp.TemporalRegularization(fps=1, lambda_velocity=.1, lambda_accel=.1, lambda_jerk=.1, lambda_snap=.1)
    points = np.repeat(np.arange(8, dtype=float)[:, None], 3, axis=1)
    points[gaps] = np.nan
    original = points.copy()
    result, confidence = regularizer.optimize_trajectory(points, np.ones(8), max_gap_fill=2)
    assert np.isnan(result[gaps]).all() if len(gaps) > 2 else np.isfinite(result).all()
    assert np.all(confidence[gaps] < 1)
    assert_allclose(confidence[[i for i in range(8) if i not in gaps]], 1)
    assert_allclose(points, original, equal_nan=True)


def test_should_fall_back_to_linear_filling_when_sparse_solver_raises(monkeypatch):
    regularizer = kp.TemporalRegularization()
    points = np.array([[0., 0, 0], [np.nan] * 3, [2, 4, 6]])
    monkeypatch.setattr(kp, "spsolve", Mock(side_effect=RuntimeError("singular")))
    with pytest.warns(UserWarning, match="Sparse solve failed") as warnings:
        result, _ = regularizer.optimize_trajectory(points, np.ones(3))
    assert len(warnings) == 3
    assert_allclose(result[1], [1, 2, 3])


def test_should_keep_single_observations_when_dimension_cannot_be_interpolated():
    points = np.array([[np.nan, 1, np.nan], [2, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    regularizer = kp.TemporalRegularization()
    result, _ = regularizer.optimize_trajectory(points, np.ones(3), mask=np.array([True, True, False]))
    assert_allclose(result, points, equal_nan=True)


def test_should_use_minimum_confidence_when_no_valid_mask_entry_exists():
    regularizer = kp.TemporalRegularization()
    result = regularizer._propagate_confidence(np.zeros(3), np.zeros(3, dtype=bool), 3)
    assert_allclose(result, .1)


@pytest.mark.parametrize("frames", [3, 13])
@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=TypeError, reason='sliding optimizer does not accept or forward max_gap_fill')
def test_should_blend_overlapping_windows_when_regularizing_long_sequences(monkeypatch, frames):
    regularizer = kp.SlidingWindowTemporalRegularization(window_size=6, overlap=4)
    points = np.repeat(np.arange(frames, dtype=float)[:, None], 3, axis=1)
    confidence = np.ones(frames)
    optimizer = Mock(side_effect=lambda data, scores, mask, **kwargs: (data.copy(), scores.copy()))
    monkeypatch.setattr(regularizer.base_regularizer, "optimize_trajectory", optimizer)
    result, scores = regularizer.optimize_trajectory(points, confidence, max_gap_fill=2)
    assert_allclose(result, points)
    assert_allclose(scores, 1)
    assert optimizer.call_count == (1 if frames <= 6 else 5)
    assert all(call.kwargs == {"max_gap_fill": 2} for call in optimizer.call_args_list)


@pytest.mark.parametrize("median", [False, True])
@pytest.mark.parametrize("valid_body", [False, True])
def test_should_center_on_body_or_fallback_to_available_points_when_aligning(median, valid_body):
    aligner = kp.PoseAligner(["back_top", "left_hip", "body", "snout"], use_median_centering=median)
    points = np.array([[0., 0, 8], [6, 0, 9], [12, 0, 10], [30, 0, 11]])
    if not valid_body:
        points[:3] = np.nan
    centroid = aligner._compute_2d_centroid(points)
    expected = (6 if median else 4) if valid_body else 30
    assert_allclose(centroid, [expected, 0, 0])


@pytest.mark.parametrize("median", [False, True])
def test_should_use_origin_when_all_points_are_missing(median):
    aligner = kp.PoseAligner(["body"], use_median_centering=median, center_keypoint_weights={"body": 2})
    result = aligner._compute_2d_centroid(np.full((1, 3), np.nan))
    assert_array_equal(result, 0)


@pytest.mark.parametrize("names, points, expected", [(["snout", "tail_tip"], [[0., 4, 1], [0, 0, 1]], np.pi / 2), (["back_top", "back_middle_upper"], [[0., 0, 1], [2, 0, 1]], 0), (["body"], [[0., 0, 1]], 0)])
def test_should_use_anterior_posterior_axis_or_spine_fallback_when_computing_orientation(names, points, expected):
    aligner = kp.PoseAligner(names)
    result = aligner._compute_2d_orientation(np.array(points))
    if names == ["back_top", "back_middle_upper"]:
        # SVD determines an axis; its sign is mathematically ambiguous.
        assert abs(np.cos(result)) == pytest.approx(1)
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize("explicit", [True, False])
def test_should_round_trip_xy_and_preserve_z_when_transforming_pose(explicit):
    aligner = kp.PoseAligner(["snout", "tail_tip", "body"], exclude_from_center=[])
    points = np.tile(np.array([[[0., 4, 8], [0, 0, 9], [0, 2, 10]]]), (8, 1, 1))
    centers, angles = aligner.compute_alignment(points, alignment_window=3)
    kwargs = {"centroids": centers, "angles": angles} if explicit else {}
    transformed = aligner.transform(points, **kwargs)
    restored = aligner.inverse_transform(transformed, **kwargs)
    assert_allclose(restored, points, atol=1e-14)
    assert_allclose(transformed[:, :, 2], points[:, :, 2])
    assert_allclose(transformed[:, 0, 1], 0, atol=1e-14)


@pytest.mark.parametrize("transform_z", [False, True])
@pytest.mark.parametrize("set_zero", [False, True])
def test_should_impute_missing_values_and_preserve_observations_when_fitting_pca(transform_z, set_zero):
    data = np.arange(48, dtype=float).reshape(8, 2, 3) + 1
    data[3, 0] = np.nan
    original = data.copy()
    aligner = kp.PoseAligner(["snout", "back_top"])
    imputer = kp.PCAImputer(n_components=1, n_iterations=2, transform_z=transform_z)
    result = imputer.impute(data, aligner, set_zero=set_zero)
    scores = imputer.transform(result)
    reconstructed = imputer.inverse_transform(scores)
    assert np.isfinite(result).all()
    assert_allclose(result[~np.isnan(data)], data[~np.isnan(data)])
    assert_allclose(data, original, equal_nan=True)
    assert scores.shape == (8, 1)
    assert reconstructed.shape == data.shape


@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_should_require_fitted_pca_when_transforming_scores(method):
    imputer = kp.PCAImputer()
    with pytest.raises(ValueError, match="not fitted") as error:
        getattr(imputer, method)(np.zeros((2, 3)))
    assert error.value is not None


@pytest.mark.parametrize("samples", [1, 2])
@pytest.mark.parametrize("kernel", [False, True])
def test_should_train_on_cluster_prior_when_kmeans_sampling_is_enabled(samples, kernel, monkeypatch):
    data = np.arange(72, dtype=float).reshape(12, 2, 3) + 1
    cluster = Mock(cluster_centers_=data.reshape(12, 6)[[0, 6]])
    cluster.fit_predict.return_value = np.repeat([0, 1], 6)
    monkeypatch.setattr(kp, "KMeans", Mock(return_value=cluster))
    kernel_args = {"kernel": "linear", "eigen_solver": "arpack"} if kernel else None
    imputer = kp.PCAImputer(n_components=1, n_iterations=2, use_kmeans_sampling=True, n_clusters=2, samples_per_cluster=samples, transform_z=True, kernel_pca_args_=kernel_args)
    result = imputer.impute(data)
    assert_allclose(result, data)
    assert cluster.fit_predict.call_count == 1
    if kernel:
        assert imputer.pca_.fit_inverse_transform


def test_should_stop_divergence_when_pca_reconstruction_error_increases(monkeypatch, capsys):
    data = np.ones((4, 1, 3))
    data[1] = np.nan
    models = []
    for error in [0, 2]:
        model = Mock()
        model.fit_transform.side_effect = lambda values: values
        model.inverse_transform.side_effect = lambda values, amount=error: values + amount
        models.append(model)
    monkeypatch.setattr(kp, "PCA", Mock(side_effect=models))
    imputer = kp.PCAImputer(n_iterations=3, mse_check=True)
    result = imputer.impute(data)
    assert_allclose(result, 1)
    assert "Divergence detected" in capsys.readouterr().out


@pytest.mark.parametrize("property_name", ["pca_components_", "pca_explained_variance_ratio_"])
def test_should_require_fitted_model_when_reading_pose_aware_pca_properties(property_name):
    imputer = kp.PoseAwareImputer(["body"])
    with pytest.raises(ValueError, match="not fitted") as error:
        getattr(imputer, property_name)
    assert error.value is not None


def test_should_restore_global_pose_and_expose_scores_when_pose_aware_imputation_completes():
    data = np.arange(24, dtype=float).reshape(8, 1, 3) + 1
    imputer = kp.PoseAwareImputer(["body"], n_components=1, n_iterations=1)
    result = imputer.impute(data)
    scores = imputer.get_pca_scores(data)
    assert_allclose(result, data)
    assert scores.shape == (8, 1)
    assert imputer.pca_components_.shape == (1, 3)
    assert imputer.pca_explained_variance_ratio_.shape == (1,)


@pytest.mark.parametrize("missing, expected", [(False, 1.), (True, .01)])
def test_should_bound_imputation_confidence_when_entire_frames_are_observed_or_missing(missing, expected):
    mask = np.full((5, 2), missing)
    result = kp.compute_imputation_confidence(mask)
    assert_allclose(result, expected)


@pytest.mark.parametrize("median_kernel", [None, 3])
def test_should_smooth_only_imputed_points_when_trajectory_has_an_isolated_spike(median_kernel):
    points = np.zeros((9, 2, 3))
    points[4, :, :] = 100
    mask = np.zeros((9, 2), bool)
    mask[4, 0] = True
    result = kp.simple_smooth_imputed(points, mask, medfilt_kernel=median_kernel, sigma=1)
    assert result[4, 0, 0] < 100
    assert_array_equal(result[~mask], points[~mask])
    assert points[4, 0, 0] == 100


def test_should_filter_only_imputed_outliers_when_hampel_window_is_even():
    values = np.array([0., 1, 0, 100, 2, 1, 0])
    points = np.repeat(values[:, None, None], 3, axis=2)
    mask = np.zeros((7, 1), bool)
    mask[3, 0] = True
    result, outliers = kp.hampel_filter(points, mask, window_size=4)
    assert_allclose(result[3], 1)
    assert outliers.sum() == 1
    assert_array_equal(result[~mask], points[~mask])


def test_should_skip_missing_or_insufficient_windows_when_hampel_filtering():
    points = np.array([[[np.nan, 1., 0]], [[np.nan, np.nan, 0]], [[np.nan, 100, 0]]])
    result, mask = kp.hampel_filter(points, np.ones((3, 1), bool), window_size=3)
    assert_allclose(result, points, equal_nan=True)
    assert not mask.any()


@pytest.mark.parametrize("method", ["linear", "slinear", "cubic"])
def test_should_interpolate_internal_gaps_and_average_anchor_confidence_when_enough_anchors_exist(method):
    data = np.repeat(np.arange(8, dtype=float)[:, None, None], 3, axis=2)
    data[3:5] = np.nan
    confidence = np.linspace(.2, .9, 8)[:, None]
    original = data.copy()
    interpolator = kp.Interpolator(length_threshold=2, method=method)
    result, log, scores = interpolator.interpolate(data, confidence)
    assert_allclose(result[:, 0, 0], np.arange(8))
    assert log == {0: [(3, 5)]}
    assert_allclose(scores[3:5], confidence[[0, 1, 2, 5, 6, 7]].mean())
    assert_allclose(data, original, equal_nan=True)
    assert not np.shares_memory(scores, confidence)


@pytest.mark.parametrize("gap, kwargs", [([0], {}), ([7], {}), ([2, 3, 4], {"length_threshold": 2}), ([3], {"low_gap_keys": [0]}), ([3], {"window": 0}), ([1, 2, 3, 4, 5, 6], {"method": "cubic"}), ([0, 1, 2, 3, 4, 5, 6], {})])
def test_should_preserve_unfillable_gaps_when_interpolation_constraints_are_not_met(gap, kwargs):
    data = np.repeat(np.arange(8, dtype=float)[:, None, None], 3, axis=2)
    data[gap] = np.nan
    interpolator = kp.Interpolator(**kwargs)
    result, log, confidence = interpolator.interpolate(data)
    assert_allclose(result, data, equal_nan=True)
    assert log == {}
    assert confidence is None


def test_should_detect_partial_coordinate_gaps_when_finding_nan_blocks():
    data = np.zeros((4, 2, 3))
    data[1:3, 0, 2] = np.nan
    data[3, 1] = np.nan
    result = kp.Interpolator._find_nan_blocks(data)
    assert result == {0: [(1, 3)], 1: [(3, 4)]}


@pytest.mark.parametrize("mode, near_weight", [("linear", .02), ("quadratic", .002), ("sigmoid", .2 / (1 + np.exp(2.4)))])
def test_should_downweight_low_confidence_near_edges_when_computing_edge_weights(mode, near_weight):
    points = np.array([[0., 0], [100, 100], [0, 0]])
    confidence = np.array([.2, .2, .8])
    result = kp.edge_weight_map(points, confidence, mode=mode)
    assert result[0] == pytest.approx(near_weight)
    assert result[2] == .8
    assert_array_equal(confidence, [.2, .2, .8])


def test_should_reject_unknown_mode_when_computing_edge_weights():
    points = np.zeros((1, 2))
    with pytest.raises(ValueError, match="Unknown edge weight mode") as error:
        kp.edge_weight_map(points, np.ones(1), mode="bad")
    assert error.value is not None


def test_should_forward_gap_limit_when_smoothing_all_keypoints(monkeypatch):
    regularizer = Mock()
    regularizer.optimize_trajectory.side_effect = lambda points, scores, *args, **kwargs: (points + 1, scores + .1)
    constructor = Mock(return_value=regularizer)
    monkeypatch.setattr(kp, "TemporalRegularization", constructor)
    points = np.zeros((3, 2, 3))
    scores = np.full((3, 2), .5)
    result, confidence = kp.smooth_all_keypoints(points, scores, max_gap_fill=2, fps=30)
    assert_allclose(result, 1)
    assert_allclose(confidence, .6)
    assert regularizer.optimize_trajectory.call_count == 2
    assert all(item.kwargs == {"max_gap_fill": 2} for item in regularizer.optimize_trajectory.call_args_list)
    constructor.assert_called_with(fps=30)


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=TypeError, reason='smooth_all_keypoints passes max_gap_fill to a sliding optimizer that does not accept it')
def test_should_support_sliding_windows_when_smoothing_long_sequences():
    points = np.ones((5001, 1, 3))
    scores = np.ones((5001, 1))
    result, confidence = kp.smooth_all_keypoints(points, scores, use_sliding_window=True, sliding_window_threshold=5000)
    assert result.shape == points.shape
    assert confidence.shape == scores.shape


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='smooth_all_keypoints ignores sliding_window_threshold and uses a hardcoded 5000 frames')
def test_should_use_custom_threshold_when_selecting_sliding_smoother(monkeypatch):
    points = np.ones((4, 1, 3))
    scores = np.ones((4, 1))
    optimizer = Mock()
    optimizer.optimize_trajectory.return_value = (points[:, 0], scores[:, 0])
    sliding = Mock(return_value=optimizer)
    monkeypatch.setattr(kp, "SlidingWindowTemporalRegularization", sliding)
    kp.smooth_all_keypoints(points, scores, use_sliding_window=True, sliding_window_threshold=3)
    sliding.assert_called_once_with()
