from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.pcl import post_processing as pp


@pytest.mark.parametrize("names, expected", [(["snout"], (5, 2)), (["tail_tip"], (5, 2)), (["body"], (7, 3))])
def test_should_use_adaptive_filter_parameters_when_smoothing_named_keypoints(monkeypatch, names, expected):
    data = np.arange(24, dtype=float).reshape(8, 1, 3)
    original = data.copy()
    filter_mock = Mock(side_effect=lambda values, *args, **kwargs: values + 10)
    monkeypatch.setattr(pp, "savgol_filter", filter_mock)
    result = pp.apply_final_smoothing(data, names)
    assert_allclose(result, original + 10)
    assert_array_equal(data, original)
    assert filter_mock.call_count == 3
    assert all(item.args[1:] == expected and item.kwargs == {"mode": "nearest"} for item in filter_mock.call_args_list)


def test_should_preserve_missing_coordinates_when_trajectory_is_entirely_nan(monkeypatch):
    data = np.full((8, 1, 3), np.nan)
    filter_mock = Mock()
    monkeypatch.setattr(pp, "savgol_filter", filter_mock)
    result = pp.apply_final_smoothing(data, ["body"])
    assert np.isnan(result).all()
    assert result is not data
    filter_mock.assert_not_called()


def test_should_reduce_noise_when_smoothing_a_constant_trajectory():
    data = np.full((15, 1, 3), 10.0)
    data[7, 0, 0] = 30
    result = pp.apply_final_smoothing(data, ["body"])
    assert 10 < result[7, 0, 0] < 30
    assert_allclose(result[:, 0, 1:], 10)


@pytest.mark.parametrize("name", ["body", "snout", "tail_tip"])
@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='final Savitzky-Golay smoothing spreads NaNs into observed frames')
def test_should_smooth_observed_runs_without_spreading_or_crossing_nan_gaps(name):
    data = np.full((15, 1, 3), 10.0)
    data[7, 0, 0] = np.nan
    data[8:, 0, 0] = 100.0
    # Include boundary gaps and runs shorter than the filter window.
    data[:, 0, 1] = np.nan
    data[1, 0, 1] = 20.0
    data[10:12, 0, 1] = 30.0
    original = data.copy()

    result = pp.apply_final_smoothing(data, [name])

    assert_array_equal(np.isnan(result), np.isnan(original))
    assert_allclose(result, original, equal_nan=True)
    assert_array_equal(data, original)
    assert not np.shares_memory(result, data)


def test_should_raise_filter_error_when_polynomial_order_is_invalid():
    data = np.ones((8, 1, 3))
    with pytest.raises(ValueError, match="polyorder") as error:
        pp.apply_final_smoothing(data, ["snout"], window_length=3, polyorder=5)
    assert error.value is not None


def test_should_reorder_keypoints_and_take_best_camera_confidence_when_extracting(processor):
    data = np.arange(18).reshape(2, 3, 3)
    confidence = np.array([[[.1, .3], [.4, np.nan], [.2, .7]]] * 2)
    points, scores = processor._extract_keypoints(data, confidence, ["tail_tip", "body"])
    assert_array_equal(points, data[:, [2, 0]])
    assert_allclose(scores, [[.7, .3]] * 2)
    assert not np.shares_memory(points, data)


def test_should_reject_unknown_keypoints_when_extracting(processor):
    data = np.zeros((1, 3, 3))
    with pytest.raises(ValueError, match="unknown") as error:
        processor._extract_keypoints(data, np.ones((1, 3, 2)), ["unknown"])
    assert error.value is not None


@pytest.mark.parametrize("order", [None, ["temporal", "bone", "bone", "fill", "final_smooth"], []])
def test_should_run_stages_in_requested_order_when_processing(processor, post_config, monkeypatch, order):
    data = np.zeros((2, 3, 3))
    scores = np.ones((2, 3, 2))
    events = []
    constraints = object()
    creator = Mock(return_value=constraints)
    monkeypatch.setattr(pp.kpoints, "create_bone_constraints_from_data", creator)

    def pair(stage):
        def run(points, confidence, *args):
            events.append(stage)
            return points + 1, confidence
        return run

    def bone(points, confidence, names, config, existing):
        events.append("bone")
        assert existing is constraints
        return points + 1, confidence, existing

    monkeypatch.setattr(processor, "_apply_temporal_smoothing", pair("temporal"))
    monkeypatch.setattr(processor, "_apply_bone_constraints", bone)
    monkeypatch.setattr(processor, "_apply_interpolation", pair("interpolate"))
    monkeypatch.setattr(processor, "_apply_final_smoothing", lambda points, *args: events.append("final_smooth") or points + 1)
    points, confidence = processor.process(data, scores, ["body"], post_config, order)
    expected = list(processor.DEFAULT_PROC_ORDER if order is None else order)
    assert events == ["interpolate" if stage == "fill" else stage for stage in expected]
    assert_allclose(points, len(expected))
    assert_allclose(confidence, 1)
    assert creator.call_count == (1 if "bone" in expected else 0)
    assert_array_equal(data, 0)


@pytest.mark.parametrize("stage", ["pca_impute", "autoencoder_impute"])
def test_should_dispatch_optional_imputation_when_stage_is_requested(processor, post_config, monkeypatch, stage):
    expected = (np.zeros((1, 1, 3)), np.ones((1, 1)))
    method = "_apply_pca_imputation" if stage == "pca_impute" else "_apply_autoencoder_imputation"
    imputer = Mock(return_value=expected)
    monkeypatch.setattr(processor, method, imputer)
    result = processor.process(np.ones((1, 3, 3)), np.ones((1, 3, 2)), ["body"], post_config, [stage])
    assert result[0] is expected[0]
    assert result[1] is expected[1]
    assert imputer.call_count == 1


@pytest.mark.parametrize("order", [["bad"], ["temporal", "final_bone"]])
def test_should_validate_all_stages_before_processing_when_order_is_invalid(processor, post_config, monkeypatch, order):
    smooth = Mock()
    monkeypatch.setattr(processor, "_apply_temporal_smoothing", smooth)
    with pytest.raises(ValueError, match="Unknown stage") as error:
        processor.process(np.zeros((1, 3, 3)), np.ones((1, 3, 2)), ["body"], post_config, order)
    assert order[-1] in str(error.value)
    smooth.assert_not_called()


def test_should_propagate_stage_failure_when_temporal_smoothing_raises(processor, post_config, monkeypatch):
    failure = RuntimeError("solver failed")
    monkeypatch.setattr(pp.kpoints, "smooth_all_keypoints", Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as error:
        processor.process(np.zeros((1, 3, 3)), np.ones((1, 3, 2)), ["body"], post_config, ["temporal"])
    assert error.value is failure


@pytest.mark.parametrize("existing", [None, "cached"])
def test_should_average_confidence_when_enforcing_bone_constraints(processor, post_config, monkeypatch, existing):
    points, scores = np.ones((2, 1, 3)), np.full((2, 1), .8)
    creator = Mock(return_value="created")
    optimizer = Mock()
    optimizer.process_sequence.return_value = (points + 2, np.full_like(scores, .4))
    constructor = Mock(return_value=optimizer)
    monkeypatch.setattr(pp.kpoints, "create_bone_constraints_from_data", creator)
    monkeypatch.setattr(pp.kpoints, "BoneConstraintOptimizer", constructor)
    result, confidence, constraints = processor._apply_bone_constraints(points, scores, ["body"], post_config, existing)
    assert_allclose(result, 3)
    assert_allclose(confidence, .6)
    assert constraints == (existing or "created")
    assert creator.call_count == (existing is None)
    constructor.assert_called_once_with(constraints, iterations=2)


@pytest.mark.parametrize("kind", ["pca", "autoencoder"])
def test_should_align_impute_filter_and_restore_coordinates_when_imputing(processor, post_config, monkeypatch, kind):
    points = np.array([[[np.nan, 1, 2]], [[3, 4, 5]]])
    confidence = np.array([[.2], [.8]])
    aligner = Mock()
    centroids, angles = np.zeros((2, 2)), np.zeros(2)
    aligner.compute_alignment.return_value = (centroids, angles)
    aligned = np.zeros_like(points)
    aligner.transform.return_value = aligned
    aligner.inverse_transform.return_value = np.ones_like(points)
    aligner_constructor = Mock(return_value=aligner)
    imputer = Mock()
    imputer.impute.return_value = aligned + 1
    imputer_constructor = Mock(return_value=imputer)
    filtered, final = np.full_like(points, 2), np.full_like(points, 3)
    imputation_conf = Mock(return_value=np.array([[.6], [1.]]))
    hampel = Mock(return_value=(filtered, np.zeros((2, 1), dtype=bool)))
    smooth = Mock(return_value=final)
    monkeypatch.setattr(pp.kpoints, "PoseAligner", aligner_constructor)
    monkeypatch.setattr(pp.kpoints, "PCAImputer" if kind == "pca" else "AutoencoderImputer", imputer_constructor)
    monkeypatch.setattr(pp.kpoints, "compute_imputation_confidence", imputation_conf)
    monkeypatch.setattr(pp.kpoints, "hampel_filter", hampel)
    monkeypatch.setattr(pp.kpoints, "simple_smooth_imputed", smooth)
    post_config.autoencoder = {"checkpoint_path": "fake.pt", "device": "cpu", "batch_size": 2}
    method = processor._apply_pca_imputation if kind == "pca" else processor._apply_autoencoder_imputation
    result, scores = method(points, confidence, ["body"], post_config)
    assert result is final
    assert_allclose(scores, [[.4], [.9]])
    aligner_constructor.assert_called_once_with(keypoint_names=["body"], **post_config.align)
    assert_array_equal(imputation_conf.call_args.args[0], [[True], [False]])
    assert smooth.call_args.args[0] is filtered
    assert imputer.impute.call_args.args[0] is aligned
    if kind == "autoencoder":
        imputer_constructor.assert_called_once_with(checkpoint_path="fake.pt", device="cpu", batch_size=2)
        assert imputer.impute.call_args.args[2] is confidence
    else:
        imputer_constructor.assert_called_once_with(n_components=1)


@pytest.mark.parametrize("settings", [None, {}, {"checkpoint_path": None}])
def test_should_require_checkpoint_when_autoencoder_imputation_is_selected(processor, post_config, settings):
    post_config.autoencoder = settings
    with pytest.raises(ValueError, match="checkpoint_path") as error:
        processor._apply_autoencoder_imputation(np.zeros((1, 1, 3)), np.ones((1, 1)), ["body"], post_config)
    assert error.value is not None


@pytest.mark.parametrize("new_confidence", [None, np.array([[.25]])])
def test_should_preserve_or_replace_confidence_when_interpolating(processor, post_config, monkeypatch, new_confidence):
    points, confidence = np.zeros((1, 1, 3)), np.ones((1, 1))
    aligner = Mock()
    aligner.compute_alignment.return_value = ("centers", "angles")
    aligner.transform.return_value = points
    aligner.inverse_transform.return_value = points + 1
    interpolator = Mock()
    interpolator.interpolate.return_value = (points + 2, object(), new_confidence)
    constructor = Mock(return_value=interpolator)
    monkeypatch.setattr(pp.kpoints, "PoseAligner", Mock(return_value=aligner))
    monkeypatch.setattr(pp.kpoints, "Interpolator", constructor)
    post_config.interpolation = {"length_threshold": 4}
    result, scores = processor._apply_interpolation(points, confidence, ["body"], post_config)
    assert_allclose(result, 1)
    assert scores is (confidence if new_confidence is None else new_confidence)
    constructor.assert_called_once_with(length_threshold=4)
    assert_array_equal(aligner.inverse_transform.call_args.args[0], points + 2)


def test_should_reuse_optimizer_when_applying_final_bone_constraints(processor, post_config, monkeypatch):
    optimizer = Mock()
    expected = (object(), object())
    optimizer.process_sequence.return_value = expected
    constructor = Mock(return_value=optimizer)
    monkeypatch.setattr(pp.kpoints, "BoneConstraintOptimizer", constructor)
    result = processor._apply_final_bone_constraints("points", "confidence", post_config, "constraints")
    assert result is expected
    constructor.assert_called_once_with("constraints", iterations=2)
    optimizer.process_sequence.assert_called_once_with("points", "confidence")


def test_should_forward_smoothing_settings_when_applying_final_smoothing(processor, monkeypatch):
    smooth = Mock(return_value="smoothed")
    monkeypatch.setattr(pp, "apply_final_smoothing", smooth)
    result = processor._apply_final_smoothing("points", ["body"], {"fps": 50})
    assert result == "smoothed"
    smooth.assert_called_once_with("points", ["body"], fps=50)


def test_should_apply_custom_weights_and_preserve_nan_when_combining_confidence(processor):
    first, second = np.array([.2, np.nan]), np.array([.8, .4])
    result = processor._combine_confidence_scores(first, second, w1=.25, w2=.75)
    assert_allclose(result, [.65, np.nan], equal_nan=True)
    assert_allclose(first, [.2, np.nan], equal_nan=True)


def test_should_construct_configuration_and_forward_order_when_using_legacy_wrapper(post_config, monkeypatch, capsys):
    processor = Mock()
    processor.process.return_value = ("points", "confidence")
    constructor = Mock(return_value=processor)
    monkeypatch.setattr(pp, "KeypointPostProcessor", constructor)
    result = pp.post_processing("data", "scores", {"node_names": ["body"]}, asdict(post_config), ["body"], [], proc_order=[])
    assert result == ("points", "confidence")
    constructor.assert_called_once_with({"node_names": ["body"]}, [])
    assert asdict(processor.process.call_args.args[3]) == asdict(post_config)
    assert processor.process.call_args.kwargs == {"proc_order": []}
    assert "Applying post processing" in capsys.readouterr().out
