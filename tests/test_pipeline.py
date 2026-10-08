import copy
from types import SimpleNamespace
from unittest.mock import Mock, mock_open

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.pcl import pipeline


@pytest.mark.parametrize("filename", [None, "missing.toml"])
def test_should_report_missing_configuration_when_path_is_null_or_absent(monkeypatch, filename):
    monkeypatch.setattr(pipeline.os.path, "exists", lambda name: False)
    with pytest.raises(FileNotFoundError, match="Config file not found") as error:
        pipeline.load_config(filename)
    assert str(filename) in str(error.value)


@pytest.mark.parametrize("config", [{"index_conf_map": {"0": .1, "2": .5}}, {"fps": 100}])
def test_should_convert_confidence_map_indices_when_loading_configuration(monkeypatch, config):
    monkeypatch.setattr(pipeline.os.path, "exists", lambda name: True)
    monkeypatch.setattr(pipeline, "open", mock_open(), raising=False)
    monkeypatch.setattr(pipeline.toml, "load", Mock(return_value=copy.deepcopy(config)))
    result = pipeline.load_config("config.toml")
    assert result == ({"index_conf_map": {0: .1, 2: .5}} if "index_conf_map" in config else config)


def test_should_ignore_missing_dimensions_when_computing_distance():
    points = np.array([[3., 4, np.nan], [np.nan] * 3, [0, 0, 0]])
    result = pipeline.nan_safe_linalg_norm(points, np.zeros(3))
    assert_allclose(result, [5, np.nan, 0], equal_nan=True)


def test_should_reject_incompatible_dimensions_when_computing_distance():
    points = np.ones((2, 3))
    with pytest.raises(ValueError, match="Trailing dimensions") as error:
        pipeline.nan_safe_linalg_norm(points, np.ones(2))
    assert error.value is not None


def test_should_clip_coordinates_and_preserve_invalid_points_when_sampling_background():
    background = np.arange(12).reshape(3, 4)
    points = np.array([[[0., 1], [-4, 20], [np.nan, 1]]])
    with np.errstate(invalid="ignore"):
        result = pipeline.get_bground_vals(points, "cam", {"cam": background}, width=3, height=4)
    assert_allclose(result, [[1, 3, np.nan]], equal_nan=True)


@pytest.fixture
def pipeline_environment(monkeypatch, memory_h5):
    names = ["body", "snout", "tail_tip", "a", "b", "c", "d"]
    cameras = ["a", "b", "c"]
    config = {
        "reference_camera": "a", "noisy_keypoints": [], "incl_kpoints_fit_transform": names,
        "plt_kpoints": names, "skeleton": [["body", "snout"]], "fps": 50,
        "index_conf_map": {0: .05}, "renderer_kwargs": {}, "proc_order": ["temporal"],
        "post_processing": {"incl_kpoints_post_processing": names, "temporal_regularization": {}},
    }
    arrays = {}
    for camera in cameras:
        data = np.zeros((5, len(names), 4))
        data[..., 0] = np.arange(len(names)) + 2
        data[..., 1] = 3
        data[..., 2] = 90
        data[..., 3] = .9
        arrays[camera] = data
    metadata = {"camera_metadata": {camera: {"Width": 20, "Height": 20} for camera in cameras}}
    transforms = {str((camera, "a")): [np.eye(3).tolist(), [0, 0, 0]] for camera in cameras}

    def load_toml(filename):
        if str(filename).endswith("metadata.toml"):
            return copy.deepcopy(metadata)
        if str(filename) == "transforms.toml":
            return copy.deepcopy(transforms)
        return {"node_names": names}

    monkeypatch.setattr(pipeline, "load_config", Mock(side_effect=lambda name: copy.deepcopy(config)))
    monkeypatch.setattr(pipeline.toml, "load", Mock(side_effect=load_toml))
    dump = Mock()
    monkeypatch.setattr(pipeline.toml, "dump", dump)
    monkeypatch.setattr(pipeline, "open", mock_open(), raising=False)
    monkeypatch.setattr(pipeline.os, "makedirs", Mock())
    monkeypatch.setattr(pipeline.joblib, "load", Mock(side_effect=lambda filename: arrays[str(filename).split("/")[-1].split(".")[0]].copy()))
    monkeypatch.setattr(pipeline.tifffile, "imread", Mock(return_value=np.full((20, 20), 400, np.float32)))
    roi = np.zeros((20, 20), np.uint8)
    roi[1, 1] = roi[2, 2] = 1
    monkeypatch.setattr(pipeline.depth.plane, "get_floor", Mock(return_value=roi))
    monkeypatch.setattr(pipeline.cv2, "undistort", Mock(side_effect=lambda data, *args: data))
    monkeypatch.setattr(pipeline.cv2, "erode", Mock(side_effect=lambda data, *args: data.astype(bool)))
    result = {"B_to_A": {"R": np.eye(3), "t": np.zeros(3)}, "C_to_A": {"R": np.eye(3), "t": np.zeros(3)}}
    estimate = Mock(return_value=result)
    bundle = Mock(return_value=result)
    monkeypatch.setattr(pipeline.pcl.registration, "estimate_transform", estimate)
    monkeypatch.setattr(pipeline.pcl.registration, "bundle_adjust_rigid_fixed_structure", bundle)
    post = Mock(side_effect=lambda data, confidence, *args, **kwargs: (data + 1, np.nanmax(confidence, axis=-1)))
    monkeypatch.setattr(pipeline, "post_processing", post)
    monkeypatch.setattr(pipeline.pd, "read_csv", Mock(return_value=pd.DataFrame({"device_timestamp_ref": np.arange(5) * .02})))
    renderers = {}
    for renderer, function in [("matplotlib", "visualize_xyz_trajectories_to_mp4"), ("vedo", "visualize_xyz_trajectories_vedo")]:
        renderers[renderer] = Mock()
        monkeypatch.setattr(pipeline.pcl.viz, function, renderers[renderer])
    return SimpleNamespace(config=config, cameras=cameras, metadata=metadata, arrays=arrays, post=post, estimate=estimate, bundle=bundle, dump=dump, renderers=renderers, files=memory_h5.files)


@pytest.mark.parametrize("camera_count, cached, bundle, renderer", [(1, False, False, None), (3, False, False, "matplotlib"), (3, False, True, "vedo"), (2, True, False, None)])
def test_should_register_merge_postprocess_and_save_when_pipeline_inputs_are_valid(pipeline_environment, camera_count, cached, bundle, renderer):
    env = pipeline_environment
    cameras = env.cameras[:camera_count]
    matrices = {camera: np.eye(3) for camera in cameras}
    distortions = {camera: np.zeros(5) for camera in cameras}
    result = pipeline.registration_pipeline("config.toml", "/session", intrinsics_matrix=matrices, distortion_coefficients=distortions, bground_erode_px=1, bundle_adjust=bundle, transforms_path="transforms.toml" if cached else None, render=renderer is not None, mp4_renderer=renderer, mp4_burn_in=0, mp4_max_render_frames=3)
    assert result is None
    saved = env.files["/session/_kpoints_v0_3d/merged_keypoints.h5"]
    assert saved.closed
    assert saved["merged_keypoints_raw"].shape == (5, 7, 3)
    assert saved["merged_keypoints_confidence"].shape == (5, 7, camera_count)
    assert_allclose(saved["merged_keypoints_raw"][..., 2], 10)
    assert_allclose(saved["merged_keypoints_smooth"], saved["merged_keypoints_raw"] + 1)
    assert_allclose(saved["device_timestamp_ref"], np.arange(5) * .02)
    assert env.post.call_args.args[3]["temporal_regularization"]["fps"] == 50
    assert env.estimate.call_count == int(camera_count > 1 and not cached and not bundle)
    assert env.bundle.call_count == int(bundle)
    assert env.dump.call_args.args[0]["reference_camera"] == "a"
    if renderer:
        assert env.renderers[renderer].call_args.args[0].shape == (3, 7, 3)


@pytest.mark.parametrize("merge_method", ["mixed", "weighted", "max"])
def test_should_return_shared_points_when_identical_cameras_are_merged_with_any_method(pipeline_environment, merge_method):
    # Discriminating merge checks against real data live in test_gt_merge.py.
    env = pipeline_environment
    matrices = {camera: np.eye(3) for camera in env.cameras}
    distortions = {camera: np.zeros(5) for camera in env.cameras}
    pipeline.registration_pipeline("config.toml", "/session", intrinsics_matrix=matrices, distortion_coefficients=distortions, bground_erode_px=1, transforms_path="transforms.toml", merge_method=merge_method)
    raw = env.files["/session/_kpoints_v0_3d/merged_keypoints.h5"]["merged_keypoints_raw"]
    expected = np.stack(np.broadcast_arrays((np.arange(7) + 2) * 10., 30., 10.), axis=-1)
    assert_allclose(raw, np.broadcast_to(expected, (5, 7, 3)))


def test_should_reject_unknown_merge_method_before_reading_inputs(pipeline_environment):
    env = pipeline_environment
    with pytest.raises(ValueError, match="Unknown merge_method 'median'"):
        pipeline.registration_pipeline("config.toml", "/session", intrinsics_matrix={"a": np.eye(3)}, distortion_coefficients={"a": np.zeros(5)}, merge_method="median")
    pipeline.load_config.assert_not_called()
    assert env.files == {}


def test_should_skip_postprocessing_and_use_alternate_output_when_order_is_empty(pipeline_environment):
    env = pipeline_environment
    env.config["proc_order"] = []
    pipeline.registration_pipeline("config.toml", "/session", intrinsics_matrix={"a": np.eye(3)}, distortion_coefficients={"a": np.zeros(5)}, bground_erode_px=1, alt_save_dir="/output")
    saved = env.files["/output/merged_keypoints.h5"]
    assert_array_equal(saved["merged_keypoints_smooth"], saved["merged_keypoints_raw"])
    env.post.assert_not_called()


def test_should_require_calibration_when_pipeline_starts(pipeline_environment):
    env = pipeline_environment
    with pytest.raises(RuntimeError, match="Need intrinsics") as error:
        pipeline.registration_pipeline("config.toml", "/session")
    assert error.value is not None
    assert env.files == {}


def test_should_warn_and_return_when_session_metadata_is_missing(pipeline_environment, monkeypatch):
    monkeypatch.setattr(pipeline.toml, "load", Mock(side_effect=FileNotFoundError("missing metadata")))
    with pytest.warns(UserWarning, match="Did not find metadata"):
        result = pipeline.registration_pipeline("config.toml", "/session", intrinsics_matrix={"a": np.eye(3)}, distortion_coefficients={"a": np.zeros(5)})
    assert result is None
    assert pipeline_environment.files == {}


def test_should_fall_back_to_present_camera_when_reference_camera_is_unknown(pipeline_environment):
    env = pipeline_environment
    env.config["reference_camera"] = "missing"
    with pytest.warns(UserWarning, match="not in"):
        pipeline.registration_pipeline("config.toml", "/session", intrinsics_matrix={"a": np.eye(3)}, distortion_coefficients={"a": np.zeros(5)}, bground_erode_px=1)
    assert env.dump.call_args.args[0]["reference_camera"] == "a"
