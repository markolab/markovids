from types import SimpleNamespace
from unittest.mock import Mock, mock_open

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids import util
from markovids.vid import io, util as video_util


@pytest.fixture
def video_environment(monkeypatch):
    cameras = ["a", "b"]
    table = {"device_timestamp_ref": np.arange(5) * .02}
    for camera in cameras:
        for field in ["frame_id", "frame_index", "device_timestamp", "system_timestamp"]:
            table[camera, field] = np.arange(5)
    timestamps = pd.DataFrame(table)
    timestamp_reader = Mock(return_value=({}, {}, timestamps, timestamps.copy()))
    monkeypatch.setattr(io, "read_timestamps_multicam", timestamp_reader)

    def read_frames(paths, frames, *args, **kwargs):
        factor = kwargs.get("downsample", 1)
        return {camera: np.array([np.arange(16).reshape(4, 4)[::factor, ::factor] + 10 + index for index in frames[camera]], dtype=np.uint16) for camera in paths.values()}

    frame_reader = Mock(side_effect=read_frames)
    monkeypatch.setattr(io, "read_frames_multicam", frame_reader)
    monkeypatch.setattr(io, "get_bground", Mock(return_value=np.zeros((4, 4))))
    monkeypatch.setattr(io, "pseudocolor_frames", Mock(side_effect=lambda data, **kwargs: np.repeat(np.clip(data, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)))
    writers = {}

    def make_writer(filename, **kwargs):
        writers[filename] = Mock()
        return writers[filename]

    avi_factory, mp4_factory = Mock(side_effect=make_writer), Mock(side_effect=make_writer)
    monkeypatch.setattr(io, "AviWriter", avi_factory)
    monkeypatch.setattr(io, "MP4WriterPreview", mp4_factory)
    reader = Mock(frame_size=(4, 4), pixel_format="gray16le", dtype=np.dtype("uint16"))
    monkeypatch.setattr(io, "AviReader", Mock(return_value=reader))
    monkeypatch.setattr(util, "compute_bground", Mock(return_value=np.full((4, 4), 100)))
    metadata = {"cameras": {camera: {} for camera in cameras}}
    monkeypatch.setattr(__import__("toml"), "load", Mock(side_effect=lambda filename: metadata))
    dump = Mock()
    monkeypatch.setattr(__import__("toml"), "dump", dump)
    monkeypatch.setattr(util, "open", mock_open(), raising=False)
    monkeypatch.setattr(util.os, "makedirs", Mock())
    monkeypatch.setattr(util.os.path, "exists", lambda filename: False)
    paths = {f"/session/{camera}.avi": camera for camera in cameras}
    load_config = {camera: {"frame_size": (4, 4), "dtype": np.dtype("uint16")} for camera in cameras}
    return SimpleNamespace(cameras=cameras, timestamps=timestamps, timestamp_reader=timestamp_reader, frame_reader=frame_reader, writers=writers, avi_factory=avi_factory, mp4_factory=mp4_factory, paths=paths, config=load_config, metadata=metadata, reader=reader, dump=dump)


@pytest.mark.parametrize("nbatches, batch_lengths", [(None, [2, 2, 1]), (1, [2])])
def test_should_split_exposures_and_close_all_writers_when_processing_batches(video_environment, nbatches, batch_lengths):
    env = video_environment
    result = util.alternating_excitation_vid_split(env.paths, {}, env.config, batch_size=2, nbatches=nbatches)
    assert result is None
    assert len(env.writers) == 4
    for writer in env.writers.values():
        assert [len(item.args[0]) for item in writer.write_frames.call_args_list] == batch_lengths
        writer.close.assert_called_once()
    assert all(item.kwargs["pixel_format"] == "gray16le" for item in env.avi_factory.call_args_list)


@pytest.mark.parametrize("dtype, exception", [(np.dtype("uint8"), None), (np.dtype("float32"), RuntimeError)])
def test_should_choose_eight_bit_encoder_or_reject_dtype_when_splitting_video(video_environment, dtype, exception):
    env = video_environment
    for config in env.config.values():
        config["dtype"] = dtype
    if exception:
        with pytest.raises(exception, match="Can't map") as error:
            util.alternating_excitation_vid_split(env.paths, {}, env.config, batch_size=2, nbatches=1)
    else:
        util.alternating_excitation_vid_split(env.paths, {}, env.config, batch_size=2, nbatches=1)
    if exception:
        assert env.writers == {}
        assert error.value is not None
    else:
        assert all(item.kwargs["pixel_format"] == "gray" for item in env.avi_factory.call_args_list)


@pytest.mark.parametrize("nbatches, filtering", [(0, False), (1, True)])
def test_should_remove_overlap_and_generate_all_previews_when_processing_excitations(video_environment, monkeypatch, nbatches, filtering):
    env = video_environment
    bandpass = Mock(side_effect=lambda frame, *args: frame)
    temporal = Mock(side_effect=lambda frames, *args: frames)
    monkeypatch.setattr(video_util, "bp_filter", bandpass)
    monkeypatch.setattr(video_util, "sos_filter", temporal)
    result = util.alternating_excitation_vid_preview(env.paths, {}, env.config, batch_size=2, overlap=1, nbatches=nbatches, downsample=1, spatial_bp=(1, 2) if filtering else None, temporal_tau=.1 if filtering else None)
    assert result is None
    expected_lengths = [2] if nbatches else [2, 2, 1]
    assert len(env.writers) == 3
    for writer in env.writers.values():
        assert [len(item.args[0]) for item in writer.write_frames.call_args_list] == expected_lengths
        assert [list(item.kwargs["frames_idx"]) for item in writer.write_frames.call_args_list] == ([[0, 1]] if nbatches else [[0, 1], [2, 3], [4]])
        writer.close.assert_called_once()
    assert bandpass.call_count == (4 if filtering else 0)
    assert temporal.call_count == (2 if filtering else 0)


@pytest.mark.parametrize("single, undistort, inpaint", [(False, True, True), (True, False, False)])
def test_should_subtract_background_save_depth_and_close_writers_when_synchronizing(video_environment, monkeypatch, single, undistort, inpaint):
    env = video_environment
    cameras = ["a"] if single else env.cameras
    env.metadata["cameras"] = {camera: {} for camera in cameras}
    env.timestamp_reader.return_value = env.timestamps if single else ({}, env.timestamps)
    fill = Mock(side_effect=lambda frame: frame + 1)
    distortion = Mock(side_effect=lambda frame, *args: frame)
    monkeypatch.setattr(video_util, "fill_holes", fill)
    monkeypatch.setattr(__import__("cv2"), "undistort", distortion)
    util.sync_depth_videos("/session", batch_size=2, vid_camera_order=cameras, undistort=undistort, intrinsics_matrix={camera: np.eye(3) for camera in cameras}, distortion_coeffs={camera: np.zeros(5) for camera in cameras}, preview_inpaint=inpaint)
    assert len(env.writers) == len(cameras) + 1
    assert fill.call_count == (5 * len(cameras) if inpaint else 0)
    assert distortion.call_count == (len(cameras) if undistort else 0)
    for camera in cameras:
        writer = env.writers[f"/session/_proc/{camera}.avi"]
        assert [len(item.args[0]) for item in writer.write_frames.call_args_list] == [2, 2, 1]
        # Writers observe the same buffer later used for preview inpainting.
        assert_allclose(writer.write_frames.call_args_list[0].args[0][0, 0, 0], 22 + int(inpaint))
        writer.close.assert_called_once()
    env.writers["/session/_proc/depth_preview.mp4"].close.assert_called_once()
    env.reader.close.assert_called()
    assert env.dump.call_count == 1


def test_should_warn_and_use_available_camera_order_when_requested_order_is_absent(video_environment):
    env = video_environment
    env.timestamp_reader.return_value = ({}, env.timestamps)
    with pytest.warns(UserWarning, match="not found"):
        result = util.sync_depth_videos("/session", batch_size=5, vid_camera_order=["missing"])
    assert result is None
    assert len(env.writers) == 3


def test_should_reject_empty_camera_configuration_when_synchronizing(video_environment):
    env = video_environment
    env.metadata["cameras"] = {}
    with pytest.warns((UserWarning, RuntimeWarning)), pytest.raises(RuntimeError, match="Number of cameras") as error:
        util.sync_depth_videos("/session")
    assert error.value is not None
    assert env.writers == {}
