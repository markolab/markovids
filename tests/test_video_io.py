import io as streams
import shlex
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, mock_open

import cv2
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.vid import io


@pytest.fixture
def process_factory(monkeypatch):
    process = Mock()
    process.communicate.return_value = (b"", b"")
    factory = Mock(return_value=process)
    monkeypatch.setattr(io.subprocess, "Popen", factory)
    return factory, process


@pytest.mark.parametrize("writer_class, filename", [(io.AviWriter, "video.mp4"), (io.MP4WriterPreview, "video.avi")])
def test_should_reject_wrong_container_when_constructing_video_writer(writer_class, filename):
    constructor = writer_class
    with pytest.raises(RuntimeError, match="container") as error:
        constructor(filename)
    assert error.value is not None


@pytest.mark.parametrize("size, expected, padding", [((4, 6), (4, 6), (0, 0)), ((3, 5), (4, 6), (1, 1))])
def test_should_pad_odd_dimensions_when_constructing_mp4_writer(size, expected, padding):
    filename = "preview.mp4"
    writer = io.MP4WriterPreview(filename, frame_size=size)
    assert writer.frame_size == expected
    assert writer.pad == padding
    assert writer.pipe is None


@pytest.mark.parametrize("writer_class, filename", [(io.AviWriter, "video.avi"), (io.MP4WriterPreview, "video.mp4")])
def test_should_start_encoder_and_close_pipe_when_writing_video(process_factory, writer_class, filename):
    factory, process = process_factory
    writer = writer_class(filename, frame_size=(4, 4))
    frames = np.ones((2, 4, 4, 3), dtype=np.uint8) if writer_class is io.MP4WriterPreview else np.ones((2, 4, 4), dtype=np.uint16)
    writer.write_frames(frames, progress_bar=False)
    writer.close()
    assert factory.call_count == 1
    assert process.stdin.write.call_count == 2
    process.stdin.close.assert_called_once()
    process.wait.assert_called_once()
    command = factory.call_args.args[0]
    assert (shlex.split(command) if isinstance(command, str) else command)[-1] == filename


def test_should_prepend_encoder_arguments_when_avi_writer_has_shell_prefix(process_factory):
    factory, _ = process_factory
    writer = io.AviWriter("video.avi", prepend_args="env TASK_TEST=1")
    writer.open()
    assert factory.call_args.args[0].startswith("env TASK_TEST=1 ; ffmpeg")
    assert factory.call_args.kwargs["shell"]


def test_should_mark_selected_frames_and_pad_when_writing_rgb_preview(process_factory, monkeypatch):
    _, process = process_factory
    writer = io.MP4WriterPreview("preview.mp4", frame_size=(3, 3))
    text, marker = Mock(), Mock()
    monkeypatch.setattr(io, "inscribe_text", text)
    monkeypatch.setattr(io, "mark_frame", marker)
    frames = np.ones((2, 3, 3, 3), dtype=np.uint8)
    original = frames.copy()
    writer.write_frames(frames, frames_idx=[4, 5], mark_frames=[5], progress_bar=False)
    assert process.stdin.write.call_count == 2
    assert text.call_count == 2
    assert marker.call_count == 1
    assert marker.call_args.args[0].shape == (4, 4, 3)
    assert_array_equal(frames, original)


def test_should_colorize_intensity_stack_when_writing_depth_preview(process_factory):
    _, process = process_factory
    writer = io.MP4WriterPreview("preview.mp4", frame_size=(4, 4))
    writer.write_frames(np.ones((2, 4, 4)), inscribe_frame_number=False, progress_bar=False)
    assert process.stdin.write.call_count == 2
    assert all(len(item.args[0]) == 24 for item in process.stdin.write.call_args_list)


@pytest.mark.parametrize("frames, indices, exception", [(np.zeros((2, 4, 4)), [0], AssertionError), (np.zeros((2, 4)), [0, 1], RuntimeError)])
def test_should_reject_invalid_frame_data_when_writing_preview(process_factory, frames, indices, exception):
    _, process = process_factory
    writer = io.MP4WriterPreview("preview.mp4", frame_size=(4, 4))
    with pytest.raises(exception) as error:
        writer.write_frames(frames, frames_idx=indices)
    assert error.value is not None
    process.stdin.write.assert_not_called()


def test_should_propagate_broken_pipe_when_encoder_exits_during_write(process_factory):
    _, process = process_factory
    failure = BrokenPipeError("encoder exited")
    process.stdin.write.side_effect = failure
    writer = io.AviWriter("video.avi")
    with pytest.raises(BrokenPipeError) as error:
        writer.write_frames(np.ones((1, 2, 2)), progress_bar=False)
    assert error.value is failure


@pytest.mark.parametrize("extension, name", [("dat", "RawFileReader"), ("avi", "AviReader")])
def test_should_select_reader_by_extension_when_opening_video(monkeypatch, extension, name):
    constructor = Mock(return_value=object())
    monkeypatch.setattr(io, name, constructor)
    result = io.AutoReader(f"cam.{extension}", threads=2)
    assert result is constructor.return_value
    constructor.assert_called_once_with(f"cam.{extension}", threads=2)


@pytest.fixture
def raw_reader(monkeypatch):
    frames = np.arange(24, dtype=np.uint16).reshape(4, 2, 3)
    stream = streams.BytesIO(frames.tobytes())
    monkeypatch.setattr(io.toml, "load", Mock(return_value={"cam": {"Width": 3, "Height": 2, "PixelFormat": "Coord3D_C16"}}))
    real_stat = io.os.stat
    monkeypatch.setattr(io.os, "stat", lambda name, *args, **kwargs: SimpleNamespace(st_size=frames.nbytes) if str(name) == "cam.dat" else real_stat(name, *args, **kwargs))
    monkeypatch.setattr(io, "open", Mock(return_value=stream), raising=False)
    # np.fromfile requires a real descriptor; this boundary fake reads bytes.
    monkeypatch.setattr(io.np, "fromfile", lambda file, dtype, count: np.frombuffer(file.read(count * dtype.itemsize), dtype=dtype).copy())
    return io.RawFileReader("cam.dat"), frames, stream


@pytest.mark.parametrize("selection, indices", [(1, 1), (range(1, 3), [1, 2]), ([3, 1], [3, 1]), (range(0, 4, 2), [0, 2])])
def test_should_read_requested_frames_when_using_raw_reader(raw_reader, selection, indices):
    reader, frames, stream = raw_reader
    result = reader.get_frames(selection)
    reader.close()
    assert_array_equal(result, frames[indices])
    assert stream.closed
    assert reader.nframes == 4
    assert reader.dims == (4, 2, 3)


def test_should_reject_unknown_selection_when_reading_raw_frames(raw_reader):
    reader, _, _ = raw_reader
    with pytest.raises(RuntimeError, match="frame range type") as error:
        reader.get_frames("bad")
    assert error.value is not None


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=ValueError, reason='RawFileReader counts frames rather than pixels when reading the entire file')
def test_should_read_all_raw_frames_when_selection_is_none(raw_reader):
    reader, frames, _ = raw_reader
    result = reader.get_frames()
    assert_array_equal(result, frames)


@pytest.mark.parametrize("selection, expected", [((1, 2), slice(1, 3)), ((1, 20), slice(1, 4))])
@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=TypeError, reason='RawFileReader mutates an immutable tuple; interval counts and bounds also need correction')
def test_should_read_interval_when_raw_frame_selection_is_a_tuple(raw_reader, selection, expected):
    reader, frames, _ = raw_reader
    result = reader.get_frames(selection)
    assert_array_equal(result, frames[expected])


@pytest.mark.parametrize("reader_class", [io.RawFileReader, io.AviReader])
@pytest.mark.parametrize("configured", [False, True])
def test_should_undistort_or_warn_when_calibration_is_present_or_missing(reader_class, configured, monkeypatch):
    reader = reader_class.__new__(reader_class)
    reader.intrinsic_matrix = np.eye(3) if configured else None
    reader.distortion_coeffs = np.zeros(5) if configured else None
    frames = np.zeros((2, 2, 2))
    undistort = Mock(side_effect=lambda frame, *args: frame + 1)
    monkeypatch.setattr(io.cv2, "undistort", undistort)
    if configured:
        result = reader.undistort_frames(frames, progress_bar=False)
    else:
        with pytest.warns(UserWarning, match="skipping undistortion"):
            result = reader.undistort_frames(frames, progress_bar=False)
    assert result is frames
    assert_allclose(result, int(configured))
    assert undistort.call_count == (2 if configured else 0)


@pytest.mark.parametrize("depth, count, expected_dtype, expected_count", [("16", "4", np.dtype("<u2"), 4), ("8", "4", np.dtype("<u1"), 4), ("N/A", "N/A", np.dtype("<u1"), 200000)])
def test_should_parse_probe_metadata_when_opening_avi(process_factory, depth, count, expected_dtype, expected_count):
    factory, process = process_factory
    process.communicate.return_value = (f"3\n2\ngray\n100/1\n{depth}\n{count}\n".encode(), b"probe warning")
    reader = io.AviReader("cam.avi", prepend_args="env TASK_TEST=1")
    reader.open()
    reader.close()
    assert reader.frame_size == (3, 2)
    assert reader.dtype == expected_dtype
    assert reader.nframes == expected_count
    assert reader.fps == 100
    assert factory.call_args.args[0].startswith("env TASK_TEST=1 ; ffprobe")


def test_should_reject_unsupported_bit_depth_when_probe_reports_it(process_factory):
    _, process = process_factory
    process.communicate.return_value = (b"3\n2\ngray\n100/1\n12\n4\n", b"")
    with pytest.raises(RuntimeError, match="bit depth 12") as error:
        io.AviReader("cam.avi")
    assert error.value is not None


@pytest.fixture
def avi_reader(process_factory):
    factory, process = process_factory
    process.communicate.return_value = (b"3\n2\ngray\n100/1\n8\n20\n", b"")
    reader = io.AviReader("cam.avi", threads=2)

    def decode(command, **kwargs):
        if " -vf " in command:
            numbers = sorted(int(value.split("\\,")[1].split(")")[0]) for value in command.split("select=")[1].split(" -vsync")[0].split("+"))
        else:
            start = 0
            if "-ss" in command:
                value = command.split("-ss ")[1].split()[0]
                h, m, s = map(float, value.split(":"))
                start = round((h * 3600 + m * 60 + s) * 100)
            count = int(command.split("-vframes ")[1].split()[0])
            numbers = range(start, start + count)
        frames = np.array([np.full((2, 3), number, np.uint8) for number in numbers])
        result = Mock()
        result.communicate.return_value = (frames.tobytes(), b"")
        return result

    factory.side_effect = decode
    return reader, factory, process


@pytest.mark.parametrize("selection, fast_seek, expected", [(None, False, list(range(20))), (range(2, 5), False, [2, 3, 4]), ([2, 3, 4], False, [2, 3, 4]), ([4, 1, 3], False, [4, 1, 3]), ([15, 1], False, [15, 1]), ([4, 1, 3], True, [4, 1, 3]), (range(1, 6, 2), False, [1, 3, 5]), (2, False, [2])])
def test_should_select_and_reorder_decoded_frames_when_avi_selection_varies(avi_reader, selection, fast_seek, expected):
    reader, factory, _ = avi_reader
    result = reader.get_frames(selection, fast_seek=fast_seek)
    assert_array_equal(np.atleast_3d(result).reshape(-1, 2, 3)[:, 0, 0], expected)
    assert "ffmpeg" in factory.call_args.args[0]


def test_should_return_none_when_decoder_reports_an_error(avi_reader):
    reader, factory, process = avi_reader
    factory.side_effect = None
    process.communicate.return_value = (b"", b"decoder failed")
    result = reader.get_frames(1)
    assert result is None


@pytest.mark.parametrize("selection", ["bad", object()])
def test_should_reject_unknown_selection_when_reading_avi(avi_reader, selection):
    reader, _, _ = avi_reader
    with pytest.raises(RuntimeError, match="frame range") as error:
        reader.get_frames(selection)
    assert error.value is not None


@pytest.mark.parametrize("pixel_format, channels", [("gray16le", 1), ("bgr0", 4), ("rgba", 4), ("rgb24", 3)])
def test_should_shape_decoded_channels_when_pixel_format_varies(avi_reader, pixel_format, channels):
    reader, factory, process = avi_reader
    reader.pixel_format = pixel_format
    reader.threads = None
    reader.prepend_args = "env TASK_TEST=1"
    factory.side_effect = None
    process.communicate.return_value = (np.ones((2, 2, 3, channels), np.uint8).tobytes(), b"")
    result = reader.get_frames(range(2))
    assert result.shape == ((2, 2, 3) if channels == 1 else (2, 2, 3, channels))
    assert factory.call_args.args[0].startswith("env TASK_TEST=1 ; ffmpeg")


def test_should_reject_unknown_pixel_format_when_decoding_frames(avi_reader):
    reader, _, _ = avi_reader
    reader.pixel_format = "unsupported"
    with pytest.raises(RuntimeError, match="pixel format") as error:
        reader.get_frames(0)
    assert error.value is not None


@pytest.mark.parametrize("factor", [1, 2])
def test_should_resize_or_return_original_when_downsampling_frames(factor):
    frames = np.full((2, 4, 6), 7, np.uint16)
    result = io.downsample_frames(frames, downsample=factor)
    assert result.shape == (2, 4 // factor, 6 // factor)
    assert_array_equal(result, 7)
    assert (result is frames) == (factor == 1)


@pytest.mark.parametrize("named_avi", [False, True])
def test_should_close_readers_and_use_camera_config_when_reading_multiple_cameras(monkeypatch, named_avi):
    reader = type("AviReader", (Mock,), {})() if named_avi else Mock()
    reader.get_frames.return_value = np.ones((2, 4, 4), np.uint16)
    reader.undistort_frames.side_effect = lambda data, **kwargs: data
    factory = Mock(return_value=reader)
    monkeypatch.setattr(io, "AutoReader", factory)
    result = io.read_frames_multicam({"cam.avi": "cam"}, {"cam": [0, 1]}, config={"cam": {"threads": 2}} if named_avi else {}, downsample=2, progress_bar=False)
    assert result["cam"].shape == (2, 2, 2)
    reader.close.assert_called_once()
    assert reader.get_frames.call_args.kwargs == ({"fast_seek": True} if named_avi else {})


@pytest.mark.parametrize("capture_field", ["frame_id", "capture_number"])
def test_should_scale_timestamps_and_normalize_capture_column_when_reading_table(monkeypatch, capture_field):
    table = f"{capture_field}\tsystem_timestamp\tdevice_timestamp\n0\t1000\t2000\n1\t2000\t3000\n"
    monkeypatch.setattr(io, "open", mock_open(read_data=table), raising=False)
    result = io.read_timestamps("timestamps.txt", tick_period=1000)
    assert_array_equal(result["frame_id"], [0, 1])
    assert_allclose(result["system_timestamp"], [1, 2])
    assert_allclose(result["device_timestamp"], [2, 3])
    assert result.index.name == "frame_index"


def test_should_fill_capture_gaps_and_drop_terminal_nan_when_padding_timestamps():
    table = pd.DataFrame({"frame_id": [0, 3, 4], "device_timestamp": [0., .03, np.nan]})
    original = table.copy()
    result = io.fill_timestamps(table)
    assert_array_equal(result.index, [0, 1, 2, 3])
    assert_allclose(result["device_timestamp"], [0, .01, .02, .03])
    assert result["frame_index"].isna().tolist() == [False, True, True, False]
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("multiplexed", [False, True])
@pytest.mark.parametrize("even, first, full", [(True, False, True), (False, True, False)])
def test_should_synchronize_camera_offsets_and_pair_exposures_when_reading_multicam_timestamps(monkeypatch, multiplexed, even, first, full):
    def load(filename, **kwargs):
        offset = 0 if filename == "a.txt" else 5
        result = pd.DataFrame({"frame_id": np.arange(8), "device_timestamp": np.arange(8) * .01 + offset, "system_timestamp": np.arange(8) * .01})
        result.index.name = "frame_index"
        return result
    monkeypatch.setattr(io, "read_timestamps", Mock(side_effect=load))
    result = io.read_timestamps_multicam({"a.txt": "a", "b.txt": "b"}, burn_in=0, multiplexed=multiplexed, is_fluorescence_even=even, is_fluo_first=first, return_full_sync_only=full)
    if multiplexed:
        _, _, fluorescence, reflectance = result
        adjustment = 1 if first else -1
        assert_array_equal(reflectance.index, fluorescence.index + adjustment)
        assert np.all(fluorescence["a", "frame_id"] % 2 == (0 if even else 1))
        assert len(fluorescence) == 3
    else:
        _, merged = result
        assert len(merged) == 8
        assert_array_equal(merged["a", "frame_id"], merged["b", "frame_id"])


def test_should_normalize_single_camera_reference_when_no_merge_is_needed(monkeypatch):
    table = pd.DataFrame({"frame_id": [0, 1], "device_timestamp": [5., 5.01]})
    table.index.name = "frame_index"
    monkeypatch.setattr(io, "read_timestamps", Mock(return_value=table))
    result = io.read_timestamps_multicam({"a.txt": "a"})
    assert_allclose(result["device_timestamp_ref"], [0, .01])
    assert ("a", "frame_id") in result.columns


@pytest.mark.parametrize("name, dtype", [("Coord3D_C16", np.dtype("<u2")), ("Mono8", np.dtype("<u1"))])
def test_should_map_camera_pixel_format_when_supported(name, dtype):
    pixel_format = name
    result = io.pixel_format_to_np_dtype(pixel_format)
    assert result == dtype


def test_should_reject_unsupported_camera_pixel_format_when_mapping_dtype():
    pixel_format = "RGB8"
    with pytest.raises(RuntimeError, match="pixel format") as error:
        io.pixel_format_to_np_dtype(pixel_format)
    assert error.value is not None


def test_should_format_camera_calibration_when_intrinsics_are_complete():
    config = {"cam": dict(zip(["CalibFocalLengthX", "CalibFocalLengthY", "CalibOpticalCenterX", "CalibOpticalCenterY", "k1", "k2", "p1", "p2", "k3"], [2, 3, 4, 5, .1, .2, .3, .4, .5]))}
    matrices, coefficients = io.format_intrinsics(config)
    assert_allclose(matrices["cam"], [[2, 0, 4], [0, 3, 5], [0, 0, 1]])
    assert_allclose(coefficients["cam"], [.1, .2, .3, .4, .5])


@pytest.mark.parametrize("name, expected", [("turbo", cv2.COLORMAP_TURBO), ("ViRiDiS", cv2.COLORMAP_VIRIDIS), (3, 3)])
def test_should_resolve_colormap_when_name_or_integer_is_supplied(name, expected):
    colormap = name
    result = io.get_cv2_colormap(colormap)
    assert result == expected


def test_should_list_available_colormaps_when_unknown_name_is_supplied():
    colormap = "missing"
    with pytest.raises(ValueError, match="Unknown colormap") as error:
        io.get_cv2_colormap(colormap)
    assert "TURBO" in str(error.value)


@pytest.mark.parametrize("stack", [np.zeros((3, 4)), np.zeros((2, 3, 4)), np.array([[[-1., 0, 2], [0, 1, 3]]] * 2)])
def test_should_colorize_and_clip_intensity_when_pseudocoloring_frames(stack):
    original = stack.copy()
    result = io.pseudocolor_frames(stack, cmap="turbo", vmin=0, vmax=1)
    assert result.dtype == np.uint8
    assert result.shape == (stack.shape + (3,))
    assert_array_equal(stack, original)
    if stack.ndim == 3 and stack.shape[2] == 3:
        assert_array_equal(result[0, 0, 0], result[0, 0, 1])


def test_should_use_data_range_when_pseudocolor_limits_are_omitted():
    frames = np.arange(24).reshape(2, 3, 4)
    result = io.pseudocolor_frames(frames)
    assert result.shape == (2, 3, 4, 3)
    assert not np.array_equal(result[0, 0, 0], result[1, -1, -1])


def test_should_draw_text_and_marker_in_place_when_annotating_frames():
    frame = np.zeros((60, 80, 3), np.uint8)
    io.inscribe_text(frame, "1")
    io.mark_frame(frame, [0, 0, 255], .2)
    assert frame.any()
    assert_array_equal(frame[-1, -1], [0, 0, 255])


def test_should_blend_category_colors_when_inscribing_masks():
    frames = np.full((1, 2, 2, 3), 100, np.uint8)
    masks = np.array([[[1, 2], [0, 1]]])
    result = io.inscribe_masks(frames, masks)
    assert_array_equal(result[0, 0, 0], [177, 50, 50])
    assert_array_equal(result[0, 0, 1], [50, 50, 177])
    assert_array_equal(result[0, 1, 0], [50, 50, 50])


@pytest.mark.parametrize("limits, interpolate", [(None, False), ((None, 20), False), ((5, None), True)])
def test_should_aggregate_valid_depth_and_close_reader_when_computing_background(monkeypatch, limits, interpolate):
    frames = np.full((2, 4, 4), 10, dtype=np.uint16)
    frames[:, 0, 0] = 0
    reader = Mock(nframes=2)
    reader.get_frames.return_value = frames
    reader.undistort_frames.side_effect = lambda data: data
    monkeypatch.setattr(io, "AutoReader", Mock(return_value=reader))
    warning_context = pytest.warns(RuntimeWarning, match="Mean of empty slice") if interpolate else nullcontext()
    with warning_context:
        result = io.get_bground("cam.avi", spacing=1, valid_range=limits, interpolate_invalid=interpolate, median_kernels=[3])
    assert_allclose(result[1:, 1:], 10)
    reader.close.assert_called_once()
    assert list(reader.get_frames.call_args.args[0]) == [0, 1]
