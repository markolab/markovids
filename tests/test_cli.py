import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, mock_open

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from numpy.testing import assert_allclose, assert_array_equal

import markovids.cli as cli
from markovids import util
from markovids.vid import io


@pytest.mark.parametrize("arguments", [["--help"], ["crop-video", "--help"], ["sync-depth-video", "--help"]])
def test_should_show_available_commands_and_options_when_help_is_requested(arguments):
    runner = CliRunner()
    result = runner.invoke(cli.cli, arguments)
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.parametrize("arguments, message", [(["missing"], "No such command"), (["crop-video"], "Missing argument"), (["crop-video", "/nonexistent/markovids.h5"], "does not exist")])
def test_should_report_usage_error_when_command_or_required_path_is_invalid(arguments, message):
    runner = CliRunner()
    result = runner.invoke(cli.cli, arguments)
    assert result.exit_code == 2
    assert message in result.output


@pytest.fixture
def cli_environment(monkeypatch, tmp_path):
    metadata = {
        "user_input": {"subject": "mouse", "session": "day1", "notes": "test"},
        "cli_parameters": {"hw_trigger_pulse_width": 20},
        "camera_metadata": {"cam": {"Width": 4, "Height": 4, "PixelFormat": "Mono8"}},
    }
    monkeypatch.setattr(cli.toml, "load", Mock(return_value=metadata))
    return SimpleNamespace(metadata=metadata, directory=tmp_path, runner=CliRunner())


@pytest.mark.parametrize("command, function", [("generate-qd-preview", "alternating_excitation_vid_preview"), ("split-qd-videos", "alternating_excitation_vid_split")])
def test_should_forward_timestamp_and_batch_options_when_excitation_command_is_invoked(cli_environment, monkeypatch, command, function):
    env = cli_environment
    process = Mock()
    monkeypatch.setattr(cli, function, process)
    result = env.runner.invoke(cli.cli, [command, str(env.directory), "--batch-size", "2", "--nbatches", "3", "--ts-burn-in", "0", "--ts-not-multiplexed", "--ts-allow-partial-sync"])
    assert result.exit_code == 0, result.output
    assert process.call_count == 1
    args, kwargs = process.call_args
    assert args[0] == {str(env.directory / "cam.avi"): "cam"}
    assert args[2]["cam"]["frame_size"] == (4, 4)
    assert kwargs["batch_size"] == 2
    assert kwargs["nbatches"] == 3
    assert kwargs["timestamp_kwargs"] == {"merge_tolerance": .003, "multiplexed": False, "burn_in": 0, "return_full_sync_only": False, "use_timestamp_field": "device_timestamp_ref"}


def test_should_stop_before_preview_processing_when_output_file_already_exists(cli_environment, monkeypatch):
    env = cli_environment
    preview = Mock()
    monkeypatch.setattr(cli, "alternating_excitation_vid_preview", preview)
    (env.directory / "mouse_day1_test_pulse-widths-20_reflectance.mp4").touch()
    monkeypatch.chdir(env.directory)
    result = env.runner.invoke(cli.cli, ["generate-qd-preview", str(env.directory)])
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    preview.assert_not_called()


def test_should_honor_environment_options_when_splitting_videos(cli_environment, monkeypatch):
    env = cli_environment
    split = Mock()
    monkeypatch.setattr(cli, "alternating_excitation_vid_split", split)
    result = env.runner.invoke(cli.cli, ["split-qd-videos", str(env.directory)], env={"MARKOVIDS_QD_PREVIEW_BATCH_SIZE": "7"})
    assert result.exit_code == 0, result.output
    assert split.call_args.kwargs["batch_size"] == 7


@pytest.mark.parametrize("force, existing", [(False, False), (False, True), (True, True)])
def test_should_forward_depth_configuration_or_skip_existing_output_when_sync_command_runs(cli_environment, monkeypatch, force, existing):
    env = cli_environment
    intrinsics = env.directory / "intrinsics.toml"
    intrinsics.touch()
    if existing:
        (env.directory / "_proc").mkdir()
    process = Mock()
    monkeypatch.setattr(util, "sync_depth_videos", process)
    matrices, coefficients = {"cam": np.eye(3)}, {"cam": np.zeros(5)}
    calibration = Mock(return_value=(matrices, coefficients))
    monkeypatch.setattr(io, "format_intrinsics", calibration)
    args = ["sync-depth-video", str(env.directory), "--intrinsics-file", str(intrinsics), "--batch-size", "2", "--no-undistort", "--bground-valid-range-min", "900"]
    if force:
        args.append("--force")
    result = env.runner.invoke(cli.cli, args)
    assert result.exit_code == 0, result.output
    assert process.call_count == int(force or not existing)
    if process.called:
        kwargs = process.call_args.kwargs
        assert kwargs["intrinsics_matrix"] is matrices
        assert kwargs["distortion_coeffs"] is coefficients
        assert kwargs["undistort"] is False
        assert kwargs["batch_size"] == 2
        assert kwargs["bground_kwargs"]["valid_range"] == (900, 2000)


def test_should_skip_undistortion_when_sync_callback_has_no_intrinsics_file(monkeypatch):
    callback = cli.cli_sync_depth_video.callback
    kwargs = {parameter.name: parameter.default for parameter in cli.cli_sync_depth_video.params if parameter.name != "data_dir"}
    kwargs.update(data_dir="/session", intrinsics_file="/missing.toml")
    process = Mock()
    monkeypatch.setattr(util, "sync_depth_videos", process)
    monkeypatch.setattr(cli.os.path, "exists", lambda filename: False)
    callback(**kwargs)
    assert process.call_args.kwargs["intrinsics_matrix"] is None
    assert process.call_args.kwargs["distortion_coeffs"] is None


def test_should_save_scalars_and_metadata_when_compute_callback_completes(monkeypatch):
    frame = Mock()
    compute = Mock(return_value=frame)
    monkeypatch.setattr(util, "compute_scalars", compute, raising=False)
    monkeypatch.setattr(cli.os, "makedirs", Mock())
    monkeypatch.setattr(cli, "open", mock_open(), raising=False)
    dump = Mock()
    monkeypatch.setattr(cli.toml, "dump", dump)
    cli.cli_compute_scalars.callback("/session/_registration/data.h5", batch_size=2, intrinsics_file="intrinsics.toml", scalar_dir="_scalars", z_range=(1, 10))
    compute.assert_called_once_with("/session/_registration/data.h5", "intrinsics.toml", batch_size=2, z_range=(1, 10))
    frame.to_parquet.assert_called_once_with("/session/_scalars/scalars.parquet")
    assert dump.call_args.args[0]["batch_size"] == 2


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='compute-scalars Click options do not match its callback; scalar extraction backend is also absent')
def test_should_report_unimplemented_scalar_extraction_without_creating_outputs(tmp_path):
    registration, calibration = tmp_path / "_registration" / "data.h5", tmp_path / "intrinsics.toml"
    registration.parent.mkdir()
    registration.touch()
    calibration.touch()
    result = CliRunner().invoke(cli.cli, ["compute-scalars", str(registration), "--intrinsics-file", str(calibration)])
    assert result.exit_code == 1
    assert "scalar extraction is not implemented" in result.output
    assert not (tmp_path / "_scalars").exists()


@pytest.fixture
def crop_environment(monkeypatch, memory_h5, tmp_path):
    registration = tmp_path / "_registration" / "data.h5"
    registration.parent.mkdir()
    registration.touch()
    source = memory_h5.File({"frames": np.arange(5 * 4 * 4, dtype=np.uint16).reshape(5, 4, 4)})
    memory_h5.files[str(registration)] = source
    scalars = pd.DataFrame({"x_mean_px": np.full(5, 2), "y_mean_px": np.full(5, 2), "orientation_rad": np.zeros(5), "orientation_rad_unwrap": np.zeros(5)})
    monkeypatch.setattr(pd, "read_parquet", Mock(return_value=scalars))
    parquet = Mock()
    monkeypatch.setattr(pd.DataFrame, "to_parquet", parquet)
    monkeypatch.setattr(cli, "crop_and_rotate_frames", Mock(side_effect=lambda data, *args, **kwargs: data.copy()))
    monkeypatch.setattr(cli, "open", mock_open(), raising=False)
    monkeypatch.setattr(cli.toml, "dump", Mock())
    monkeypatch.setattr(cli.toml, "load", Mock(return_value={"frame_downsample": 1, "classifier_categories": {"flip": 1}}))
    writer = Mock()
    monkeypatch.setattr(cli, "MP4WriterPreview", Mock(return_value=writer))
    monkeypatch.setattr(cli.os, "makedirs", Mock())
    ort = ModuleType("onnxruntime")
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="frames")]
    session.run.side_effect = lambda output, values: [None, np.tile([.1, .9], (len(values["frames"]), 1))]
    ort.InferenceSession = Mock(return_value=session)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    model = tmp_path / "model.onnx"
    model.touch()
    return SimpleNamespace(registration=registration, source=source, frames=source["frames"].copy(), files=memory_h5.files, writer=writer, scalars=scalars, model=model, ort=ort, parquet=parquet)


@pytest.mark.parametrize("flip", [False, True])
def test_should_crop_write_batches_and_apply_predicted_flips_when_crop_command_runs(crop_environment, flip):
    env = crop_environment
    arguments = ["crop-video", str(env.registration), "--batch-size", "2", "--crop-size", "4", "4"]
    if flip:
        arguments.extend(["--flip-model", str(env.model), "--flip-model-proba-smoothing", "1"])
    result = CliRunner().invoke(cli.cli, arguments)
    assert result.exit_code == 0, result.output
    output = env.files[str(env.registration.parent.parent / "_crop" / "data.h5")]
    expected = np.rot90(env.frames, 2, axes=(1, 2)) if flip else env.frames
    assert_array_equal(output["cropped_frames"], expected)
    assert output.closed
    assert [len(item.args[0]) for item in env.writer.write_frames.call_args_list] == [2, 2, 1]
    env.writer.close.assert_called_once()
    assert env.ort.InferenceSession.call_count == int(flip)
    assert env.parquet.call_count == int(flip)


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='crop-video overwrites its input HDF5 handle with the TOML stream and leaves it open')
def test_should_close_input_handle_when_cropping_finishes(crop_environment):
    env = crop_environment
    result = CliRunner().invoke(cli.cli, ["crop-video", str(env.registration), "--crop-size", "4", "4"])
    assert result.exit_code == 0, result.output
    assert env.source.closed
