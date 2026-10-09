import shlex
import sys
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import MagicMock, Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from markovids.pcl import viz


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_should_quote_output_path_when_starting_custom_ffmpeg_writer(monkeypatch):
    writer = viz.CustomFFMpegWriter.__new__(viz.CustomFFMpegWriter)
    arguments = ["ffmpeg"] + ["x"] * 19 + ["video with spaces.mp4"]
    monkeypatch.setattr(writer, "_args", Mock(return_value=arguments))
    process = Mock()
    factory = Mock(return_value=process)
    monkeypatch.setattr(viz.subprocess, "Popen", factory)
    writer._run()
    assert writer._proc is process
    command = factory.call_args.args[0]
    assert command[:2] == ["bash", "-c"]
    assert command[2].endswith(shlex.quote("video with spaces.mp4"))


def test_should_cap_trails_and_reset_missing_points_when_updating_trail_buffers():
    axis = plt.figure().add_subplot(projection="3d")
    trails = viz.KeypointTrails3D(axis, 2, trail_length=2)
    trails.update(np.array([[0., 0, 0], [1, 1, 1]]))
    trails.update(np.array([[2., 2, 2], [3, 3, 3]]))
    trails.update(np.array([[4., 4, 4], [np.nan] * 3]))
    assert_array_equal(trails.buffers[0], [[2, 2, 2], [4, 4, 4]])
    assert trails.buffers[1] == []
    assert len(trails.collections[0]._segments3d) == 1
    assert len(trails.collections[1]._segments3d) == 0


def test_should_add_plane_grid_lines_and_tick_labels_when_drawing_height_and_floor_grids():
    axis = plt.figure().add_subplot(projection="3d")
    limits = (0, 100)
    viz.add_height_grid(axis, limits, limits, limits, tick_spacing=50)
    viz.add_floor_grid(axis, limits, limits, limits, tick_spacing=50)
    assert len(axis.collections) == 3
    assert len(axis.collections[0]._segments3d) == 12
    assert len(axis.collections[2]._segments3d) == 6
    assert len(axis.texts) == 9


@pytest.mark.parametrize("annotated, frame_ids", [(True, [10, 11, 12]), (False, None)])
def test_should_render_each_frame_and_hide_missing_points_when_exporting_matplotlib_video(monkeypatch, annotated, frame_ids):
    points = np.array([[[0., 0, 0], [1, 2, 3]], [[2, 3, 4], [np.nan] * 3], [[4, 5, 6], [7, 8, 9]]])
    writer = Mock()
    writer.saving.side_effect = lambda *args, **kwargs: nullcontext()
    factory = Mock(return_value=writer)
    monkeypatch.setattr(viz, "CustomFFMpegWriter", factory)
    viz.visualize_xyz_trajectories_to_mp4(points, "fake.mp4", skeleton_edges=[(0, 1)] if annotated else None, trail_length=2 if annotated else 0, show_labels=annotated, frame_ids=frame_ids, progress_bar=False, xlim=(0, 10), ylim=(0, 10), zlim=(0, 10))
    assert writer.grab_frame.call_count == 3
    factory.assert_called_once_with(fps=100)
    axis = plt.gcf().axes[0]
    assert axis.get_title() == ("Frame 12" if annotated else "Frame 2")
    assert_array_equal(np.asarray(axis.collections[1]._offsets3d).ravel(), [7, 8, 9])
    if annotated:
        assert len(axis.lines) == 1
        assert len(axis.lines[0].get_xdata()) == 2


@pytest.fixture
def vedo_environment(monkeypatch):
    module = ModuleType("vedo")
    plotter = MagicMock()
    plotter.__iadd__.return_value = plotter
    video = Mock()
    points, lines = [], []

    def point(*args, **kwargs):
        actor = Mock()
        points.append(actor)
        return actor

    def line(*args, **kwargs):
        actor = Mock()
        lines.append(actor)
        return actor

    module.Plotter = Mock(return_value=plotter)
    module.Video = Mock(return_value=video)
    module.Point = Mock(side_effect=point)
    module.Line = Mock(side_effect=line)
    module.Points = Mock()
    module.Plane = Mock()
    module.Arrows = Mock()
    module.Text2D = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "vedo", module)
    return module, plotter, video, points, lines


@pytest.mark.parametrize("bounded", [False, True])
def test_should_update_points_skeleton_and_frame_labels_when_exporting_vedo_video(vedo_environment, bounded):
    module, plotter, video, actors, lines = vedo_environment
    data = np.array([[[0., 0, 0], [1, 2, 3]], [[2, 3, 4], [np.nan] * 3]])
    kwargs = {"xlim": (0, 10), "ylim": (0, 10), "zlim": (0, 10), "frame_ids": [10, 11]} if bounded else {"colors": [(1, 0, 0), (0, 0, 1)], "azimuth": None}
    viz.visualize_xyz_trajectories_vedo(data, "fake.mp4", skeleton_edges=[(0, 1)], progress_bar=False, **kwargs)
    assert actors[0].pos.call_count == 2
    assert actors[1].pos.call_count == 1
    assert actors[1].point.shape == (0, 3)
    assert lines[-1].points.shape == (0, 3)
    assert video.add_frame.call_count == 2
    video.close.assert_called_once()
    plotter.close.assert_called_once()
    plotter.show.assert_called_once_with(resetcam=True, zoom=1.2)
    plotter.render.assert_called_once()
    assert module.Plane.call_count == int(bounded)
    assert module.Text2D.return_value.text.call_args.args == ("Frame 11" if bounded else "Frame 1",)
