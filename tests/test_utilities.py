from unittest.mock import Mock, mock_open

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids import util
from markovids.vid import util as video_util


@pytest.mark.parametrize("size, expected", [(1, [[0], [1], [2], [3], [4]]), (2, [[0, 1], [2, 3], [4]]), (10, [[0, 1, 2, 3, 4]])])
def test_should_yield_complete_and_partial_batches_when_chunking_sequence(size, expected):
    data = list(range(5))
    result = list(util.batch(data, size))
    assert result == expected


def test_should_yield_no_batches_when_sequence_is_empty():
    data = []
    result = list(util.batch(data, 3))
    assert result == []


def test_should_reject_zero_batch_size_when_chunking_sequence():
    data = [1]
    with pytest.raises(ValueError) as error:
        list(util.batch(data, 0))
    assert error.value is not None


def test_should_threshold_before_exponentiation_when_squashing_confidence():
    scores = np.array([0, .05, .1, .5, 1, np.nan])
    result = util.squash_conf(scores)
    assert_allclose(result, [0, 0, .01, .25, 1, 0])


def test_should_use_per_keypoint_cutoffs_and_ignore_invalid_indices_when_squashing_dynamic_confidence():
    scores = np.array([[[.1], [.2], [.8]]])
    result = util.squash_conf_dynamic(scores, {-1: .9, 0: .1, 1: .1, 3: .9})
    assert_allclose(result, [[[0], [.04], [.64]]])
    assert_array_equal(scores, [[[.1], [.2], [.8]]])


@pytest.mark.parametrize("value, next_value, previous", [(0, 0, 0), (1, 2, 0), (2, 2, 2), (2.1, 4, 2), (-1, 0, -2)])
def test_should_round_to_even_boundary_when_number_is_positive_or_negative(value, next_value, previous):
    number = np.float64(value)
    next_result, previous_result = util.next_even_number(number), util.prev_even_number(number)
    assert next_result == next_value
    assert previous_result == previous


def test_should_preserve_missing_positions_and_index_when_smoothing_series():
    series = pd.Series([0., 1, np.nan, 2, 3, 4, 5, 6], index=list("abcdefgh"))
    result = util.savgol_filter_missing(series, window_length=5, poly_order=1)
    assert result.index.equals(series.index)
    assert np.isnan(result.iloc[2])
    assert_allclose(result.dropna(), np.arange(7), atol=1e-12)


@pytest.mark.parametrize("replace", [True, False])
def test_should_replace_or_remove_outliers_when_hampel_filtering_series(replace):
    series = pd.Series([0., 1, 0, 1, 100, 1, 0, 1, 0])
    original = series.copy()
    result = util.hampel(series, window=5, replace=replace, insert_nans=False)
    assert result.iloc[4] == 1 if replace else np.isnan(result.iloc[4])
    pd.testing.assert_series_equal(series, original)


def test_should_insert_nans_when_rolling_deviation_is_undefined():
    series = pd.Series([1.] * 5)
    result = util.hampel(series)
    assert result.isna().all()


def test_should_tile_videos_and_leave_unused_cells_empty_when_montage_is_incomplete():
    videos = [np.full((2, 3, 4, 1), value, np.uint8) for value in [1, 2, 3]]
    result = video_util.video_montage(videos, ncols=2)
    assert result.shape == (2, 6, 8, 1)
    assert_array_equal(result[:, :3, :4], 1)
    assert_array_equal(result[:, :3, 4:], 2)
    assert_array_equal(result[:, 3:, :4], 3)
    assert_array_equal(result[:, 3:, 4:], 0)


def test_should_return_original_video_when_montage_has_one_input():
    video = np.ones((2, 3, 4, 1))
    result = video_util.video_montage([video])
    assert result is video


def test_should_reject_inconsistent_dimensions_when_creating_montage():
    videos = [np.zeros((2, 3, 4, 1)), np.zeros((1, 3, 4, 1))]
    with pytest.raises(RuntimeError, match="dimensions not consistent") as error:
        video_util.video_montage(videos)
    assert error.value is not None


@pytest.mark.parametrize("has_hole", [False, True])
def test_should_inpaint_only_near_mouse_region_when_depth_contains_holes(has_hole, monkeypatch):
    image = np.full((10, 10), 100, np.float32)
    if has_hole:
        image[5, 5] = 0
    inpainter = Mock(side_effect=lambda frame, *args: np.full_like(frame, 100))
    monkeypatch.setattr(video_util.cv2, "inpaint", inpainter)
    result = video_util.fill_holes(image)
    assert_allclose(result, 100)
    assert inpainter.call_count == int(has_hole)
    if has_hole:
        assert inpainter.call_args.args[1][5, 5] == 1


def test_should_preserve_constant_signal_when_low_pass_filtering():
    signal = np.full((50, 2, 2), 10.)
    result = video_util.sos_filter(signal, fps=100, tau=.1)
    assert_allclose(result, signal, atol=1e-10)


@pytest.mark.parametrize("clip", [True, False])
def test_should_clip_negative_bandpass_values_when_requested(clip):
    image = np.zeros((15, 15), np.float32)
    image[7, 7] = 100
    blurred = video_util.lp_filter(image, 1)
    result = video_util.bp_filter(image, 1, 3, clip=clip)
    assert blurred[7, 7] < 100
    assert result[7, 7] > 0
    assert (result.min() >= 0) == clip


@pytest.mark.parametrize("module", [util, video_util])
@pytest.mark.parametrize("cached, force", [(False, False), (True, False), (True, True)])
def test_should_use_cache_or_compute_and_save_when_background_is_requested(monkeypatch, module, cached, force):
    import tifffile
    import toml
    from markovids.vid import io
    background = np.full((3, 3), 15.8)
    compute = Mock(return_value=background)
    write = Mock()
    read = Mock(return_value=background)
    monkeypatch.setattr(io, "get_bground", compute)
    monkeypatch.setattr(tifffile, "imwrite", write)
    monkeypatch.setattr(tifffile, "imread", read)
    monkeypatch.setattr(module.os if module is util else __import__("os"), "makedirs", Mock())
    monkeypatch.setattr(__import__("os").path, "exists", lambda name: cached)
    monkeypatch.setattr(module, "open", mock_open(), raising=False)
    monkeypatch.setattr(toml, "dump", Mock())
    result = module.compute_bground("/session/cam.avi", force=force)
    reused = cached and not force
    assert compute.call_count == int(not reused)
    assert write.call_count == int(not reused)
    assert read.call_count == int(reused)
    assert_allclose(result, 15.8 if reused else 15)
    assert result.dtype == (background.dtype if reused else np.uint16)
