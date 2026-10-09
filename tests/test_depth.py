from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.depth import filter as filtering, io, moments, plane, track
from markovids.vid import util as video_util


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("fill", ["median", 7, "leave"])
def test_should_handle_temporal_outliers_when_hampel_filtering(inplace, fill):
    frames = np.full((9, 2, 2), 10.0)
    frames[4, 0, 0] = 100
    original = frames.copy()
    result = filtering.time_hampel_filter(frames, length=3, fill_value=fill, correction=None, inplace=inplace)
    assert result[4, 0, 0] == ({"median": 10, 7: 7, "leave": 100}[fill])
    assert (result is frames) == inplace
    if not inplace:
        assert_array_equal(frames, original)


def test_should_clean_spatial_noise_without_mutation_when_filtering_frames():
    frames = np.full((2, 9, 9), 10, dtype=np.uint16)
    frames[:, 4, 4] = 255
    original = frames.copy()
    result = filtering.clean_frames(frames, prefilter_space=(3,), iters_min=1, iters_tail=1, progress_bar=False)
    assert result.dtype == np.uint8
    assert_array_equal(result, 10)
    assert_array_equal(frames, original)


def test_should_only_cast_and_copy_when_spatial_filters_are_disabled():
    frames = np.arange(8).reshape(2, 2, 2)
    result = filtering.clean_frames(frames, prefilter_space=None, frame_dtype="float32", progress_bar=False)
    assert_array_equal(result, frames)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, frames)


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=NameError, reason='clean_frames uses scipy.signal without importing it')
def test_should_filter_temporal_noise_when_temporal_prefilter_is_enabled():
    frames = np.full((5, 2, 2), 10, dtype=np.uint8)
    frames[2] = 200
    result = filtering.clean_frames(frames, prefilter_space=None, prefilter_time=(3,), progress_bar=False)
    assert_array_equal(result, 10)


@pytest.mark.parametrize("function", [track.get_largest_cc, track.get_largest_contour])
def test_should_select_largest_foreground_region_when_components_differ_in_size(function):
    image = np.zeros((20, 20), dtype=np.uint8)
    image[1:4, 1:4] = 255
    image[8:16, 8:16] = 255
    result = function(image)
    assert result[10, 10]
    assert not result[2, 2]
    assert result.sum() == 64


@pytest.mark.parametrize("function", [track.get_largest_cc, track.get_largest_contour])
def test_should_return_none_when_no_foreground_component_exists(function):
    image = np.zeros((5, 5), dtype=np.uint8)
    result = function(image)
    assert result is None


@pytest.mark.parametrize("inplace", [True, False])
def test_should_fill_enclosed_holes_when_cleaning_binary_image(inplace):
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:8, 2:8] = 255
    image[4:6, 4:6] = 0
    original = image.copy()
    result = track.binary_fill_holes(image, inplace=inplace)
    assert result[4, 4] == 255
    assert (result is image) == inplace
    if not inplace:
        assert_array_equal(image, original)


def test_should_skip_empty_masks_and_remove_small_regions_when_cleaning_roi():
    masks = np.zeros((2, 15, 15), dtype=np.uint8)
    masks[1, 3:12, 3:12] = 1
    masks[1, 6:8, 6:8] = 0
    result = track.clean_roi(masks, pre_strels={}, post_strels={}, progress_bar=False)
    assert result.dtype == bool
    assert not result[0].any()
    assert result[1].sum() == 81
    assert masks[1, 6, 6] == 0


def test_should_apply_morphology_when_roi_cleaning_is_configured():
    masks = np.zeros((1, 15, 15), dtype=np.uint8)
    masks[0, 3:12, 3:12] = 1
    kernel = np.ones((3, 3), dtype=np.uint8)
    result = track.clean_roi(masks, pre_strels={cv2.MORPH_OPEN: kernel}, post_strels={cv2.MORPH_CLOSE: kernel}, fill_holes=False, use_cc=False, progress_bar=False)
    assert_array_equal(result, masks.astype(bool))


def test_should_threshold_depth_and_apply_roi_filters_when_extracting_roi():
    frames = np.zeros((1, 20, 20), dtype=np.float32)
    frames[0, 5:15, 5:15] = 100
    result = track.get_roi(frames, depth_range=(40, 200), median_kernels=[3], gaussian_kernels=[1], gradient_kernel=3, roi_strels={}, fill_holes=True, use_cc=True)
    assert result.dtype == bool
    assert result[0, 10, 10]
    assert not result[0, 0, 0]


@pytest.mark.parametrize("cropper", [track.crop_and_rotate_frames, video_util.crop_and_rotate_frames])
def test_should_crop_valid_centroids_and_skip_invalid_frames_when_rotating(cropper):
    frames = np.full((3, 12, 12), 9, dtype=np.uint16)
    features = {"centroid": np.array([[6, 6], [np.nan, 6], [-100, -100]]), "orientation": np.zeros(3)}
    result = cropper(frames, features, crop_size=(4, 4), progress_bar=False)
    assert result.shape == (3, 4, 4)
    assert result.dtype == frames.dtype
    assert_array_equal(result[0], 9)
    assert_array_equal(result[1:], 0)


def test_should_report_centroid_and_axes_when_image_has_mass():
    image = np.zeros((9, 11), dtype=np.uint8)
    image[2:7, 3:8] = 1
    result = moments.im_moment_features(image)
    assert_allclose(result["centroid"], [5, 4])
    assert_allclose(result["orientation"], 0)
    assert np.all(np.array(result["axis_length"]) > 0)


def test_should_report_nan_features_when_image_has_no_mass():
    image = np.zeros((4, 4), dtype=np.uint8)
    result = moments.im_moment_features(image)
    assert np.isnan(result["centroid"])
    assert np.isnan(result["orientation"])
    assert np.isnan(result["axis_length"]).all()


@pytest.mark.parametrize("masked", [True, False])
def test_should_select_largest_contour_and_skip_empty_frame_when_computing_features(masked):
    frames = np.zeros((2, 20, 20), dtype=np.uint16)
    frames[1, 4:12, 4:12] = 100
    mask = np.full_like(frames, 1) if masked else np.array([])
    features, output_mask = moments.get_frame_features(frames, mask=mask, progress_bar=False)
    assert np.isnan(features["centroid"][0]).all()
    assert_allclose(features["centroid"][1], [7.5, 7.5])
    assert output_mask.shape == frames.shape


@pytest.mark.parametrize("intensity", [0, 50])
@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=NameError, reason='get_frame_features does not import get_largest_cc; frame shape and empty components also need handling')
def test_should_use_connected_components_when_feature_extraction_requests_them(intensity):
    frames = np.full((1, 5, 5), intensity, dtype=np.uint16)
    result, _ = moments.get_frame_features(frames, use_cc=True, mask_threshold=1, progress_bar=False)
    if intensity:
        assert np.isfinite(result["centroid"]).all()
    else:
        assert np.isnan(result["centroid"]).all()


@pytest.mark.parametrize("points, expected", [(np.array([[0., 0, 5], [1, 0, 5], [0, 1, 5]]), [0, 0, 1, -5]), (np.array([[0., 0, 0], [1, 1, 1], [2, 2, 2]]), [np.nan] * 4)])
def test_should_fit_plane_or_report_degeneracy_when_given_three_points(points, expected):
    original = points.copy()
    result = plane._fit3(points)
    assert_allclose(result, expected, equal_nan=True)
    assert_array_equal(points, original)


def test_should_find_horizontal_plane_when_ransac_samples_valid_points(monkeypatch):
    image = np.full((3, 3), 700.0)
    choices = Mock(side_effect=[np.array([0, 0, 0]), np.array([0, 1, 3]), np.array([0, 1, 3])])
    monkeypatch.setattr(plane.np.random, "choice", choices)
    coefficients, distances = plane.fit_ransac(image, iters=3, mask=np.ones_like(image, bool))
    assert_allclose(coefficients, [0, 0, 1, -700])
    assert_array_equal(distances, 0)
    assert choices.call_count == 3


def test_should_raise_when_ransac_never_fits_a_plane():
    image = np.full((2, 2), 700.0)
    with pytest.raises(RuntimeError, match="Plane never fit") as error:
        plane.fit_ransac(image, iters=0)
    assert error.value is not None


def test_should_choose_largest_floor_region_when_ranking_plane_distances(monkeypatch):
    distances = np.full((10, 10), 100, dtype=np.float32)
    distances[1:3, 1:3] = 0
    distances[4:9, 4:9] = 0
    monkeypatch.setattr(plane, "fit_ransac", Mock(return_value=(np.zeros(4), distances.ravel())))
    result = plane.get_floor(np.ones((10, 10)), median_kernels=(1,), dilations=0)
    assert result.dtype == np.uint8
    assert result.sum() == 25
    assert result[6, 6]
    assert not result[1, 1]


@pytest.mark.parametrize("selection, indices", [(1, [1]), ([1, 2], [1, 2])])
@pytest.mark.parametrize("clean", [True, False])
def test_should_read_and_clean_masks_when_segmentation_file_exists(monkeypatch, memory_h5, selection, indices, clean):
    filename = "/session/_segmentation-tau-5/cam.hdf5"
    masks = np.ones((3, 2, 2), dtype=np.uint8)
    memory_h5.files[filename] = memory_h5.File({"/labels": masks})
    cleaner = Mock(side_effect=lambda data, **kwargs: data > 0)
    monkeypatch.setattr(io, "clean_roi", cleaner)
    result = io.load_segmentation_masks("/session/cam.avi", selection, clean_masks=clean)
    assert_array_equal(result, masks[indices] > 0)
    assert memory_h5.files[filename].closed
    assert cleaner.call_count == int(clean)


def test_should_reject_null_selection_when_loading_segmentation_masks(memory_h5):
    selection = None
    with pytest.raises(RuntimeError, match="format of frames") as error:
        io.load_segmentation_masks("/session/cam.avi", selection)
    assert error.value is not None
    memory_h5.factory.assert_not_called()


def test_should_propagate_file_error_when_segmentation_file_is_missing(memory_h5):
    failure = FileNotFoundError("segmentation missing")
    memory_h5.factory.side_effect = failure
    with pytest.raises(FileNotFoundError) as error:
        io.load_segmentation_masks("/session/cam.avi", 0)
    assert error.value is failure
