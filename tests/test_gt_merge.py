"""Multi-camera registration and merging checked against curated ground truth.

Every test runs the real ``registration_pipeline`` (floor fitting stubbed) on
mini sessions built from ``tests/data/gt_session_v1.npz``. "Truth" in 3D is the
ground-truth labels merged with the re-fitted transforms; the independent check
is reprojection into each camera against that camera's 2D labels.

Thresholds sit roughly 25-40 % above the values measured when the fixture was
created (noted next to each threshold).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import gt_support
from markovids.pcl import registration

pytestmark = pytest.mark.gt

MERGE_METHODS = ["mixed", "weighted", "max"]


def merged(gt_runs, kind, **kwargs):
    return gt_runs(kind, **kwargs)["merged_keypoints_raw"].astype("float64")


@pytest.fixture(scope="module")
def truth(gt_runs):
    return merged(gt_runs, "gt")


@pytest.mark.parametrize("kind", ["gt", "pred"])
def test_should_reproduce_stored_raw_merge_when_rerun_with_stored_transforms(gt_fixture, gt_runs, kind):
    saved = gt_runs(kind, transforms="stored")
    stored = getattr(gt_fixture, f"{kind}_merged_raw")
    assert_array_equal(np.isnan(saved["merged_keypoints_raw"]), np.isnan(stored))
    assert np.nanmax(np.abs(saved["merged_keypoints_raw"] - stored)) <= 1e-3  # measured 3.1e-5 mm
    assert_array_equal(saved["merged_keypoints_confidence"], getattr(gt_fixture, f"{kind}_merged_conf"))
    assert_allclose(saved["device_timestamp_ref"], gt_fixture.timestamps)


def test_should_match_world_projection_oracle_when_only_reference_camera_is_loaded(gt_fixture, gt_runs):
    ref = gt_fixture.ref_index
    saved = gt_runs("gt", cameras=[gt_fixture.reference_camera])
    expected = gt_support.keypoints_to_world(
        gt_fixture.gt_kp3d[ref], gt_fixture.bground[ref], gt_fixture.K[ref], gt_fixture.dist[ref]
    )
    assert_array_equal(np.isnan(saved["merged_keypoints_raw"]), np.isnan(expected))
    assert np.nanmax(np.abs(saved["merged_keypoints_raw"] - expected)) <= 1e-3
    assert_allclose(gt_fixture.gt_world[ref], expected, atol=1e-3, equal_nan=True)


@pytest.mark.parametrize("camera, max_median, max_p90, min_pck", [
    (0, 7.5, 12.5, 0.30),   # measured 5.81 / 9.60 px, PCK@5 0.39
    (1, 6.0, 14.0, 0.40),   # measured 4.68 / 10.63 px, PCK@5 0.53
    (2, 4.0, 7.5, 0.75),    # reference camera; measured 2.96 / 5.85 px, PCK@5 0.85
])
def test_should_reproject_onto_labels_when_merging_ground_truth_with_refit_transforms(
    gt_fixture, truth, camera, max_median, max_p90, min_pck
):
    errors = gt_support.summarize(gt_support.reprojection_error(truth, gt_fixture, camera))
    assert errors.n >= 0.99 * np.isfinite(gt_fixture.gt_kp2d[camera][..., 0]).sum()
    assert errors.median <= max_median
    assert errors.p90 <= max_p90
    assert errors.pck >= min_pck


def test_should_reduce_reprojection_error_when_refit_transforms_replace_stored_ones(gt_fixture, gt_runs, truth):
    camera = gt_fixture.cameras.index("Lucid Vision Labs-HTP003S-001-223702266")
    stored = merged(gt_runs, "gt", transforms="stored")
    stored_error = gt_support.summarize(gt_support.reprojection_error(stored, gt_fixture, camera, "stored"))
    refit_error = gt_support.summarize(gt_support.reprojection_error(truth, gt_fixture, camera))
    assert stored_error.median > 25  # stored transform is misregistered by ~25 mm (measured 37 px)
    assert refit_error.median <= stored_error.median / 4  # measured 4.68 vs 37.08 px


@pytest.mark.parametrize("pair, max_median, max_p90", [
    ((0, 2), 8.5, 15.0),    # measured 6.60 / 11.95 mm
    ((1, 2), 7.0, 12.5),    # measured 5.27 / 9.59 mm
    ((0, 1), 13.0, 25.0),   # measured 10.03 / 19.81 mm
])
def test_should_agree_across_cameras_when_labels_are_mapped_with_refit_transforms(gt_fixture, pair, max_median, max_p90):
    a, b = (
        gt_support.to_reference(gt_fixture.gt_world[i].astype("float64"), gt_fixture.refit_R[i], gt_fixture.refit_t[i])
        for i in pair
    )
    distances = gt_support.point_distance(a, b)
    assert np.isfinite(distances).sum() >= 400
    assert np.nanmedian(distances) <= max_median
    assert np.nanpercentile(distances, 90) <= max_p90


@pytest.mark.parametrize("estimator", ["estimate_transform", "bundle_adjust_rigid_fixed_structure"])
def test_should_recover_camera_transforms_when_fitting_ground_truth_correspondences(gt_fixture, estimator):
    ref = gt_fixture.ref_index
    keypoints = [gt_fixture.node_names.index(name) for name in gt_fixture.refit_transforms["fit_keypoints"]]
    world = gt_fixture.gt_world.astype("float64")[:, :, keypoints].reshape(len(gt_fixture.cameras), -1, 3)
    for camera in range(len(gt_fixture.cameras)):
        if camera == ref:
            continue
        both = ~(np.isnan(world[ref]).any(axis=1) | np.isnan(world[camera]).any(axis=1))
        target, source = world[ref][both], world[camera][both]
        result = getattr(registration, estimator)(target, source, source)
        rotation, translation = result["B_to_A"]["R"], result["B_to_A"]["t"]
        residual = gt_support.point_distance(gt_support.to_reference(source, rotation, translation), target)
        assert np.median(residual) <= 7.0  # measured 5.73 / 5.20 mm (estimate_transform)
        assert np.percentile(residual, 90) <= 12.0  # measured 9.52 / 9.36 mm
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation.T @ gt_fixture.refit_R[camera]) - 1) / 2, -1, 1)))
        if estimator == "estimate_transform":
            # The fixture's refit transforms were produced by this estimator.
            assert angle <= 1e-3
            assert_allclose(translation, gt_fixture.refit_t[camera], atol=1e-2)
        else:
            # Huber-robust fit; measured 1.00 / 0.85 deg and 6.3 / 5.4 mm from the refit.
            assert angle <= 2.0
            assert np.linalg.norm(translation - gt_fixture.refit_t[camera]) <= 10.0


@pytest.mark.parametrize("method, max_mean, min_coverage", [
    ("mixed", 5.0, 0.96),      # measured 4.04 mm, coverage 0.975
    ("weighted", 5.3, 0.96),   # measured 4.27 mm, coverage 0.975
    ("max", 5.8, 1.0),         # measured 4.63 mm, coverage 1.0
])
def test_should_track_ground_truth_when_merging_predictions(gt_runs, truth, method, max_mean, min_coverage):
    prediction = merged(gt_runs, "pred", merge_method=method)
    distances = gt_support.point_distance(prediction, truth)
    assert np.nanmean(distances) <= max_mean
    assert np.nanpercentile(distances, 90) <= 9.5  # measured 7.5-7.9 mm
    assert (~np.isnan(prediction).any(axis=-1)).mean() >= min_coverage


def test_should_beat_every_single_camera_when_merging_predictions(gt_fixture, gt_runs, truth):
    prediction = merged(gt_runs, "pred")
    merged_median = np.nanmedian(gt_support.point_distance(prediction, truth))
    single_medians = []
    for camera in range(len(gt_fixture.cameras)):
        world = gt_support.keypoints_to_world(
            gt_fixture.pred_kp3d[camera], gt_fixture.bground[camera], gt_fixture.K[camera], gt_fixture.dist[camera]
        )
        world = gt_support.to_reference(world, gt_fixture.refit_R[camera], gt_fixture.refit_t[camera])
        single_medians.append(np.nanmedian(gt_support.point_distance(world, truth)))
    # measured merged 3.41 mm vs single cameras 5.29 / 34.8 / 5.62 mm
    assert merged_median <= 0.8 * min(single_medians)


@pytest.fixture(scope="module")
def misregistered(gt_fixture, gt_runs):
    """Refit transforms with the sparse side camera shifted by 50 mm."""
    translations = gt_fixture.refit_t.copy()
    translations[gt_fixture.cameras.index("Lucid Vision Labs-HTP003S-001-223702266")] += [50.0, 0.0, 0.0]
    gt_runs.write_transforms("misregistered", gt_fixture.refit_R, translations)
    return "misregistered"


def test_should_reject_misregistered_camera_when_merge_method_is_mixed(gt_fixture, gt_runs, truth, misregistered):
    ref = gt_fixture.ref_index
    results = {}
    for method in MERGE_METHODS:
        points = merged(gt_runs, "gt", transforms=misregistered, merge_method=method)
        results[method] = SimpleNamespace(
            mean_shift=np.nanmean(gt_support.point_distance(points, truth)),
            ref_p90=gt_support.summarize(gt_support.reprojection_error(points, gt_fixture, ref)).p90,
        )
    assert results["mixed"].mean_shift <= 0.5  # measured 0.23 mm: the 15 mm gate drops the bad camera
    assert results["mixed"].ref_p90 <= 7.5  # measured 5.80 px
    assert results["weighted"].ref_p90 >= 15.0  # measured 22.64 px: averaging pulls toward the bad camera
    assert results["weighted"].mean_shift >= 4 * results["mixed"].mean_shift  # measured 2.23 vs 0.23 mm


def test_should_tolerate_misregistered_camera_when_merging_predictions(gt_runs, truth, misregistered):
    clean = np.nanmean(gt_support.point_distance(merged(gt_runs, "pred"), truth))
    shifted = np.nanmean(gt_support.point_distance(merged(gt_runs, "pred", transforms=misregistered), truth))
    assert shifted - clean <= 0.8  # measured +0.39 mm (weighted: +1.18 mm)
