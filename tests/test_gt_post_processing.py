"""Keypoint smoothing and gap filling checked against curated ground truth.

Two kinds of tests:

* Pipeline level: SLEAP predictions run through ``registration_pipeline`` with the
  lab configuration (``tests/data/gt_pipeline_config.toml``) and are scored
  against the merged ground-truth labels.
* Stage level: a clean ground-truth trajectory (camera 223702048 labels in the
  reference frame) is corrupted in a controlled way, seeded noise or held-out
  segments, and each ``KeypointPostProcessor`` stage is scored on how much of the
  corruption it removes. Stage parameters come from the same configuration.

Thresholds sit roughly 25-40 % above the values measured when the fixture was
created (noted next to each threshold).
"""

import contextlib
import copy
import io

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import gt_support
from markovids.pcl.post_processing import KeypointPostProcessor, PostProcessingConfig, apply_final_smoothing

pytestmark = pytest.mark.gt

NOISE_MM = 3.0
LABELLED_CAMERA = "Lucid Vision Labs-HTP003S-001-223702048"


@pytest.fixture(scope="module")
def stages(gt_fixture):
    """Stage-level harness built the same way registration_pipeline builds it."""
    params = copy.deepcopy(gt_fixture.config["post_processing"])
    included = params.pop("incl_kpoints_post_processing")
    params["temporal_regularization"]["fps"] = gt_fixture.config["fps"]
    processor = KeypointPostProcessor(
        {"node_names": gt_fixture.node_names}, [tuple(bone) for bone in gt_fixture.config["skeleton"]]
    )
    camera = gt_fixture.cameras.index(LABELLED_CAMERA)
    truth = gt_support.fill_linear(gt_support.to_reference(
        gt_fixture.gt_world[camera].astype("float64"), gt_fixture.refit_R[camera], gt_fixture.refit_t[camera]
    ))
    selected = [gt_fixture.node_names.index(name) for name in included]

    def run(data, proc_order, **overrides):
        config = copy.deepcopy(params)
        for stage, values in overrides.items():
            config[stage] = {**config[stage], **values}
        confidence = np.ones(data.shape[:2] + (1,))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            points, _ = processor.process(
                data, confidence, included, PostProcessingConfig(**config), proc_order=proc_order
            )
        return points

    return type("Stages", (), dict(
        run=staticmethod(run), truth=truth, selected=selected, included=included,
        skeleton=gt_fixture.config["skeleton"], proc_order=gt_fixture.config["proc_order"],
    ))


# ----------------------------------------------------------------------------
# Pipeline level: predictions vs ground truth
# ----------------------------------------------------------------------------

def test_should_move_predictions_toward_ground_truth_when_post_processing_with_pipeline_config(gt_fixture, gt_runs):
    truth = gt_runs("gt")["merged_keypoints_raw"].astype("float64")
    saved = gt_runs("pred", post_processing=True)
    raw = saved["merged_keypoints_raw"].astype("float64")
    smooth = saved["merged_keypoints_smooth"].astype("float64")
    raw_error = gt_support.point_distance(raw, truth)
    smooth_error = gt_support.point_distance(smooth, truth)
    skeleton = gt_fixture.config["skeleton"]

    assert np.nanmean(smooth_error) <= 4.6  # measured 3.85 mm
    assert np.nanmean(smooth_error) <= 0.98 * np.nanmean(raw_error)  # measured 3.85 vs 4.04 mm
    assert np.nanpercentile(smooth_error, 90) <= 0.95 * np.nanpercentile(raw_error, 90)  # 6.88 vs 7.54 mm
    assert gt_support.rmse(smooth, truth) <= 0.96 * gt_support.rmse(raw, truth)  # 4.56 vs 4.96 mm
    assert gt_support.jitter(smooth) <= 0.15 * gt_support.jitter(raw)  # 0.26 vs 4.37 mm/frame^2
    assert gt_support.bone_length_cv(smooth, skeleton, gt_fixture.node_names) <= 0.8 * gt_support.bone_length_cv(
        raw, skeleton, gt_fixture.node_names
    )  # 0.053 vs 0.080

    raw_missing, smooth_missing = np.isnan(raw).any(axis=-1), np.isnan(smooth).any(axis=-1)
    assert not (smooth_missing & ~raw_missing).any()
    filled = raw_missing & ~smooth_missing
    assert filled.sum() >= 15  # measured 21 keypoint-frames filled by interpolation
    assert np.nanmedian(smooth_error[filled]) <= 6.0  # measured 3.66 mm


@pytest.mark.parametrize("kind", ["gt", "pred"])
def test_should_stay_close_to_stored_smoothed_output_when_rerun_with_pipeline_config(gt_fixture, gt_runs, kind):
    """The stored smoothed output came from this configuration but slightly different code.

    Re-running reproduces it to ~0.015 mm median. It is not exact: the stored
    prediction output also has every gap filled, while the current stages leave
    gaps longer than ``length_threshold`` missing.
    """
    saved = gt_runs(kind, transforms="stored", post_processing=True)["merged_keypoints_smooth"]
    stored = getattr(gt_fixture, f"{kind}_merged_smooth")
    difference = np.abs(saved - stored)
    assert np.nanmedian(difference) <= 0.05  # measured 0.016 (gt) / 0.015 (pred) mm
    assert np.nanpercentile(difference, 99) <= 1.0  # measured 0.37 / 0.72 mm
    assert np.nanmax(difference) <= 2.5  # measured 1.30 / 1.77 mm
    raw_missing = np.isnan(getattr(gt_fixture, f"{kind}_merged_raw"))
    assert not (np.isnan(saved) & ~raw_missing).any()


# ----------------------------------------------------------------------------
# Stage level: controlled corruption of ground truth
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("stage, max_error, max_jitter, max_bone_cv", [
    # Ratios to the noisy input (RMSE, jitter, bone-length CV).
    ("temporal", 0.55, 0.03, 0.60),         # measured 0.44 / 0.013 / 0.48
    ("bone", 1.00, 1.00, 0.97),             # measured 0.98 / 0.98 / 0.91 
    ("config_order", 0.55, 0.03, 0.55),     # measured 0.44 / 0.017 / 0.42
    ("final_smooth", 0.60, 0.15, 0.72),     # measured 0.48 / 0.11 / 0.58
])
def test_should_remove_noise_when_smoothing_corrupted_ground_truth(stages, stage, max_error, max_jitter, max_bone_cv):
    noisy = stages.truth + np.random.default_rng(0).normal(0, NOISE_MM, stages.truth.shape)
    order = stages.proc_order if stage == "config_order" else [stage]
    points = stages.run(noisy, order)
    truth, noisy = stages.truth[:, stages.selected], noisy[:, stages.selected]

    assert not np.isnan(points).any()
    assert gt_support.rmse(points, truth) <= max_error * gt_support.rmse(noisy, truth)
    assert gt_support.jitter(points) <= max_jitter * gt_support.jitter(noisy)
    cv = gt_support.bone_length_cv(points, stages.skeleton, stages.included)
    assert cv <= max_bone_cv * gt_support.bone_length_cv(noisy, stages.skeleton, stages.included)


def held_out_segments(n_frames, n_keypoints, length):
    """One segment per keypoint, staggered across the clip."""
    held = np.zeros((n_frames, n_keypoints), dtype=bool)
    for keypoint in range(n_keypoints):
        start = 30 + (keypoint * 9) % 130
        held[start:start + length, keypoint] = True
    return held


@pytest.mark.parametrize("length", [3, 8])
@pytest.mark.parametrize("stage, max_median, max_p90", [
    ("interpolate", 3.0, 7.0),      # measured median 1.52 / 2.16, p90 3.92 / 5.29 mm
    ("config_order", 4.0, 8.5),     # measured median 2.55 / 3.09, p90 6.09 / 6.84 mm
])
def test_should_fill_short_gaps_close_to_ground_truth_when_segments_are_held_out(stages, stage, max_median, max_p90, length):
    held = held_out_segments(*stages.truth.shape[:2], length)
    data = stages.truth.copy()
    data[held] = np.nan
    order = stages.proc_order if stage == "config_order" else [stage]
    points = stages.run(data, order)
    held, truth = held[:, stages.selected], stages.truth[:, stages.selected]
    errors = gt_support.point_distance(points, truth)

    assert not np.isnan(points).any()
    assert np.median(errors[held]) <= max_median
    assert np.percentile(errors[held], 90) <= max_p90
    if stage == "interpolate":
        # Observed frames only pass through alignment and its inverse.
        assert_allclose(points[~held], truth[~held], rtol=0, atol=1e-9)


@pytest.mark.parametrize("stage", ["temporal", "interpolate", "config_order"])
def test_should_leave_long_gaps_missing_when_they_exceed_fill_limits(stages, stage):
    # The configuration disables temporal gap filling (max_gap_fill = 0) and
    # interpolates gaps of at most length_threshold = 10 frames.
    held = held_out_segments(*stages.truth.shape[:2], 3 if stage == "temporal" else 20)
    data = stages.truth.copy()
    data[held] = np.nan
    order = stages.proc_order if stage == "config_order" else [stage]
    points = stages.run(data, order)
    held = held[:, stages.selected]
    assert_array_equal(np.isnan(points).any(axis=-1), held)


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Final smoothing spreads NaNs into valid observations")
def test_should_preserve_observations_when_final_smoothing_meets_a_missing_ground_truth_frame(gt_fixture):
    raw = gt_fixture.gt_merged_raw.astype("float64")
    params = gt_fixture.config["post_processing"]["post_align_sgolay"]
    smoothed = apply_final_smoothing(raw, gt_fixture.node_names, **params)
    # measured: 1 missing keypoint-frame becomes 13 with the configured 11-frame window
    assert_array_equal(np.isnan(smoothed), np.isnan(raw))
