"""Small synthetic sequences through real post-processing stages."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.pcl.post_processing import KeypointPostProcessor, PostProcessingConfig


@pytest.mark.parametrize("gap_length", [0, 2, pytest.param(12, marks=[pytest.mark.known_bug, pytest.mark.xfail(strict=True, raises=AssertionError, reason="Final smoothing expands the long missing-data gap")])])
def test_default_pipeline_preserves_rigid_motion_and_handles_missing_observations(gap_length):
    names = ["tail_tip", "unused", "body", "snout", "back_top", "back_bottom"]
    selected = ["body", "snout", "tail_tip", "back_top", "back_bottom"]
    # An oblique body axis exercises alignment and inverse alignment; linear
    # translation supplies an independent, known ground truth for missing data.
    pose = np.array([[-8., -6., 3.], [99., 99., 99.],
                     [0., 0., 5.], [8., 6., 7.], [8., 6., 5.], [-8., -6., 5.]])
    translation = np.arange(120)[:, None, None] * np.array([.05, -.02, .01])
    truth = pose[None] + translation
    data = truth.copy()
    confidence = np.broadcast_to([.7, .95, .9, .8, .9, .9], (120, 6)).copy()
    camera_confidence = np.stack([confidence * .5, confidence], axis=-1)
    gap = slice(50, 50 + gap_length)
    data[gap, 3] = np.nan
    camera_confidence[gap, 3] = .1
    original_data, original_confidence = data.copy(), camera_confidence.copy()
    config = PostProcessingConfig(
        temporal_regularization={"fps": 1, "max_gap_fill": 1},
        bone_length_regularization={},
        align={"exclude_from_center": ["snout", "tail_tip", "back_top", "back_bottom"]},
        align_compute={"smooth_alignment": False},
        pca={}, post_align_hampel={}, post_align_imputed_smoothing={},
        post_align_sgolay={"window_length": 5, "polyorder": 3},
        interpolation={"length_threshold": 3},
    )
    processor = KeypointPostProcessor(
        {"node_names": names}, [("body", "snout"), ("body", "tail_tip")]
    )

    points, scores = processor.process(data, camera_confidence, selected, config)

    expected = truth[:, [2, 3, 0, 4, 5]]
    expected_nan = np.zeros(points.shape, dtype=bool)
    if gap_length > 3:
        expected_nan[gap, 1] = True
    assert_array_equal(np.isnan(points), expected_nan)
    # Nearest padding in the final filter slightly flattens sequence endpoints.
    assert_allclose(points[4:-4][~expected_nan[4:-4]],
                    expected[4:-4][~expected_nan[4:-4]], atol=.05)
    assert scores.shape == (120, 5)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert_allclose(scores[:, [0, 2]], [[.9, .7]] * 120, atol=1e-6)
    if gap_length:
        assert (scores[gap, 1] > 0).all()
        assert (scores[gap, 1] < .8).all()
    else:
        assert_allclose(scores[:, 1], .8, atol=1e-6)
    assert_array_equal(data, original_data)
    assert_array_equal(camera_confidence, original_confidence)
    assert not np.shares_memory(points, data)
    assert not np.shares_memory(scores, camera_confidence)
