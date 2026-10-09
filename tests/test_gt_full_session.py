"""Opt-in checks against the full example sessions (~220 MB, not in git).

Run with ``pytest --run-gt-full``. The sessions are read from
``$MARKOVIDS_GT_DATA`` / ``$MARKOVIDS_PRED_DATA``, defaulting to
``<repo>/_tmp/example_session_gt`` and ``<repo>/_tmp/example_session``; tests skip
when they are absent. The pipeline runs without stubs (~11 s and ~1.1 GB per run).
"""

import importlib.util
import json

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_allclose, assert_array_equal

import gt_support

pytestmark = pytest.mark.gt_full


def load_generator():
    path = gt_support.DATA_DIR / "make_gt_fixture.py"
    spec = importlib.util.spec_from_file_location("make_gt_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_should_match_committed_fixture_when_regenerated_from_full_sessions(gt_fixture, gt_session_path, pred_session_path):
    arrays, manifest = load_generator().build(gt_session_path, pred_session_path, gt_support.FIXTURE_CONFIG)
    with np.load(gt_support.FIXTURE_NPZ, allow_pickle=False) as committed:
        assert sorted(committed.files) == sorted(arrays)
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.floating):
                assert_allclose(committed[key], value, rtol=0, atol=1e-4, equal_nan=True, err_msg=key)
            else:
                assert_array_equal(committed[key], value, err_msg=key)
    assert json.loads(gt_support.FIXTURE_JSON.read_text()) == json.loads(json.dumps(manifest))


@pytest.mark.parametrize("kind", ["gt", "pred"])
def test_should_reproduce_stored_raw_merge_when_full_pipeline_runs_unstubbed(
    gt_fixture, gt_session_path, pred_session_path, tmp_path, kind
):
    session = gt_session_path if kind == "gt" else pred_session_path
    gt_support.write_transforms(gt_fixture, tmp_path, "stored", gt_fixture.stored_R, gt_fixture.stored_t)
    config = gt_support.write_config(gt_fixture, tmp_path / "config.toml", proc_order=[])
    np.random.seed(0)
    saved = gt_support.run_pipeline(gt_fixture, session, tmp_path / "out", config, "stored", transforms_dir=tmp_path)
    stored = getattr(gt_fixture, f"{kind}_merged_raw")
    assert_array_equal(np.isnan(saved["merged_keypoints_raw"]), np.isnan(stored))
    assert np.nanmax(np.abs(saved["merged_keypoints_raw"] - stored)) <= 1e-3
    assert saved["roi"].shape == (gt_fixture.height, gt_fixture.width)


def test_should_fit_identical_floor_when_ransac_is_seeded(gt_session_path, gt_fixture):
    from markovids.depth.plane import get_floor

    image = tifffile.imread(gt_session_path / "_bground" / f"{gt_fixture.reference_camera}.tiff").astype("float")
    np.random.seed(0)
    first = get_floor(image, dilations=0)
    np.random.seed(0)
    second = get_floor(image, dilations=0)
    assert_array_equal(first, second)
    assert first.mean() > 0.3  # the arena floor dominates the frame
