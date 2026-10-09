"""Helpers for tests that use the committed ground-truth fixture.

The fixture (``tests/data/gt_session_v1.npz`` + ``.json``) is produced by
``tests/data/make_gt_fixture.py``. It holds one 200-frame, 3-camera clip twice:
``gt_*`` arrays are curated 2D labels (confidence 1 = labelled, 0 = absent)
lifted to 3D, and ``pred_*`` arrays are SLEAP predictions for the same frames.
"""

import contextlib
import io
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import tifffile
import toml

DATA_DIR = Path(__file__).resolve().parent / "data"
FIXTURE_NPZ = DATA_DIR / "gt_session_v1.npz"
FIXTURE_JSON = DATA_DIR / "gt_session_v1.json"
FIXTURE_CONFIG = DATA_DIR / "gt_pipeline_config.toml"
KPOINTS_DIR = "_kpoints_v1_3d"


def load_gt_fixture():
    """Load the fixture arrays and manifest into one namespace."""
    with np.load(FIXTURE_NPZ, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    manifest = json.loads(FIXTURE_JSON.read_text())
    fixture = SimpleNamespace(**arrays, **manifest)
    fixture.ref_index = fixture.cameras.index(fixture.reference_camera)
    fixture.config = toml.load(FIXTURE_CONFIG)
    return fixture


def materialize_session(fixture, root, kind):
    """Write a minimal session directory the real registration_pipeline can read.

    Camera names keep their spaces so the pipeline's path handling is exercised.
    """
    root = Path(root)
    (root / "_bground").mkdir(parents=True)
    keypoints_dir = root / "_proc" / KPOINTS_DIR
    keypoints_dir.mkdir(parents=True)
    metadata = {"camera_metadata": {c: {"Width": fixture.width, "Height": fixture.height} for c in fixture.cameras}}
    (root / "metadata.toml").write_text(toml.dumps(metadata))
    kp3d = getattr(fixture, f"{kind}_kp3d")
    for i, camera in enumerate(fixture.cameras):
        tifffile.imwrite(root / "_bground" / f"{camera}.tiff", fixture.bground[i])
        joblib.dump(kp3d[i].copy(), keypoints_dir / f"{camera}.pkl.gz")
        (keypoints_dir / f"{camera}.toml").write_text(toml.dumps({"node_names": fixture.node_names}))
    lines = ["device_timestamp_ref"] + [repr(float(value)) for value in fixture.timestamps]
    (root / "_proc" / "timestamps.txt").write_text("\n".join(lines) + "\n")
    for transforms in ("stored", "refit"):
        write_transforms(fixture, root, transforms, getattr(fixture, f"{transforms}_R"), getattr(fixture, f"{transforms}_t"))
    return root


def write_transforms(fixture, root, name, rotations, translations):
    """Write ``transforms_<name>.toml`` for ``run_pipeline(..., transforms=name)``."""
    table = {
        str((camera, fixture.reference_camera)): [np.asarray(rotations[i]).tolist(), np.asarray(translations[i]).tolist()]
        for i, camera in enumerate(fixture.cameras)
    }
    path = Path(root) / f"transforms_{name}.toml"
    path.write_text(toml.dumps(table))
    return path


def write_config(fixture, path, **overrides):
    """Write the committed pipeline config with top-level keys replaced."""
    config = toml.load(FIXTURE_CONFIG)
    config.update(overrides)
    Path(path).write_text(toml.dumps(config))
    return path


def stub_floor_fit(monkeypatch, pipeline):
    """Replace RANSAC floor fitting, which only feeds the roi/roi_merged outputs.

    ``depth.plane.get_floor`` is unseeded and takes ~2.4 s per call (six calls per
    run). Keypoint outputs do not depend on it: the z reference is the full
    background image. ``cv2.erode`` is wrapped to return a boolean mask so that
    ``bground[mask]`` is a boolean selection rather than a 480x640x640 integer
    gather (see ISSUES.md).
    """
    def central_floor(image, *args, **kwargs):
        roi = np.zeros(image.shape, np.uint8)
        roi[100:-100, 100:-100] = 1
        return roi

    real_erode = pipeline.cv2.erode
    monkeypatch.setattr(pipeline.depth.plane, "get_floor", central_floor)
    monkeypatch.setattr(pipeline.cv2, "erode", lambda image, kernel: real_erode(image, kernel).astype(bool))


def run_pipeline(fixture, session_root, output, config_path, transforms, merge_method="mixed",
                 cameras=None, transforms_dir=None):
    """Run registration_pipeline quietly and return the saved HDF5 datasets.

    ``transforms`` names ``transforms_<name>.toml`` in ``transforms_dir``
    (default: the session root). ``cameras`` restricts the calibration
    dictionaries, and therefore the cameras the pipeline loads; the default
    uses all fixture cameras.
    """
    import h5py
    from markovids.pcl import pipeline

    cameras = fixture.cameras if cameras is None else cameras
    intrinsics = {c: fixture.K[fixture.cameras.index(c)] for c in cameras}
    distortion = {c: fixture.dist[fixture.cameras.index(c)] for c in cameras}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipeline.registration_pipeline(
            str(config_path), str(session_root), kpoints_save_dir=KPOINTS_DIR,
            intrinsics_matrix=intrinsics, distortion_coefficients=distortion,
            alt_save_dir=str(output),
            transforms_path=str(Path(transforms_dir or session_root) / f"transforms_{transforms}.toml"),
            merge_method=merge_method,
        )
    with h5py.File(Path(output) / "merged_keypoints.h5", "r") as handle:
        return {key: handle[key][()] for key in handle}


# ----------------------------------------------------------------------------
# Geometry and metrics
# ----------------------------------------------------------------------------

def keypoints_to_world(kp3d, bground, intrinsics, distortion):
    """Mirror registration_pipeline: height above background -> camera-frame mm.

    Samples the undistorted background (depth units / 4 = mm) at each keypoint
    pixel, converts height to distance from the camera and back-projects with
    the pinhole model (``project_world_coordinates`` with ``z_scale=1``).
    Used by the fixture generator and as a test oracle; ``test_gt_merge.py``
    checks it against the pipeline's own merge output.
    """
    import cv2
    from markovids.pcl import io as pcl_io

    background = cv2.undistort(bground, intrinsics, distortion).T / 4.0
    points = kp3d.astype("float64").copy()
    xy = points[..., :2]
    valid = ~np.isnan(xy).any(axis=-1)
    width, height = bground.shape[1], bground.shape[0]
    xy_int = np.clip(np.nan_to_num(xy).astype(np.int32), 0, [width - 1, height - 1])
    background_values = np.full(points.shape[:2], np.nan)
    background_values[valid] = background[xy_int[valid, 0], xy_int[valid, 1]]
    points[..., 2] = -points[..., 2] + background_values
    world = pcl_io.project_world_coordinates(
        points[..., :3].reshape(-1, 3), floor_distance=None, z_scale=1.0,
        cx=intrinsics[0, 2], cy=intrinsics[1, 2], fx=intrinsics[0, 0], fy=intrinsics[1, 1],
    )
    return world.reshape(points.shape[:2] + (3,))


def to_reference(points, rotation, translation):
    """Camera-frame points (..., 3) -> reference frame."""
    return points @ rotation.T + translation


def reproject(points, rotation, translation, intrinsics):
    """Reference-frame mm (..., 3) -> pixel coordinates (..., 2) of one camera."""
    local = (points - translation) @ rotation
    u = local[..., 0] * intrinsics[0, 0] / local[..., 2] + intrinsics[0, 2]
    v = local[..., 1] * intrinsics[1, 1] / local[..., 2] + intrinsics[1, 2]
    return np.stack([u, v], axis=-1)


def reprojection_error(points, fixture, camera_index, transforms="refit"):
    """Pixel distance to the camera's ground-truth labels; NaN where unlabelled."""
    rotation = getattr(fixture, f"{transforms}_R")[camera_index]
    translation = getattr(fixture, f"{transforms}_t")[camera_index]
    projected = reproject(points, rotation, translation, fixture.K[camera_index])
    return np.linalg.norm(projected - fixture.gt_kp2d[camera_index][..., :2], axis=-1)


def summarize(errors, pck_threshold=5.0):
    finite = errors[np.isfinite(errors)]
    return SimpleNamespace(
        n=finite.size,
        median=float(np.median(finite)),
        p90=float(np.percentile(finite, 90)),
        pck=float(np.mean(finite < pck_threshold)),
    )


def point_distance(a, b):
    """Per keypoint-frame Euclidean distance; NaN where either side is missing."""
    return np.linalg.norm(a - b, axis=-1)


def rmse(estimate, truth, trim=5):
    """Root mean squared 3D error, excluding ``trim`` frames at both ends."""
    squared = np.sum((estimate[trim:-trim] - truth[trim:-trim]) ** 2, axis=-1)
    return float(np.sqrt(np.nanmean(squared)))


def jitter(points):
    """RMS second temporal difference (mm / frame^2)."""
    return float(np.sqrt(np.nanmean(np.sum(np.diff(points, 2, axis=0) ** 2, axis=-1))))


def bone_length_cv(points, skeleton, node_names):
    """Mean coefficient of variation of bone lengths over time."""
    values = []
    for start, end, *_ in skeleton:
        lengths = np.linalg.norm(points[:, node_names.index(start)] - points[:, node_names.index(end)], axis=-1)
        values.append(np.nanstd(lengths) / np.nanmean(lengths))
    return float(np.mean(values))


def fill_linear(points):
    """Linearly interpolate NaNs along time for each keypoint coordinate."""
    filled = points.astype("float64").copy()
    frames = np.arange(len(filled))
    for trajectory in filled.reshape(len(filled), -1).T:
        missing = np.isnan(trajectory)
        if missing.any():
            trajectory[missing] = np.interp(frames[missing], frames[~missing], trajectory[~missing])
    return filled.reshape(points.shape)


def require_fixture():
    if not (FIXTURE_NPZ.exists() and FIXTURE_JSON.exists() and FIXTURE_CONFIG.exists()):
        pytest.skip("Ground-truth fixture missing; regenerate with tests/data/make_gt_fixture.py")
