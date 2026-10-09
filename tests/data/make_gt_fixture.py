"""Regenerate the committed ground-truth fixture from the full example sessions.

The full sessions (~80 MB labels-only and ~140 MB predictions) live outside git,
by default under ``<repo>/_tmp``. This script extracts the small subset that the
``gt`` test tier needs and writes:

* ``tests/data/gt_session_v1.npz``: numeric arrays only (load with
  ``allow_pickle=False``).
* ``tests/data/gt_session_v1.json``: manifest with camera/keypoint metadata,
  derived transforms and sha256 checksums of every source file.
* ``tests/data/gt_pipeline_config.toml``: the post-processing configuration,
  copied from ``--config`` when given.

Usage::

    python tests/data/make_gt_fixture.py \
        [--gt-session PATH] [--pred-session PATH] [--config PATH]

Session paths default to ``$MARKOVIDS_GT_DATA`` / ``$MARKOVIDS_PRED_DATA`` and
then to ``<repo>/_tmp/example_session_gt`` / ``<repo>/_tmp/example_session``.
``build()`` performs no subprocess or network calls, so the drift test in
``tests/test_gt_full_session.py`` can call it directly.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import tifffile
import toml

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from markovids.pcl import registration  # noqa: E402
from gt_support import keypoints_to_world  # noqa: E402

FIXTURE_VERSION = 1
FIXTURE_NPZ = DATA_DIR / f"gt_session_v{FIXTURE_VERSION}.npz"
FIXTURE_JSON = DATA_DIR / f"gt_session_v{FIXTURE_VERSION}.json"
FIXTURE_CONFIG = DATA_DIR / "gt_pipeline_config.toml"
KPOINTS_DIR = "_kpoints_v1_3d"
# The labelled clip corresponds to these rows of the synced _proc/timestamps.txt
# (SLEAP frame_idx of the ground-truth labels).
FRAME_WINDOW = (3900, 4100)
_NP_FLOAT = re.compile(r"np\.float64\((.*)\)")


def default_session(env_var, name):
    return Path(os.environ.get(env_var) or REPO_ROOT / "_tmp" / name)


def _to_float(value):
    if isinstance(value, list):
        return [_to_float(item) for item in value]
    if isinstance(value, str):
        return float(_NP_FLOAT.sub(r"\1", value))
    return float(value)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration(session, cameras):
    """Intrinsics and distortion from sync_metadata.toml (stringified np.float64)."""
    sync = toml.load(session / "_proc" / "sync_metadata.toml")
    intrinsics = np.stack([np.array(_to_float(sync["intrinsics_matrix"][c])) for c in cameras])
    distortion = np.stack([np.array(_to_float(sync["distortion_coeffs"][c])) for c in cameras])
    return intrinsics, distortion


def load_stored_transforms(session, cameras, reference):
    merged = toml.load(session / "_proc" / KPOINTS_DIR / "merged_keypoints.toml")
    transforms = {ast.literal_eval(k): v for k, v in merged["transforms"].items()}
    rotations = np.stack([np.array(transforms[(c, reference)][0], float) for c in cameras])
    translations = np.stack([np.array(transforms[(c, reference)][1], float) for c in cameras])
    return rotations, translations, merged


def load_sleap_skeleton(slp_path):
    """Return the labelling skeleton edges as (source, target) node names."""
    with h5py.File(slp_path, "r") as handle:
        metadata = json.loads(handle["metadata"].attrs["json"])
    names = [node["name"] for node in metadata["nodes"]]
    return [
        [names[link["source"]], names[link["target"]]]
        for link in metadata["skeletons"][0]["links"]
        if "edge_insert_idx" in link
    ]


def refit_transforms(world, cameras, reference, node_names, fit_keypoints):
    """Rigid camera->reference transforms from ground-truth correspondences.

    Each non-reference camera is fitted pairwise against the reference camera
    using every frame where both cameras have a labelled ``fit_keypoints`` point.
    """
    ref = cameras.index(reference)
    keypoint_idx = [node_names.index(name) for name in fit_keypoints]
    rotations = np.tile(np.eye(3), (len(cameras), 1, 1))
    translations = np.zeros((len(cameras), 3))
    details = {}
    for i, camera in enumerate(cameras):
        if i == ref:
            continue
        target = world[ref][:, keypoint_idx].reshape(-1, 3)
        source = world[i][:, keypoint_idx].reshape(-1, 3)
        both = ~(np.isnan(target).any(axis=1) | np.isnan(source).any(axis=1))
        result = registration.estimate_transform(target[both], source[both], source[both])
        rotations[i], translations[i] = result["B_to_A"]["R"], result["B_to_A"]["t"]
        residual = np.linalg.norm(source[both] @ rotations[i].T + translations[i] - target[both], axis=1)
        details[camera] = {
            "correspondences": int(both.sum()),
            "median_residual_mm": round(float(np.median(residual)), 3),
            "p90_residual_mm": round(float(np.percentile(residual, 90)), 3),
        }
    return rotations, translations, details


def _load_session_keypoints(session, cameras):
    kp2d = np.stack([
        joblib.load(session / "_proc" / "_kpoints_v1_2d" / f"{c}.pkl.gz") for c in cameras
    ]).astype("float32")
    kp3d = np.stack([
        joblib.load(session / "_proc" / KPOINTS_DIR / f"{c}.pkl.gz") for c in cameras
    ]).astype("float32")
    with h5py.File(session / "_proc" / KPOINTS_DIR / "merged_keypoints.h5", "r") as handle:
        merged = {
            "merged_raw": handle["merged_keypoints_raw"][()],
            "merged_smooth": handle["merged_keypoints_smooth"][()],
            "merged_conf": handle["merged_keypoints_confidence"][()],
            "post_conf": handle["post_processing_confidence"][()],
        }
    return kp2d, kp3d, merged


def _source_files(session, cameras):
    files = ["metadata.toml", "_proc/sync_metadata.toml", "_proc/timestamps.txt",
             f"_proc/{KPOINTS_DIR}/merged_keypoints.h5", f"_proc/{KPOINTS_DIR}/merged_keypoints.toml"]
    for camera in cameras:
        files += [f"_bground/{camera}.tiff", f"_proc/_kpoints_v1_2d/{camera}.pkl.gz",
                  f"_proc/{KPOINTS_DIR}/{camera}.pkl.gz", f"_proc/{KPOINTS_DIR}/{camera}.toml",
                  f"_proc/_keypoints_v1/{camera}.slp"]
    return {name: _sha256(session / name) for name in files}


def build(gt_session, pred_session, config_path=FIXTURE_CONFIG):
    """Return ``(arrays, manifest)`` for the fixture without writing anything."""
    gt_session, pred_session = Path(gt_session), Path(pred_session)
    config = toml.load(config_path)
    reference = config["reference_camera"]
    merged_meta = toml.load(gt_session / "_proc" / KPOINTS_DIR / "merged_keypoints.toml")
    cameras = list(merged_meta["cameras"])
    if merged_meta["reference_camera"] != reference:
        raise ValueError(f"Config reference {reference} != stored {merged_meta['reference_camera']}")
    node_names = list(merged_meta["kpoints"]["node_names"])
    metadata = toml.load(gt_session / "metadata.toml")["camera_metadata"]
    width, height = metadata[reference]["Width"], metadata[reference]["Height"]

    intrinsics, distortion = load_calibration(gt_session, cameras)
    stored_R, stored_t, _ = load_stored_transforms(gt_session, cameras, reference)
    bground = np.stack([tifffile.imread(gt_session / "_bground" / f"{c}.tiff") for c in cameras])
    for camera in cameras:
        pred_bground = tifffile.imread(pred_session / "_bground" / f"{camera}.tiff")
        if not np.array_equal(pred_bground, bground[cameras.index(camera)]):
            raise ValueError(f"Background images differ between sessions for {camera}")

    arrays = {"bground": bground, "K": intrinsics, "dist": distortion,
              "stored_R": stored_R, "stored_t": stored_t}
    for prefix, session in [("gt", gt_session), ("pred", pred_session)]:
        kp2d, kp3d, merged = _load_session_keypoints(session, cameras)
        arrays[f"{prefix}_kp2d"] = kp2d
        arrays[f"{prefix}_kp3d"] = kp3d
        for key, value in merged.items():
            arrays[f"{prefix}_{key}"] = value

    world = np.stack([
        keypoints_to_world(arrays["gt_kp3d"][i], bground[i], intrinsics[i], distortion[i])
        for i in range(len(cameras))
    ])
    arrays["gt_world"] = world.astype("float32")
    fit_keypoints = list(config["incl_kpoints_fit_transform"])
    arrays["refit_R"], arrays["refit_t"], refit_details = refit_transforms(
        world, cameras, reference, node_names, fit_keypoints
    )
    timestamps = pd.read_csv(gt_session / "_proc" / "timestamps.txt", usecols=["device_timestamp_ref"])
    arrays["timestamps"] = timestamps["device_timestamp_ref"].to_numpy()[slice(*FRAME_WINDOW)]
    n_frames = arrays["gt_kp3d"].shape[1]
    if len(arrays["timestamps"]) != n_frames:
        raise ValueError(f"Timestamp window has {len(arrays['timestamps'])} rows for {n_frames} frames")

    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "description": "200-frame, 3-camera clip: curated 2D labels (gt_*) and SLEAP predictions (pred_*)",
        "cameras": cameras,
        "reference_camera": reference,
        "node_names": node_names,
        "labelling_skeleton": load_sleap_skeleton(
            gt_session / "_proc" / "_keypoints_v1" / f"{cameras[0]}.slp"
        ),
        "frame_window": list(FRAME_WINDOW),
        "width": int(width),
        "height": int(height),
        "fps": int(config["fps"]),
        "refit_transforms": {"fit_keypoints": fit_keypoints, "per_camera": refit_details},
        "config_sha256": _sha256(config_path),
        "sources": {
            "gt_session": _source_files(gt_session, cameras),
            "pred_session": _source_files(pred_session, cameras),
        },
    }
    return arrays, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gt-session", type=Path, default=default_session("MARKOVIDS_GT_DATA", "example_session_gt"))
    parser.add_argument("--pred-session", type=Path, default=default_session("MARKOVIDS_PRED_DATA", "example_session"))
    parser.add_argument("--config", type=Path, help="Copy this pipeline config into tests/data first")
    args = parser.parse_args(argv)

    if args.config is not None:
        header = f"# Copied by tests/data/make_gt_fixture.py from {args.config.name}; do not edit by hand.\n"
        FIXTURE_CONFIG.write_text(header + args.config.read_text())
    arrays, manifest = build(args.gt_session, args.pred_session, FIXTURE_CONFIG)
    np.savez_compressed(FIXTURE_NPZ, **arrays)
    FIXTURE_JSON.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {FIXTURE_NPZ.name} ({FIXTURE_NPZ.stat().st_size / 1024:.0f} KB) and {FIXTURE_JSON.name}")
    for camera, details in manifest["refit_transforms"]["per_camera"].items():
        print(f"  refit {camera}: {details}")


if __name__ == "__main__":
    main()
