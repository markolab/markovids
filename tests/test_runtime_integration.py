"""Small real-runtime checks; optional dependencies run with --run-integration."""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.depth.io import load_segmentation_masks


def test_segmentation_hdf5_round_trip(tmp_path):
    labels = np.zeros((4, 8, 8), dtype=np.uint8)
    labels[0, 1:4, 2:5] = 1
    labels[2, 3:7, 1:6] = 1
    labels[3, 0:2, 0:2] = 2
    folder = tmp_path / "segmentation"
    folder.mkdir()
    path = folder / "camera.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("labels", data=labels, compression="gzip", chunks=(1, 8, 8))
    args = dict(segmentation_dir="segmentation", clean_masks=False)
    selected = load_segmentation_masks(str(tmp_path / "camera.dat"), [0, 2, 3], **args)
    single = load_segmentation_masks(str(tmp_path / "camera.dat"), 2, **args)
    assert selected.dtype == np.bool_
    assert_array_equal(selected, labels[[0, 2, 3]] == 1)
    assert_array_equal(single, labels[2:3] == 1)
    # Returned arrays remain usable after the production reader closes its file.
    with h5py.File(path, "r+") as handle:
        handle["labels"][:] = 0
    assert selected.sum() == 29


@pytest.mark.integration
def test_preview_video_encode_decode(tmp_path):
    import cv2
    from markovids.vid.io import MP4WriterPreview

    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg executable is required for video integration")
    frames = np.stack([np.full((32, 32, 3), value, dtype=np.uint8) for value in (20, 100, 220)])
    path = tmp_path / "preview.mp4"
    writer = MP4WriterPreview(str(path), frame_size=(32, 32), fps=10, threads=1)
    try:
        writer.write_frames(frames, progress_bar=False, inscribe_frame_number=False)
    finally:
        if writer.pipe is not None:
            writer.close()
    assert writer.pipe.returncode == 0
    capture = cv2.VideoCapture(str(path))
    decoded = []
    try:
        assert capture.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
    finally:
        capture.release()
    assert len(decoded) == len(frames)
    assert np.asarray(decoded).shape == frames.shape
    assert_allclose(np.asarray(decoded).mean(axis=(1, 2, 3)), [20, 100, 220], atol=5)


@pytest.mark.integration
def test_real_torch_checkpoint_imputation(tmp_path):
    torch = pytest.importorskip("torch")
    from markovids.pcl.kpoints import AutoencoderImputer

    model = torch.nn.Sequential(torch.nn.Linear(8, 6))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.copy_(torch.tensor([1., 2., 3., 4., 5., 6.]))
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "mean": np.zeros(6, np.float32),
                "std": np.ones(6, np.float32), "config": {"hidden_sizes": [], "dropout": 0}}, path)
    points = np.full((3, 2, 3), 7., dtype=np.float32)
    points[1, 1] = np.nan
    original = points.copy()
    result = AutoencoderImputer(str(path), batch_size=2).impute(points)
    assert_allclose(result[1, 1], [4, 5, 6])
    assert_array_equal(result[~np.isnan(points)], points[~np.isnan(points)])
    assert_array_equal(points, original)


@pytest.mark.integration
def test_numba_gaussian_compiles_and_matches_reference():
    pytest.importorskip("numba")
    # Unit collection imports this module with JIT disabled. A fresh interpreter
    # verifies compilation without changing the unit suite's process-wide state.
    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "0"
    result = subprocess.run([sys.executable, "-c", """
import numpy as np
from markovids.pcl.fluo import gaussian_2d
x, y = np.meshgrid(np.arange(5., dtype=float), np.arange(4., dtype=float))
args = ((x, y), 2., 1., 1.2, 0.8, 0.3, 7., 0.5)
actual = gaussian_2d(*args)
np.testing.assert_allclose(actual, gaussian_2d.py_func(*args), rtol=1e-12)
assert gaussian_2d.nopython_signatures
assert actual.shape == (4, 5)
assert np.isfinite(actual).all()
"""], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
