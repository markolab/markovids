"""Isolated runtime boundaries shared by the unit tests."""

import os
import socket
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "markovids-matplotlib"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Run optional real video, Torch, and JIT smoke tests")
    parser.addoption("--run-gt-full", action="store_true", help="Run slow checks against the full example sessions")


def pytest_collection_modifyitems(config, items):
    optional = [
        ("--run-integration", "integration", "Use --run-integration to run optional runtime smoke tests"),
        ("--run-gt-full", "gt_full", "Use --run-gt-full to run checks against the full example sessions"),
    ]
    for option, marker, reason in optional:
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if item.get_closest_marker(marker):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def prevent_external_execution(monkeypatch, request):
    def forbidden(*args, **kwargs):
        raise AssertionError("External execution must be mocked by the test")

    if not request.node.get_closest_marker("integration"):
        monkeypatch.setattr(subprocess, "Popen", forbidden)
        monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def memory_h5(monkeypatch):
    import h5py

    class MemoryFile(dict):
        def __init__(self, values=None):
            super().__init__(values or {})
            self.attrs = {}
            self.closed = False
            self.created = []

        def create_dataset(self, name, shape=None, dtype=None, data=None, **kwargs):
            value = np.array(data, dtype=dtype) if data is not None else np.zeros(shape, dtype=dtype)
            self[name] = value
            self.created.append((name, kwargs))
            return value

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    files = {}

    def open_file(filename, mode="r", **kwargs):
        key = str(filename)
        if mode == "w":
            files[key] = MemoryFile()
        return files[key]

    factory = Mock(side_effect=open_file)
    monkeypatch.setattr(h5py, "File", factory)
    return SimpleNamespace(files=files, File=MemoryFile, factory=factory)


@pytest.fixture(scope="session")
def gt_fixture():
    """Committed ground-truth fixture (tests/data/gt_session_v1.*)."""
    import gt_support

    gt_support.require_fixture()
    return gt_support.load_gt_fixture()


@pytest.fixture(scope="session")
def gt_runs(gt_fixture, tmp_path_factory):
    """Memoized real registration_pipeline runs on mini sessions built from the fixture.

    ``gt_runs(kind, transforms="refit", merge_method="mixed", post_processing=False)``
    returns the saved HDF5 datasets. ``kind`` is "gt" (curated labels) or "pred"
    (SLEAP predictions); ``transforms`` is "stored", "refit" or a name written
    with ``gt_runs.write_transforms``. Floor fitting is stubbed (see
    ``gt_support.stub_floor_fit``); nothing else is mocked.
    """
    import gt_support
    from markovids.pcl import pipeline

    root = tmp_path_factory.mktemp("gt_sessions")
    sessions = {
        kind: gt_support.materialize_session(gt_fixture, root / f"{kind} session", kind)
        for kind in ("gt", "pred")
    }
    configs = {
        False: gt_support.write_config(gt_fixture, root / "no_post_processing.toml", proc_order=[]),
        True: gt_support.FIXTURE_CONFIG,
    }
    cache = {}

    def run(kind, transforms="refit", merge_method="mixed", post_processing=False, cameras=None):
        key = (kind, transforms, merge_method, post_processing, tuple(cameras or ()))
        if key not in cache:
            output = root / "_".join(map(str, key)).replace(" ", "")
            with pytest.MonkeyPatch.context() as patch:
                gt_support.stub_floor_fit(patch, pipeline)
                cache[key] = gt_support.run_pipeline(
                    gt_fixture, sessions[kind], output, configs[post_processing],
                    transforms, merge_method=merge_method, cameras=cameras,
                )
        return cache[key]

    def write_transforms(name, rotations, translations):
        for session in sessions.values():
            gt_support.write_transforms(gt_fixture, session, name, rotations, translations)

    run.write_transforms = write_transforms
    return run


def _full_session(env_var, name):
    from pathlib import Path

    path = Path(os.environ.get(env_var) or Path(__file__).resolve().parents[1] / "_tmp" / name)
    if not path.is_dir():
        pytest.skip(f"Full example session not found at {path}; set {env_var}=/path/to/{name}")
    return path


@pytest.fixture(scope="session")
def gt_session_path():
    """Full labels-only session (opt-in gt_full tier)."""
    return _full_session("MARKOVIDS_GT_DATA", "example_session_gt")


@pytest.fixture(scope="session")
def pred_session_path():
    """Full predictions session (opt-in gt_full tier)."""
    return _full_session("MARKOVIDS_PRED_DATA", "example_session")


@pytest.fixture
def post_config():
    from markovids.pcl.post_processing import PostProcessingConfig

    return PostProcessingConfig(
        temporal_regularization={"fps": 100},
        bone_length_regularization={"iterations": 2},
        align={"exclude_from_center": ["snout", "tail_tip"]},
        align_compute={"smooth_alignment": False},
        pca={"n_components": 1},
        post_align_hampel={"window_size": 3},
        post_align_imputed_smoothing={"sigma": 1},
        post_align_sgolay={"window_length": 5, "polyorder": 2},
    )


@pytest.fixture
def processor():
    from markovids.pcl.post_processing import KeypointPostProcessor

    return KeypointPostProcessor({"node_names": ["body", "snout", "tail_tip"]}, [("body", "snout")])
