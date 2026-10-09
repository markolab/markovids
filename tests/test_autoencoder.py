import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from markovids.pcl import kpoints as kp


@pytest.fixture
def fake_torch(monkeypatch):
    # Model loading and tensor devices are simulated; no checkpoint or GPU is used.
    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        def to(self, device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    class Module:
        training = True

        def __call__(self, value):
            return self.forward(value)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.training = False
            return self

        def load_state_dict(self, state):
            self.state = state

    class Linear(Module):
        def __init__(self, input_size, output_size):
            self.input_size = input_size
            self.output_size = output_size

        def forward(self, value):
            return np.zeros((len(value), self.output_size))

    class Identity(Module):
        def __init__(self, *args):
            pass

        def forward(self, value):
            return value

    class Sequential(Module):
        def __init__(self, *layers):
            self.layers = layers

        def forward(self, value):
            for layer in self.layers:
                value = layer(value)
            return value

    torch = ModuleType("torch")
    nn = ModuleType("torch.nn")
    nn.Module, nn.Linear, nn.Sequential = Module, Linear, Sequential
    nn.LayerNorm = nn.BatchNorm1d = nn.ReLU = nn.Dropout = Identity
    torch.nn = nn
    torch.load = Mock()
    torch.from_numpy = Tensor
    torch.no_grad = nullcontext
    torch.exp = np.exp
    torch.randn_like = Mock(side_effect=lambda values: np.ones_like(values))
    torch.Tensor = Tensor
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    return torch


@pytest.mark.parametrize("training", [True, False])
def test_should_sample_latent_only_during_training_when_building_vae(fake_torch, training):
    model = kp._build_vae(8, 6, [4, 3], latent_dim=2, dropout=0)
    model.training = training
    reconstruction, mean, log_variance = model(np.zeros((5, 8)))
    assert reconstruction.shape == (5, 6)
    assert mean.shape == log_variance.shape == (5, 2)
    assert fake_torch.randn_like.call_count == int(training)
    assert model.encoder.layers[0].input_size == 8
    assert model.decoder.layers[-1].output_size == 6


@pytest.mark.parametrize("model_type", ["ae", "vae"])
def test_should_load_scaler_and_model_state_when_checkpoint_is_valid(fake_torch, model_type):
    fake_torch.load.return_value = {
        "config": {"hidden_sizes": [4], "dropout": 0, "latent_dim": 2},
        "mean": [1, 2, 3, 4, 5, 6], "std": [2] * 6,
        "model_state_dict": {"fake": 1}, "model_type": model_type,
    }
    imputer = kp.AutoencoderImputer("checkpoint.pt", batch_size=2)
    fake_torch.load.assert_called_once_with("checkpoint.pt", map_location="cpu", weights_only=False)
    assert_array_equal(imputer._mean, [1, 2, 3, 4, 5, 6])
    assert imputer._mean.dtype == np.float32
    assert imputer._model.state == {"fake": 1}
    assert imputer._model.training is False
    assert imputer._is_vae == (model_type == "vae")


def test_should_propagate_checkpoint_failure_when_model_file_cannot_be_loaded(fake_torch):
    failure = FileNotFoundError("checkpoint missing")
    fake_torch.load.side_effect = failure
    with pytest.raises(FileNotFoundError) as error:
        kp.AutoencoderImputer("missing.pt")
    assert error.value is failure


@pytest.mark.parametrize("vae_output", [False, True])
def test_should_fill_only_missing_coordinates_and_batch_inference_when_imputing(fake_torch, monkeypatch, vae_output):
    batches = []

    def infer(batch):
        batches.append(batch.numpy().copy())
        reconstruction = fake_torch.Tensor(np.ones((len(batch.numpy()), 6)))
        return (reconstruction, object(), object()) if vae_output else reconstruction

    model = Mock(side_effect=infer)
    loader = Mock(return_value=(model, np.full(6, 10, np.float32), np.full(6, 2, np.float32), {"model_type": "vae" if vae_output else "ae"}))
    monkeypatch.setattr(kp.AutoencoderImputer, "_load_checkpoint", loader)
    data = np.ones((5, 2, 3), dtype=float)
    data[1, 0] = np.nan
    data[2, 1, 1] = np.nan
    data[3, 1, 2] = -3
    original = data.copy()
    imputer = kp.AutoencoderImputer("fake.pt", batch_size=2)
    result = imputer.impute(data)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert_allclose(result[np.isnan(data)], 12)
    assert result[3, 1, 2] == 0
    observed = ~np.isnan(data)
    observed[3, 1, 2] = False
    assert_allclose(result[observed], data[observed])
    assert_array_equal(batches[0][:, -2:], [[1, 1], [0, 1]])
    assert_allclose(data, original, equal_nan=True)


def test_should_propagate_inference_error_when_model_execution_fails(fake_torch, monkeypatch):
    failure = RuntimeError("inference failed")
    monkeypatch.setattr(kp.AutoencoderImputer, "_load_checkpoint", Mock(return_value=(Mock(side_effect=failure), np.zeros(3), np.ones(3), {})))
    imputer = kp.AutoencoderImputer("fake.pt")
    with pytest.raises(RuntimeError) as error:
        imputer.impute(np.ones((1, 1, 3)))
    assert error.value is failure
