"""Unit tests for model architecture and data pipeline."""
import pytest
import torch
import numpy as np
from signal_classifier.model import SignalResNet, build_model
from signal_classifier.dataset import SyntheticDataset, get_dataloaders
from signal_classifier.augmentations import (
    RandomPhaseRotation, RandomTimeShift, AddGaussianNoise, Normalize, Compose
)


class TestSignalResNet:
    def test_output_shape(self):
        model = SignalResNet(num_classes=24)
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert out.shape == (4, 24)

    def test_output_shape_custom(self):
        model = SignalResNet(num_classes=11, base_channels=32, num_blocks=1)
        x = torch.randn(2, 2, 1024)
        out = model(x)
        assert out.shape == (2, 11)

    def test_gradient_flow(self):
        model = SignalResNet(num_classes=24, base_channels=32)
        x = torch.randn(2, 2, 1024, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_parameter_count(self):
        model = SignalResNet(base_channels=64, num_blocks=2)
        count = model.count_parameters()
        assert count > 100_000  # Should be a reasonable model size
        assert count < 50_000_000  # Not absurdly large

    def test_build_from_config(self):
        cfg = {"num_classes": 10, "base_channels": 32, "num_blocks": 1, "dropout": 0.1}
        model = build_model(cfg)
        x = torch.randn(2, 2, 1024)
        out = model(x)
        assert out.shape == (2, 10)


class TestDataset:
    def test_synthetic_dataset(self):
        ds = SyntheticDataset(num_samples=100, seq_len=1024, num_classes=24)
        assert len(ds) == 100
        x, y = ds[0]
        assert x.shape == (2, 1024)
        assert 0 <= y.item() < 24

    def test_dataloaders(self):
        train_dl, val_dl, test_dl = get_dataloaders(
            hdf5_path=None, batch_size=16, num_workers=0, synthetic=True
        )
        batch_x, batch_y = next(iter(train_dl))
        assert batch_x.shape[1] == 2
        assert batch_x.shape[2] == 1024
        assert batch_y.shape[0] == batch_x.shape[0]


class TestAugmentations:
    def test_phase_rotation_preserves_power(self):
        x = torch.randn(2, 1024)
        power_before = (x ** 2).sum()
        x_rot = RandomPhaseRotation()(x)
        power_after = (x_rot ** 2).sum()
        assert torch.allclose(power_before, power_after, rtol=1e-5)

    def test_time_shift_preserves_shape(self):
        x = torch.randn(2, 1024)
        x_shifted = RandomTimeShift(max_shift=100)(x)
        assert x_shifted.shape == x.shape

    def test_normalize_zero_mean_unit_var(self):
        x = torch.randn(2, 1024) * 5 + 3
        x_norm = Normalize()(x)
        assert torch.allclose(x_norm.mean(dim=1), torch.zeros(2), atol=1e-5)
        assert torch.allclose(x_norm.std(dim=1), torch.ones(2), atol=1e-2)

    def test_compose(self):
        transform = Compose([RandomPhaseRotation(), Normalize()])
        x = torch.randn(2, 1024)
        x_out = transform(x)
        assert x_out.shape == x.shape

    def test_gaussian_noise_changes_signal(self):
        x = torch.randn(2, 1024)
        x_noisy = AddGaussianNoise(snr_range=(10, 10))(x)
        assert not torch.allclose(x, x_noisy)
