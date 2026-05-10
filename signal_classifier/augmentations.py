"""Signal-domain data augmentations for IQ samples.

All transforms operate on tensors of shape (2, L) where channel 0 is I, channel 1 is Q.
"""
from __future__ import annotations
import torch
import numpy as np


class RandomPhaseRotation:
    """Rotate IQ constellation by a random angle ∈ [0, 2π).

    This is the most important augmentation for modulation recognition
    because the absolute carrier phase is arbitrary.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        theta = torch.rand(1).item() * 2 * np.pi
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        i, q = x[0], x[1]
        x_out = x.clone()
        x_out[0] = i * cos_t - q * sin_t
        x_out[1] = i * sin_t + q * cos_t
        return x_out


class RandomTimeShift:
    """Circular shift along the time axis by a random amount."""
    def __init__(self, max_shift: int = 128):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=1)


class AddGaussianNoise:
    """Add white Gaussian noise at a specified SNR range (dB)."""
    def __init__(self, snr_range: tuple[float, float] = (5.0, 30.0)):
        self.snr_lo, self.snr_hi = snr_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        snr_db = torch.empty(1).uniform_(self.snr_lo, self.snr_hi).item()
        signal_power = (x ** 2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise


class RandomAmplitudeScale:
    """Scale amplitude by a random factor."""
    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        self.lo, self.hi = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.lo, self.hi).item()
        return x * scale


class Normalize:
    """Normalize IQ samples to zero mean and unit variance per channel."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std


class Compose:
    """Chain multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def default_train_transform():
    return Compose([
        RandomPhaseRotation(),
        RandomTimeShift(max_shift=64),
        RandomAmplitudeScale((0.9, 1.1)),
        Normalize(),
    ])


def default_eval_transform():
    return Normalize()
