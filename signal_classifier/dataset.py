"""RadioML 2018.01A dataset loader.

Expected HDF5 structure (from DeepSig):
    /X  — complex IQ samples, shape (N, 1024, 2)  → I and Q channels
    /Y  — one-hot labels, shape (N, 24)
    /Z  — SNR values, shape (N, 1)

Download: https://www.deepsig.ai/datasets/
"""
from __future__ import annotations
import pathlib
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


# 24 modulation classes in RadioML 2018.01A
MODULATIONS = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


class RadioMLDataset(Dataset):
    """PyTorch dataset wrapping the RadioML 2018.01A HDF5 file."""

    def __init__(
        self,
        hdf5_path: str | pathlib.Path,
        snr_filter: Optional[tuple[int, int]] = None,
        transform=None,
    ):
        self.path = pathlib.Path(hdf5_path)
        self.transform = transform

        with h5py.File(self.path, "r") as f:
            self.X = f["X"][:]          # (N, 1024, 2)
            self.Y = f["Y"][:]          # (N, 24) one-hot
            self.Z = f["Z"][:].ravel()  # (N,) SNR

        # Convert one-hot to class index
        self.labels = np.argmax(self.Y, axis=1)

        # Optional SNR filtering
        if snr_filter is not None:
            lo, hi = snr_filter
            mask = (self.Z >= lo) & (self.Z <= hi)
            self.X = self.X[mask]
            self.labels = self.labels[mask]
            self.Z = self.Z[mask]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Shape: (1024, 2) → (2, 1024) for Conv1d
        x = torch.from_numpy(self.X[idx].T).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            x = self.transform(x)
        return x, y

    @property
    def num_classes(self):
        return len(MODULATIONS)


class SyntheticDataset(Dataset):
    """Synthetic IQ dataset for testing without RadioML download."""

    def __init__(self, num_samples: int = 10000, seq_len: int = 1024, num_classes: int = 24):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.X = np.random.randn(num_samples, 2, seq_len).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def get_dataloaders(
    hdf5_path: str | pathlib.Path | None,
    batch_size: int = 256,
    val_split: float = 0.15,
    test_split: float = 0.15,
    snr_filter: Optional[tuple[int, int]] = None,
    num_workers: int = 4,
    synthetic: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""

    if synthetic or hdf5_path is None:
        dataset = SyntheticDataset()
    else:
        dataset = RadioMLDataset(hdf5_path, snr_filter=snr_filter)

    n = len(dataset)
    indices = np.arange(n)

    # Stratified split
    train_idx, test_idx = train_test_split(
        indices, test_size=test_split, stratify=dataset.labels, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_split / (1 - test_split),
        stratify=dataset.labels[train_idx], random_state=42
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
