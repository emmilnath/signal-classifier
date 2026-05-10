"""1D ResNet for Automatic Modulation Recognition.

Architecture:
    Input: (batch, 2, 1024)  — I/Q channels
    Conv1d stem → 4 ResBlocks → Global Average Pool → FC → 24 classes

References:
    O'Shea, West, "Radio Machine Learning Dataset Generation with GNU Radio" (2016)
    Rajendran et al., "Deep Learning Models for Wireless Signal Classification" (2018)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Residual block with 1D convolutions."""

    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class DownsampleBlock(nn.Module):
    """ResBlock that halves spatial dimension and doubles channels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=2, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class SignalResNet(nn.Module):
    """1D ResNet for modulation classification.

    Args:
        num_classes: Number of modulation types (default 24 for RadioML 2018.01A)
        base_channels: Initial feature map width (doubled at each downsample)
        num_blocks: Residual blocks per stage
        dropout: Dropout rate in residual blocks
    """

    def __init__(
        self,
        num_classes: int = 24,
        base_channels: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Stem: (2, 1024) → (base, 1024)
        self.stem = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 1: (base, 1024) → (base, 1024)
        self.stage1 = nn.Sequential(
            *[ResBlock1D(base_channels, dropout=dropout) for _ in range(num_blocks)]
        )

        # Stage 2: (base, 1024) → (2*base, 512)
        c2 = base_channels * 2
        self.stage2 = nn.Sequential(
            DownsampleBlock(base_channels, c2),
            *[ResBlock1D(c2, dropout=dropout) for _ in range(num_blocks)]
        )

        # Stage 3: (2*base, 512) → (4*base, 256)
        c3 = c2 * 2
        self.stage3 = nn.Sequential(
            DownsampleBlock(c2, c3),
            *[ResBlock1D(c3, dropout=dropout) for _ in range(num_blocks)]
        )

        # Stage 4: (4*base, 256) → (8*base, 128)
        c4 = c3 * 2
        self.stage4 = nn.Sequential(
            DownsampleBlock(c3, c4),
            *[ResBlock1D(c4, dropout=dropout) for _ in range(num_blocks)]
        )

        # Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> SignalResNet:
    """Build model from config dict."""
    return SignalResNet(
        num_classes=cfg.get("num_classes", 24),
        base_channels=cfg.get("base_channels", 64),
        num_blocks=cfg.get("num_blocks", 2),
        dropout=cfg.get("dropout", 0.2),
    )
