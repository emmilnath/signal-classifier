"""Training script for modulation classifier.

Usage:
    python -m signal_classifier.train --config configs/default.yaml
    python -m signal_classifier.train --synthetic   # Quick test without dataset
"""
from __future__ import annotations
import argparse
import pathlib
import time
from datetime import datetime

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .dataset import get_dataloaders, MODULATIONS
from .model import build_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train modulation classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # Load config
    cfg_path = pathlib.Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # CLI overrides
    train_cfg = cfg.get("training", {})
    epochs = args.epochs or train_cfg.get("epochs", 60)
    batch_size = args.batch_size or train_cfg.get("batch_size", 256)
    lr = args.lr or train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    data_path = cfg.get("data", {}).get("hdf5_path", None)
    snr_filter = cfg.get("data", {}).get("snr_filter", None)
    if snr_filter:
        snr_filter = tuple(snr_filter)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    synthetic = args.synthetic or data_path is None
    train_loader, val_loader, test_loader = get_dataloaders(
        hdf5_path=data_path,
        batch_size=batch_size,
        snr_filter=snr_filter,
        num_workers=0 if synthetic else 4,
        synthetic=synthetic,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Model
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Output directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": cfg,
            }, out_dir / "best_model.pt")

    # Final test evaluation
    ckpt = torch.load(out_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")
    print(f"Best checkpoint saved to {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
