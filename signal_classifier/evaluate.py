"""Evaluation script — confusion matrix, per-SNR accuracy, classification report.

Usage:
    python -m signal_classifier.evaluate --checkpoint runs/YYYYMMDD/best_model.pt
    python -m signal_classifier.evaluate --checkpoint runs/YYYYMMDD/best_model.pt --plot
"""
from __future__ import annotations
import argparse
import pathlib

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import get_dataloaders, MODULATIONS, RadioMLDataset
from .model import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(cm, labels, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")


def plot_snr_accuracy(model, dataset, device, save_path, batch_size=512):
    """Plot accuracy vs SNR curve."""
    snrs = sorted(set(dataset.Z.tolist()))
    accs = []

    for snr in snrs:
        mask = dataset.Z == snr
        indices = np.where(mask)[0]
        if len(indices) == 0:
            accs.append(0.0)
            continue

        correct, total = 0, 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            x = torch.from_numpy(dataset.X[batch_idx].transpose(0, 2, 1)).float().to(device)
            y = torch.from_numpy(dataset.labels[batch_idx])
            with torch.no_grad():
                preds = model(x).argmax(1).cpu()
            correct += (preds == y).sum().item()
            total += len(batch_idx)
        accs.append(correct / total)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snrs, [a * 100 for a in accs], "o-", color="#2980b9", linewidth=2, markersize=5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy vs SNR")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved SNR-accuracy curve to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate modulation classifier")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data", type=str, default=None, help="HDF5 dataset path")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = build_model(cfg.get("model", {})).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    # Data
    data_path = args.data or cfg.get("data", {}).get("hdf5_path")
    synthetic = args.synthetic or data_path is None
    _, _, test_loader = get_dataloaders(
        hdf5_path=data_path,
        batch_size=256,
        num_workers=0 if synthetic else 4,
        synthetic=synthetic,
    )

    # Predictions
    preds, labels = collect_predictions(model, test_loader, device)

    # Classification report
    report = classification_report(labels, preds, target_names=MODULATIONS, digits=3)
    print("\n" + report)

    # Save outputs
    out_dir = pathlib.Path(args.checkpoint).parent
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    if args.plot:
        cm = confusion_matrix(labels, preds, normalize="true")
        plot_confusion_matrix(cm, MODULATIONS, out_dir / "confusion_matrix.png")

        # SNR curve (only with real dataset)
        if not synthetic and data_path:
            from .dataset import RadioMLDataset
            dataset = RadioMLDataset(data_path)
            plot_snr_accuracy(model, dataset, device, out_dir / "snr_accuracy.png")


if __name__ == "__main__":
    main()
