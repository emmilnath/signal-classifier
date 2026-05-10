# 📡 Signal Classifier

Deep learning automatic modulation recognition (AMR) using a 1D ResNet architecture on IQ samples. Designed for the RadioML 2018.01A dataset (24 modulation classes, varying SNR from -20 dB to +30 dB).

## Features

- **1D ResNet Architecture** — Four-stage residual network with downsampling, batch normalization, and global average pooling operating directly on raw IQ time series
- **RadioML 2018.01A Support** — HDF5 dataset loader with stratified train/val/test splits and optional SNR filtering
- **Signal-Domain Augmentations** — Random phase rotation, time shifting, amplitude scaling, and additive Gaussian noise designed specifically for RF signals
- **Configurable Training** — YAML-based configs, cosine annealing LR schedule, gradient clipping, AdamW optimizer
- **Evaluation Suite** — Per-class classification report, normalized confusion matrix, accuracy-vs-SNR curve generation
- **Synthetic Mode** — Full training pipeline works without the dataset for development and testing

## Quick Start

### Train with synthetic data (no download needed)
```bash
pip install -e .
python -m signal_classifier.train --synthetic --epochs 5
```

### Train on RadioML 2018.01A
```bash
# Download dataset from https://www.deepsig.ai/datasets/
# Update configs/default.yaml with the HDF5 path
python -m signal_classifier.train --config configs/default.yaml
```

### Evaluate a trained model
```bash
python -m signal_classifier.evaluate --checkpoint runs/YYYYMMDD/best_model.pt --plot
```

### Run tests
```bash
pytest tests/ -v
```

## Project Structure

```
signal-classifier/
├── signal_classifier/
│   ├── __init__.py
│   ├── model.py             # 1D ResNet (SignalResNet)
│   ├── dataset.py           # RadioML HDF5 loader + synthetic fallback
│   ├── augmentations.py     # RF-domain transforms
│   ├── train.py             # Training loop with logging
│   └── evaluate.py          # Metrics, confusion matrix, SNR curves
├── configs/
│   ├── default.yaml         # Standard training config
│   ├── high_snr.yaml        # High-SNR subset (≥10 dB)
│   └── lightweight.yaml     # Smaller model for edge deployment
├── tests/
│   └── test_model.py        # Architecture, data pipeline, augmentation tests
├── pyproject.toml
└── README.md
```

## Architecture

```
Input: (batch, 2, 1024) — I/Q channels
  ↓
Conv1d Stem (2 → 64, k=7) + BN + ReLU
  ↓
Stage 1: 2× ResBlock1D (64 channels, 1024 length)
  ↓
Stage 2: Downsample + 2× ResBlock1D (128 channels, 512 length)
  ↓
Stage 3: Downsample + 2× ResBlock1D (256 channels, 256 length)
  ↓
Stage 4: Downsample + 2× ResBlock1D (512 channels, 128 length)
  ↓
Global Average Pooling → Dropout → Linear(512, 24)
```

Total parameters: ~3.2M (default config)

## Modulation Classes (RadioML 2018.01A)

OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK

## References

- O'Shea, West — "Radio Machine Learning Dataset Generation with GNU Radio" (2016)
- Rajendran et al. — "Deep Learning Models for Wireless Signal Classification" (2018)
- O'Shea, Roy, Clancy — "Over-the-Air Deep Learning Based Radio Signal Classification" (2018)

## License

MIT
