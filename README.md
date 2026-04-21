# CSC 4851/6851 — MNIST Classification from Scratch

**Team:** Farhan Baburi · Ibe Mohammed Ali · Charan Tej Peeriga  
**Course:** CSC 4851/6851 Introduction to Deep Learning · Spring 2026

Manual implementation of MLP and CNN neural networks using **only NumPy** — no PyTorch, TensorFlow, or any deep learning library.

---

## Results

| Model | Accuracy (no TTA) | Accuracy (with TTA) |
|-------|-------------------|---------------------|
| Task 1: MLP (784 → 1024 → 10) | 98.66% | **98.92%** |
| Task 2: CNN (Conv 5×5 + FC) | 94.71% | **94.88%** |

---

## Setup

**Requirements:** Python 3.13+, NumPy

```bash
pip install numpy
```

The MNIST dataset is automatically downloaded on first run into the `data/` folder.

---

## How to Run

```bash
# Task 1: Multi-Layer Perceptron
python3 task1_mlp.py

# Task 2: Convolutional Neural Network
python3 task2_cnn.py
```

---

## Project Structure

```
.
├── dataloader.py     # MNIST download & batch loader
├── task1_mlp.py      # Task 1 — MLP implementation
├── task2_cnn.py      # Task 2 — CNN implementation
├── data/             # MNIST binary files (auto-downloaded)
└── README.md
```

---

## Implemented Components (Both Tasks)

Each task manually implements all 7 required components:

| # | Component | MLP | CNN |
|---|-----------|-----|-----|
| 1 | Forward Propagation | FC+Sigmoid → FC+Softmax | Conv2D+ReLU → FC+Softmax |
| 2 | Backward Propagation | Chain rule, both layers | Conv kernel gradient + FC gradient |
| 3 | Train Function | Mini-batch loop | Mini-batch loop |
| 4 | Activation Function | Sigmoid (clipped) | ReLU |
| 5 | Softmax | Numerically stable (max subtraction) | Numerically stable (max subtraction) |
| 6 | Cross-Entropy Loss | Mean CE, log-clipped | Mean CE, log-clipped |
| 7 | Main Function | Train + evaluate | Train + evaluate |

---

## Task 1: MLP

### Architecture
```
Input (784) → FC (1024) + Sigmoid → FC (10) + Softmax → Output
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Learning rate | 0.15 × 0.97^epoch |
| Optimizer | Nesterov Momentum SGD (μ=0.9) |
| Weight init | Xavier |
| Weight decay | 1×10⁻⁴ |
| Data augmentation | Random shifts ≤ 2px |
| SWA | Epochs 20–30 |
| TTA | 9 shifts ({−1,0,+1}²) |
| Epochs | 30 |
| Batch size | 128 |

### Advanced Techniques
- **Nesterov SGD** — look-ahead gradient updates for faster convergence
- **Stochastic Weight Averaging (SWA)** — averages weights from epochs 20–30 to find flatter minima
- **Test-Time Augmentation (TTA)** — 9-shift ensemble at inference; averages softmax outputs

### Optimization History
| Version | Hidden | LR | Optimizer | Notes | Accuracy |
|---------|--------|----|-----------|-------|----------|
| Baseline | 256 | 0.50 | Vanilla SGD | — | 97.90% |
| v2 | 512 | 1.00 | Momentum 0.9 | LR too high | 94.66% |
| v3 | 512 | 0.50 | Momentum 0.9 | — | 95.46% |
| v4 | 512 | 0.10 | Momentum 0.9 | — | 98.05% |
| v5 | 784 | 0.15 | Momentum 0.9 | — | 98.16% |
| v6 | 784 | 0.15 | Momentum 0.9 | + normalization | 98.47% |
| v7 | 784 | 0.15 | Momentum 0.9 | + augmentation | 98.81% |
| v8 (Best) | 1024 | 0.15 | Nesterov | + SWA + TTA | 98.92% |

---

## Task 2: CNN

### Architecture
```
Input (28×28) → Conv2D (1 kernel, 5×5, valid) + ReLU → Flatten (576) → FC (10) + Softmax → Output
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Kernel size | 5×5 |
| Learning rate | 0.02 (constant) |
| Optimizer | Nesterov Momentum SGD (μ=0.9) |
| Weight init | He |
| Weight decay | 0 (disabled) |
| TTA | 9 shifts ({−1,0,+1}²) |
| Epochs | 5 |
| Batch size | 128 |

### Vectorized Convolution
Uses `np.lib.stride_tricks.as_strided` + `np.einsum` — no Python loops over spatial dimensions.

### Optimization History
| Version | LR | Optimizer | Notes | Accuracy |
|---------|----|-----------|-------|----------|
| Baseline | 0.01 | Vanilla SGD | — | 91.75% |
| v2–v8 | various | Momentum | hyperparameter search | 92.43–92.85% |
| v9 | 0.01 | Momentum | + normalization | 94.28% |
| v10 | 0.01 | Momentum | + weight decay | 94.33% |
| v11 (Best) | 0.02 | Nesterov | + TTA, wd=0 | 94.88% |

### Architectural Limitation
The CNN is constrained to **1 kernel** — it can learn only one feature detector. All 10-class discrimination comes from the spatial distribution of that single feature. This caps accuracy at ~95%. A standard CNN with 32+ kernels reaches 99%+.

---

## Implementation Notes

- **No deep learning libraries** — NumPy only
- **Reproducible** — `np.random.seed(42)` throughout
- **Numerically stable** — softmax with max subtraction; cross-entropy with log clipping to [1e-12, 1]
- **Vectorized convolution** — `as_strided` patch extraction + `einsum` for efficient batch processing
