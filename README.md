# CSC4851/6851 Course Project: MNIST Classification

Manual implementation of MLP and CNN neural networks from scratch using only NumPy for handwritten digit classification on the MNIST dataset.

## Environment

- **Python**: 3.13+
- **Dependencies**: NumPy (only)
- **OS**: macOS / Linux / Windows

### Setup

```bash
pip install numpy
```

No other libraries are required. The MNIST dataset is automatically downloaded on first run.

## How to Run

```bash
# Task 1: MLP
python3 task1_mlp.py

# Task 2: CNN
python3 task2_cnn.py
```

## Project Structure

| File | Description |
|------|-------------|
| `dataloader.py` | Shared MNIST dataloader — downloads data and yields batches |
| `task1_mlp.py` | Task 1: Multi-Layer Perceptron (MLP) from scratch |
| `task2_cnn.py` | Task 2: Convolutional Neural Network (CNN) from scratch |

---

## Task 1: MLP — Multi-Layer Perceptron

### Architecture

```
Input (784) → FC (1024 neurons) + Sigmoid → FC (10 neurons) + Softmax → Output
```

### Best Model Configuration

| Hyperparameter | Value |
|----------------|-------|
| Hidden size | 1024 |
| Learning rate | 0.15 (decay 0.97/epoch) |
| Optimizer | Nesterov Momentum SGD (momentum=0.9) |
| Weight init | Xavier (sigmoid-optimized) |
| Input normalization | Yes (global zero mean, unit variance) |
| Weight decay | 1e-4 |
| Data augmentation | Random shifts (max 2px) |
| SWA | Epochs 20-30 |
| TTA | 9 shifts ({-1,0,+1}²) |
| Epochs | 30 |
| Batch size | 128 |
| **Final Test Accuracy** | **98.92%** (with TTA) |

### Optimization History

| Version | Hidden Size | LR | Optimizer | Normalization | Accuracy |
|---------|------------|-----|-----------|---------------|----------|
| Baseline | 256 | 0.5 | Vanilla SGD | No | 97.90% |
| v2 | 512 | 1.0 (decay 0.95) | Momentum SGD (0.9) | No | 94.66% |
| v3 | 512 | 0.5 (decay 0.98) | Momentum SGD (0.9) | No | 95.46% |
| v4 | 512 | 0.1 (decay 0.98) | Momentum SGD (0.9) | No | 98.05% |
| v5 | 784 | 0.15 (decay 0.97) | Momentum SGD (0.9) | No | 98.16% |
| v6 | 784 | 0.15 (decay 0.97) | Momentum SGD (0.9) | Yes | 98.47% |
| v7 | 784 | 0.15 (decay 0.97) | Momentum SGD (0.9) | Yes + augment | 98.81% |
| **v8 (Best)** | **1024** | **0.15 (decay 0.97)** | **Nesterov SGD (0.9)** | **Yes + augment + SWA + TTA** | **98.92%** |

### What Improved Accuracy (v7 → v8)

1. **Nesterov momentum**: Replaced classical momentum with the Nesterov "look-ahead" variant. Uses the reformulation: `v_prev = v.copy(); v = μ*v - lr*grad; w += -μ*v_prev + (1+μ)*v`. This provides better gradient estimates and slightly faster convergence.

2. **Increased hidden size (784 → 1024)**: More capacity to learn complex decision boundaries. Still within architecture constraints (2 layers).

3. **Stochastic Weight Averaging (SWA)**: Averages model weights from epochs 20-30. SWA finds flatter minima that generalize better, providing a small but consistent improvement.

4. **Test-Time Augmentation (TTA)**: At inference time, predicts on 9 versions of each test image (original + 8 pixel shifts from {-1,0,+1}×{-1,0,+1}), averages the softmax outputs, then takes argmax. Pure inference trick that costs nothing during training.

### Key Lesson

More aggressive optimization (higher LR, stronger momentum) does NOT always help. v2 was the most "optimized" on paper but performed the worst (94.66%) because the combined effect of LR=1.0 + momentum=0.9 made the effective step size too large, causing the training to overshoot. Finding the right balance was critical.

---

## Task 2: CNN — Convolutional Neural Network

### Architecture

```
Input (28x28) → Conv2D (1 kernel, 5x5, valid mode) + ReLU → Flatten (576) → FC (10 neurons) + Softmax → Output
```

### Best Model Configuration

| Hyperparameter | Value |
|----------------|-------|
| Kernel size | 5x5 |
| Learning rate | 0.02 (constant) |
| Optimizer | Nesterov Momentum SGD (momentum=0.9) |
| Weight init | He (ReLU-optimized) |
| Conv bias | Yes (scalar) |
| Input normalization | Yes (global zero mean, unit variance) |
| Weight decay | 0.0 (disabled) |
| TTA | 9 shifts ({-1,0,+1}²) |
| Epochs | 5 |
| Batch size | 128 |
| **Final Test Accuracy** | **94.88%** (with TTA) |

### Optimization History

| Version | Kernel | LR | Optimizer | Normalization | Accuracy |
|---------|--------|-----|-----------|---------------|----------|
| Baseline | 5x5 | 0.01 | Vanilla SGD | No | 91.75% |
| v2 | 5x5 | 0.05 | Momentum SGD (0.9) | No | 92.81% |
| v3 | 3x3 | 0.05 | Momentum SGD (0.9) | No | 92.55% |
| v4 | 7x7 | 0.05 | Momentum SGD (0.9) | No | 92.33% |
| v5 | 5x5 | 0.1 | Momentum SGD (0.9) | No | 92.43% |
| v6 | 5x5 | 0.03 | Momentum SGD (0.95) | No | 92.60% |
| v7 | 5x5 | 0.05 | Momentum SGD (0.9) | Yes | 92.48% |
| v8 | 5x5 | 0.02 | Momentum SGD (0.9) | Yes | 92.85% |
| v9 | 5x5 | 0.01 | Momentum SGD (0.9) | Yes | 94.28% |
| v10 | 5x5 | 0.01 | Momentum SGD (0.9) | Yes, wd=1e-4 | 94.33% |
| **v11 (Best)** | **5x5** | **0.02** | **Nesterov SGD (0.9)** | **Yes, wd=0, TTA** | **94.88%** |

### What Improved Accuracy (v10 → v11)

1. **Nesterov momentum**: Look-ahead gradient updates provided better convergence within the limited 5 epochs.

2. **Higher learning rate (0.01 → 0.02)**: With Nesterov's better gradient estimates, the model can tolerate a slightly higher LR, extracting more learning from each epoch.

3. **Removed weight decay**: With only 5 epochs and 1 kernel, the model is nowhere near overfitting. L2 regularization was actively constraining the kernel and hurting performance.

4. **Test-Time Augmentation (TTA)**: Same 9-shift ensemble as MLP. Provides a consistent +0.17% boost at inference time.

### Architectural Limitation

The CNN is constrained to **1 convolutional kernel**, which means it can only learn a single feature detector (e.g., one type of edge or pattern). All discrimination power for the 10 digit classes must come from the spatial distribution of that single feature, interpreted by the FC layer. This fundamentally caps accuracy around 93-95%. A standard CNN with 32+ kernels would easily reach 99%+.

### What Didn't Help the CNN

- **Label smoothing**: Slowed convergence with only 5 epochs available
- **Per-pixel normalization**: Hurt the single-kernel CNN which benefits from globally uniform feature scales
- **Cosine annealing**: Decayed LR too aggressively for 5-epoch training
- **Data augmentation**: Not enough epochs for the model to benefit from augmented data

---

## All 7 Implemented Components (Both Tasks)

| # | Component | MLP | CNN |
|---|-----------|-----|-----|
| 1 | Forward Propagation | FC+Sigmoid, FC+Softmax | Conv2D+ReLU, FC+Softmax |
| 2 | Backward Propagation | Chain rule through both layers | Conv kernel gradients + FC gradients |
| 3 | Train Function | Batch-based, loss tracking | Batch-based, loss tracking |
| 4 | Activation (Layer 1) | Sigmoid | ReLU |
| 5 | Softmax (Layer 2) | Numerically stable softmax | Numerically stable softmax |
| 6 | Cross-Entropy Loss | Mean CE over batch | Mean CE over batch |
| 7 | Main Function | Train loop + evaluation | Train loop + evaluation |

## Implementation Details

- **No deep learning libraries** — only NumPy for matrix operations
- **Vectorized convolution** using `np.lib.stride_tricks.as_strided` and `np.einsum` for efficient batch processing
- **Numerically stable** softmax (max subtraction) and cross-entropy (log clipping)
- **Reproducible results** via `np.random.seed(42)`
- **Automatic MNIST download** on first run

## Advanced Techniques

- **Nesterov Momentum SGD**: Look-ahead gradient updates for better convergence
- **Stochastic Weight Averaging (SWA)**: Averages weights from last 11 epochs for flatter minima
- **Test-Time Augmentation (TTA)**: 9-way shift ensemble at inference time
- **Data Augmentation**: Random pixel shifts during MLP training
