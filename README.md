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
Input (784) → FC (784 neurons) + Sigmoid → FC (10 neurons) + Softmax → Output
```

### Best Model Configuration

| Hyperparameter | Value |
|----------------|-------|
| Hidden size | 784 |
| Learning rate | 0.15 (decay 0.97/epoch) |
| Optimizer | Momentum SGD (momentum=0.9) |
| Weight init | Xavier (sigmoid-optimized) |
| Input normalization | Yes (zero mean, unit variance) |
| Epochs | 30 |
| Batch size | 128 |
| **Final Test Accuracy** | **98.47%** |

### Optimization History

| Version | Hidden Size | LR | Optimizer | Normalization | Accuracy |
|---------|------------|-----|-----------|---------------|----------|
| Baseline | 256 | 0.5 | Vanilla SGD | No | 97.90% |
| v2 | 512 | 1.0 (decay 0.95) | Momentum SGD (0.9) | No | 94.66% |
| v3 | 512 | 0.5 (decay 0.98) | Momentum SGD (0.9) | No | 95.46% |
| v4 | 512 | 0.1 (decay 0.98) | Momentum SGD (0.9) | No | 98.05% |
| v5 | 784 | 0.15 (decay 0.97) | Momentum SGD (0.9) | No | 98.16% |
| **v6 (Best)** | **784** | **0.15 (decay 0.97)** | **Momentum SGD (0.9)** | **Yes** | **98.47%** |

### What Improved Accuracy

1. **Xavier initialization** (`sqrt(2 / (fan_in + fan_out))`): Properly scaled initial weights for sigmoid activation, preventing vanishing gradients at the start of training.

2. **Momentum SGD (0.9)**: Accelerated convergence by accumulating past gradient direction, but required careful LR tuning — too high a learning rate with momentum (v2: LR=1.0) caused instability and *decreased* accuracy to 94.66%.

3. **Learning rate tuning**: With momentum, the effective step size is approximately `lr / (1 - momentum)`. LR=0.1 with momentum=0.9 gives an effective LR of ~1.0, which was optimal. LR=1.0 with momentum=0.9 (effective ~10.0) was far too aggressive.

4. **Larger hidden layer (784 neurons)**: More capacity allowed the model to learn more complex decision boundaries. Going from 512 to 784 improved accuracy by ~0.1%.

5. **LR decay (0.97/epoch)**: Gradually reducing the learning rate allowed fine-tuning in later epochs, preventing oscillation around the optimum.

6. **Input normalization (zero mean, unit variance)**: The single biggest improvement (+0.3%). Normalizing pixel values from [0,1] to standardized form made gradient magnitudes more uniform across features, significantly accelerating convergence.

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
| Learning rate | 0.01 |
| Optimizer | Momentum SGD (momentum=0.9) |
| Weight init | He (ReLU-optimized) |
| Conv bias | Yes (scalar) |
| Input normalization | Yes (zero mean, unit variance) |
| Epochs | 5 |
| Batch size | 128 |
| **Final Test Accuracy** | **94.28%** |

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
| **v9 (Best)** | **5x5** | **0.01** | **Momentum SGD (0.9)** | **Yes** | **94.28%** |

### What Improved Accuracy

1. **He initialization** (`sqrt(2 / fan_in)`): Proper weight scaling for ReLU activation, ensuring gradients flow well through the network from the start.

2. **Momentum SGD (0.9)**: Helped the single kernel converge faster within the limited 5 epochs. With only 5 epochs, every update counts.

3. **Convolution bias**: Adding a learnable scalar bias to the conv layer gave the network an extra degree of freedom.

4. **Input normalization + lower LR**: The combination of normalized input and LR=0.01 (with momentum) was the key breakthrough, jumping from ~92% to 94.28%. Normalization made the gradient landscape smoother, allowing more stable convergence.

5. **5x5 kernel was optimal**: 3x3 captures too little spatial structure; 7x7 captures more but has fewer FC features (484 vs 576) and more parameters to learn. 5x5 was the best balance.

### Architectural Limitation

The CNN is constrained to **1 convolutional kernel**, which means it can only learn a single feature detector (e.g., one type of edge or pattern). All discrimination power for the 10 digit classes must come from the spatial distribution of that single feature, interpreted by the FC layer. This fundamentally caps accuracy around 93-95%. A standard CNN with 32+ kernels would easily reach 99%+.

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
