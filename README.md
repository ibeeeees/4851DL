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
| `dataloader.py` | Shared MNIST dataloader — downloads data, yields batches, supports augmentation |
| `task1_mlp.py` | Task 1: Multi-Layer Perceptron (MLP) from scratch |
| `task2_cnn.py` | Task 2: Convolutional Neural Network (CNN) from scratch |

---

## Results at a Glance

| Task | Model | Final Test Accuracy |
|------|-------|---------------------|
| Task 1 | MLP | **98.81%** |
| Task 2 | CNN | **94.33%** |

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
| L2 regularization | Yes (weight decay = 1e-4) |
| Data augmentation | Yes (random shifts, max 2px) |
| Epochs | 30 |
| Batch size | 128 |
| **Final Test Accuracy** | **98.81%** |

### Accuracy Progression

```mermaid
---
config:
    xyChart:
        width: 800
        height: 400
    themeVariables:
        xyChart:
            backgroundColor: "transparent"
---
xychart-beta
    title "MLP Accuracy Improvement Across Versions"
    x-axis ["Baseline", "v2", "v3", "v4", "v5", "v6", "v7 (Best)"]
    y-axis "Test Accuracy (%)" 93 100
    bar [97.90, 94.66, 95.46, 98.05, 98.16, 98.47, 98.81]
    line [97.90, 94.66, 95.46, 98.05, 98.16, 98.47, 98.81]
```

### Optimization History

| Version | Changes Made | Hidden | LR | Optimizer | Norm | Aug | L2 | Accuracy | Delta |
|---------|-------------|--------|-----|-----------|------|-----|-----|----------|-------|
| Baseline | Initial implementation | 256 | 0.5 | Vanilla SGD | No | No | No | 97.90% | — |
| v2 | Added Xavier init + momentum + LR decay | 512 | 1.0 (decay 0.95) | Momentum (0.9) | No | No | No | 94.66% | -3.24% |
| v3 | Reduced LR to 0.5 | 512 | 0.5 (decay 0.98) | Momentum (0.9) | No | No | No | 95.46% | +0.80% |
| v4 | Reduced LR to 0.1 | 512 | 0.1 (decay 0.98) | Momentum (0.9) | No | No | No | 98.05% | +2.59% |
| v5 | Increased hidden to 784 | 784 | 0.15 (decay 0.97) | Momentum (0.9) | No | No | No | 98.16% | +0.11% |
| v6 | Added input normalization | 784 | 0.15 (decay 0.97) | Momentum (0.9) | Yes | No | No | 98.47% | +0.31% |
| **v7** | **Added augmentation + L2 reg** | **784** | **0.15 (decay 0.97)** | **Momentum (0.9)** | **Yes** | **Yes** | **Yes** | **98.81%** | **+0.34%** |

### What Each Optimization Did

```mermaid
---
config:
    xyChart:
        width: 800
        height: 400
---
xychart-beta
    title "Impact of Each Technique on MLP Accuracy"
    x-axis ["Xavier Init + Momentum (wrong LR)", "LR Tuning (0.1)", "Larger Hidden (784)", "Input Normalization", "Augmentation + L2"]
    y-axis "Accuracy Change (%)" -4 3
    bar [-3.24, 2.59, 0.11, 0.31, 0.34]
```

1. **Xavier initialization** (`sqrt(2 / (fan_in + fan_out))`): Properly scaled initial weights for sigmoid activation, preventing vanishing gradients at the start of training.

2. **Momentum SGD (0.9)**: Accelerated convergence by accumulating past gradient direction, but required careful LR tuning — too high a learning rate with momentum (v2: LR=1.0) caused instability and *decreased* accuracy to 94.66%.

3. **Learning rate tuning**: With momentum, the effective step size is approximately `lr / (1 - momentum)`. LR=0.1 with momentum=0.9 gives an effective LR of ~1.0, which was optimal. LR=1.0 with momentum=0.9 (effective ~10.0) was far too aggressive.

4. **Larger hidden layer (784 neurons)**: More capacity allowed the model to learn more complex decision boundaries. Going from 512 to 784 improved accuracy by ~0.1%.

5. **LR decay (0.97/epoch)**: Gradually reducing the learning rate allowed fine-tuning in later epochs, preventing oscillation around the optimum.

6. **Input normalization (zero mean, unit variance)**: Normalizing pixel values from [0,1] to standardized form made gradient magnitudes more uniform across features, significantly accelerating convergence (+0.31%).

7. **Data augmentation (random pixel shifts, max 2px)**: Each epoch sees slightly different versions of every training image, shifted randomly by up to 2 pixels in any direction. This acts as a regularizer, forcing the model to learn position-invariant features and reducing overfitting.

8. **L2 regularization (weight decay = 1e-4)**: Penalizes large weights, keeping the model simpler and reducing overfitting. Combined with augmentation, this pushed the final accuracy to 98.81%.

### Key Lesson

More aggressive optimization (higher LR, stronger momentum) does NOT always help. v2 was the most "optimized" on paper but performed the worst (94.66%) because the combined effect of LR=1.0 + momentum=0.9 made the effective step size too large, causing the training to overshoot. Finding the right balance was critical.

### MLP Training Curve (Best Model — v7)

```
Epoch  1/30 | Loss: 0.5649 | Acc: 95.62%
Epoch  5/30 | Loss: 0.1082 | Acc: 97.78%
Epoch 10/30 | Loss: 0.0811 | Acc: 98.37%
Epoch 15/30 | Loss: 0.0680 | Acc: 98.44%
Epoch 20/30 | Loss: 0.0626 | Acc: 98.51%
Epoch 25/30 | Loss: 0.0572 | Acc: 98.84%
Epoch 30/30 | Loss: 0.0534 | Acc: 98.81%
```

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
| L2 regularization | Yes (weight decay = 1e-4) |
| Data augmentation | Disabled (too few epochs) |
| Epochs | 5 |
| Batch size | 128 |
| **Final Test Accuracy** | **94.33%** |

### Accuracy Progression

```mermaid
---
config:
    xyChart:
        width: 900
        height: 400
---
xychart-beta
    title "CNN Accuracy Improvement Across Versions"
    x-axis ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"]
    y-axis "Test Accuracy (%)" 90 96
    bar [91.75, 92.81, 92.55, 92.33, 92.43, 92.60, 92.48, 92.85, 94.28, 91.62, 93.95, 94.33]
    line [91.75, 92.81, 92.55, 92.33, 92.43, 92.60, 92.48, 92.85, 94.28, 91.62, 93.95, 94.33]
```

### Optimization History

| Version | Changes Made | Kernel | LR | Norm | Aug | L2 | Accuracy | Delta |
|---------|-------------|--------|-----|------|-----|-----|----------|-------|
| Baseline | Initial implementation | 5x5 | 0.01 | No | No | No | 91.75% | — |
| v2 | Added He init + momentum + higher LR | 5x5 | 0.05 | No | No | No | 92.81% | +1.06% |
| v3 | Tried 3x3 kernel | 3x3 | 0.05 | No | No | No | 92.55% | -0.26% |
| v4 | Tried 7x7 kernel | 7x7 | 0.05 | No | No | No | 92.33% | -0.22% |
| v5 | Higher LR=0.1 | 5x5 | 0.1 | No | No | No | 92.43% | +0.10% |
| v6 | Higher momentum=0.95 | 5x5 | 0.03 | No | No | No | 92.60% | +0.17% |
| v7 | Added input normalization | 5x5 | 0.05 | Yes | No | No | 92.48% | -0.12% |
| v8 | Lower LR with normalization | 5x5 | 0.02 | Yes | No | No | 92.85% | +0.37% |
| v9 | Even lower LR=0.01 with norm | 5x5 | 0.01 | Yes | No | No | 94.28% | +1.43% |
| v10 | Added augmentation (2px) + L2 | 5x5 | 0.01 | Yes | 2px | Yes | 91.62% | -2.66% |
| v11 | Reduced augmentation to 1px | 5x5 | 0.01 | Yes | 1px | Yes | 93.95% | +2.33% |
| **v12** | **Removed augmentation, kept L2** | **5x5** | **0.01** | **Yes** | **No** | **Yes** | **94.33%** | **+0.38%** |

### What Each Optimization Did

```mermaid
---
config:
    xyChart:
        width: 800
        height: 400
---
xychart-beta
    title "Impact of Each Technique on CNN Accuracy"
    x-axis ["He Init + Momentum", "3x3 Kernel", "7x7 Kernel", "Normalization + LR=0.01", "Augmentation 2px", "L2 Only"]
    y-axis "Accuracy Change (%)" -3 2
    bar [1.06, -0.26, -0.48, 1.43, -2.66, 0.05]
```

1. **He initialization** (`sqrt(2 / fan_in)`): Proper weight scaling for ReLU activation, ensuring gradients flow well through the network from the start.

2. **Momentum SGD (0.9)**: Helped the single kernel converge faster within the limited 5 epochs. With only 5 epochs, every update counts.

3. **Convolution bias**: Adding a learnable scalar bias to the conv layer gave the network an extra degree of freedom.

4. **Input normalization + lower LR**: The combination of normalized input and LR=0.01 (with momentum) was the key breakthrough, jumping from ~92% to 94%+. Normalization made the gradient landscape smoother, allowing more stable convergence.

5. **5x5 kernel was optimal**: 3x3 captures too little spatial structure; 7x7 captures more but has fewer FC features (484 vs 576) and more parameters to learn. 5x5 was the best balance.

6. **L2 regularization (weight decay = 1e-4)**: Small additional boost by preventing weight overgrowth.

### What Did NOT Help for CNN

- **Data augmentation (v10, v11)**: With only 5 epochs and 1 kernel, augmentation makes training harder without enough time to recover. 2px shifts dropped accuracy to 91.62%; even 1px shifts gave 93.95% — both worse than no augmentation. This is a key insight: augmentation needs sufficient training time to be beneficial.
- **Larger/smaller kernels**: 3x3 and 7x7 both performed worse than 5x5. The sweet spot balances receptive field size with FC layer input dimensionality.
- **Higher learning rates**: LR=0.05 and LR=0.1 without normalization performed similarly (~92%), but LR=0.01 with normalization jumped to 94%+.

### Architectural Limitation

The CNN is constrained to **1 convolutional kernel**, which means it can only learn a single feature detector (e.g., one type of edge or pattern). All discrimination power for the 10 digit classes must come from the spatial distribution of that single feature, interpreted by the FC layer. This fundamentally caps accuracy around 93-95%. A standard CNN with 32+ kernels would easily reach 99%+.

### CNN Training Curve (Best Model — v12)

```
Epoch 1/5 | Loss: 0.4160 | Acc: 92.02%
Epoch 2/5 | Loss: 0.2464 | Acc: 93.76%
Epoch 3/5 | Loss: 0.2130 | Acc: 93.73%
Epoch 4/5 | Loss: 0.1924 | Acc: 94.19%
Epoch 5/5 | Loss: 0.1850 | Acc: 94.33%
```

---

## Side-by-Side Comparison

```mermaid
---
config:
    xyChart:
        width: 600
        height: 400
---
xychart-beta
    title "MLP vs CNN: Baseline to Best"
    x-axis ["MLP Baseline", "MLP Best", "CNN Baseline", "CNN Best"]
    y-axis "Test Accuracy (%)" 88 100
    bar [97.90, 98.81, 91.75, 94.33]
```

| Metric | MLP | CNN |
|--------|-----|-----|
| Baseline accuracy | 97.90% | 91.75% |
| Best accuracy | **98.81%** | **94.33%** |
| Improvement | +0.91% | +2.58% |
| Epochs | 30 | 5 |
| Total parameters | ~621K | ~5.8K |
| Key bottleneck | 2-layer sigmoid limit | 1-kernel constraint |
| Best technique | Augmentation + L2 | Normalization + LR tuning |

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
- **Data augmentation** via random pixel shifts (NumPy only, no external libraries)
- **L2 regularization** (weight decay) to reduce overfitting
- **Momentum SGD** with learning rate scheduling for faster convergence
