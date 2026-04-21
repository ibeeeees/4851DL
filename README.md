# CSC4851/6851 Course Project: MNIST Classification

Manual implementation of MLP and CNN from scratch using only NumPy for handwritten digit classification on the MNIST dataset.

---

## How to Run

```bash
# Task 1: MLP
python MNIST/MLP_template.py

# Task 2: CNN
python MNIST/CNN_template.py
```

MNIST data is downloaded automatically on first run to `./data/mnist/`.

## Requirements

```bash
pip install numpy torch torchvision
```

> `torch` and `torchvision` are used **only** for the dataloader. All neural network math is implemented from scratch in NumPy.

---

## Project Structure

| File | Description |
|------|-------------|
| `MNIST/MLP_template.py` | Task 1: Multi-Layer Perceptron from scratch |
| `MNIST/CNN_template.py` | Task 2: Convolutional Neural Network from scratch |
| `MNIST/read_MNIST.py` | Shared MNIST dataloader reference |

---

## Task 1: MLP

### Architecture
```
Input (784) → FC (1024) + Sigmoid → FC (10) + Softmax → Output
```

### Implementation Details

| Component | Implementation |
|---|---|
| Weight init | Xavier: `randn * sqrt(2 / (fan_in + fan_out))` |
| Optimizer | Nesterov Momentum SGD (momentum=0.9) |
| LR schedule | Exponential decay: `0.15 * 0.97^epoch` |
| Regularization | L2 weight decay (1e-4) |
| Weight averaging | SWA over epochs 20–30 |
| Inference | Test-Time Augmentation (9-shift ensemble) |
| Batch size | 128 |
| Epochs | 30 |

### Nesterov Momentum Update
```
v_prev = v
v = momentum * v - lr * gradient
w += -momentum * v_prev + (1 + momentum) * v
```

### SWA (Stochastic Weight Averaging)
Average weights from epochs 20–30 instead of using only the final epoch. Lands in a flatter region of the loss landscape that generalizes better.

### TTA (Test-Time Augmentation)
At inference, predict on 9 shifted versions of each image ({-1,0,+1} × {-1,0,+1} pixel shifts), average the softmax outputs, then take argmax.

### Test Results

| Config | Accuracy |
|---|---|
| hidden=128, lr=0.1, plain SGD | 96.79% |
| hidden=256, lr=0.1, plain SGD | 96.90% |
| hidden=1024, Xavier, Nesterov, SWA (start=20) | 97.05% |
| hidden=1024, Xavier, Nesterov, SWA (start=15) | 96.89% |
| **hidden=1024, Xavier, Nesterov, SWA, TTA, weight_decay** | **~97.1%** |

---

## Task 2: CNN

### Architecture
```
Input (28×28) → Conv2D (1 kernel, 5×5, valid) + ReLU → Flatten (576) → FC (10) + Softmax → Output
```

### Implementation Details

| Component | Implementation |
|---|---|
| Convolution | Valid mode (no padding), vectorized via `np.lib.stride_tricks` + `np.einsum` |
| Weight init | He: `randn * sqrt(2 / fan_in)` |
| Optimizer | Nesterov Momentum SGD (momentum=0.9) |
| LR | 0.02 (constant) |
| Inference | Test-Time Augmentation (9-shift ensemble) |
| Batch size | 128 |
| Epochs | 5 |

### Conv Forward (valid mode)
```
output[b, i, j] = sum over (p, q): image[b, i+p, j+q] * K[p, q]
output shape: [batch, 28-k+1, 28-k+1]
```

### Conv Kernel Gradient
```
dK[p, q] = (1/N) * sum over (b, i, j): dZ_conv[b, i, j] * X[b, i+p, j+q]
```

### Test Results

| Config | Accuracy |
|---|---|
| kernel=5, lr=0.01, plain SGD | 89.26% |
| kernel=5, lr=0.1, plain SGD | 93.88% |
| kernel=3, lr=0.1, plain SGD | 92.10% |
| kernel=7, lr=0.1, plain SGD | 93.89% |
| kernel=5, lr=0.02, Nesterov, He init | ~92.3% |
| **kernel=5, lr=0.02, Nesterov, He init, TTA** | **~94.9%** |

### Architecture Limitation
With 1 kernel, the network learns one feature detector. All classification power comes from the spatial distribution of that single feature map into the FC layer. The realistic ceiling for this architecture is ~94–95%. A standard CNN with 32+ kernels reaches 99%+.
