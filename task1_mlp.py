"""
Task 1: Manual Implementation of MLP for MNIST Classification
=============================================================
2-layer MLP implemented from scratch using only NumPy.

Architecture:
  Layer 1: Fully Connected (784 -> 1024) + Sigmoid
  Layer 2: Fully Connected (1024 -> 10)  + Softmax

Training:
  - Batch size: 128
  - Max epochs: 30
  - Optimizer: Nesterov Momentum SGD (momentum=0.9), LR=0.15 * 0.97^epoch
  - Loss: Cross-Entropy
  - Initialization: Xavier for sigmoid
  - Stochastic Weight Averaging (SWA) over epochs 20-30
  - Test-Time Augmentation (TTA) at inference
"""

import numpy as np
from dataloader import get_mnist_dataloaders


# ============================================================
# (4) Sigmoid Activation
# ============================================================
def sigmoid(z):
    """
    Sigmoid activation function.
    σ(z) = 1 / (1 + exp(-z))
    """
    # Clip to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# (5) Softmax Activation
# ============================================================
def softmax(z):
    """
    Softmax activation function.
    softmax(z_i) = exp(z_i) / sum(exp(z_j))
    Applied row-wise for batched input of shape (batch_size, num_classes).
    """
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================
# (6) Cross-Entropy Loss
# ============================================================
def cross_entropy(predictions, labels):
    """
    Compute cross-entropy loss.

    Parameters
    ----------
    predictions : np.ndarray, shape (batch_size, 10)
        Softmax output probabilities.
    labels : np.ndarray, shape (batch_size,)
        Ground-truth class indices (0-9).

    Returns
    -------
    loss : float
        Mean cross-entropy loss over the batch.
    """
    batch_size = predictions.shape[0]
    # Clip to avoid log(0)
    preds_clipped = np.clip(predictions, 1e-12, 1.0)
    # Select the predicted probability for the true class
    log_probs = -np.log(preds_clipped[np.arange(batch_size), labels])
    return np.mean(log_probs)


# ============================================================
# MLP Class
# ============================================================
class MLP:
    def __init__(self, input_size=784, hidden_size=512, output_size=10, learning_rate=1.0, momentum=0.9, weight_decay=0.0, label_smoothing=0.0):
        """
        Initialize the 2-layer MLP.

        (1) Xavier initialization for sigmoid: w * sqrt(2 / (fan_in + fan_out))
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

        # Layer 1: input -> hidden  (Xavier init for sigmoid)
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: hidden -> output (Xavier init for sigmoid)
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))

        # Velocity terms for momentum SGD
        self.vw1 = np.zeros_like(self.w1)
        self.vb1 = np.zeros_like(self.b1)
        self.vw2 = np.zeros_like(self.w2)
        self.vb2 = np.zeros_like(self.b2)

    # --------------------------------------------------------
    # (1) Forward Propagation
    # --------------------------------------------------------
    def forward(self, X):
        """
        Forward pass through the MLP.

        Parameters
        ----------
        X : np.ndarray, shape (batch_size, 784)
            Flattened MNIST images.

        Returns
        -------
        predictions : np.ndarray, shape (batch_size, 10)
            Softmax output probabilities.
        cache : dict
            Intermediate values needed for backpropagation.
        """
        # Layer 1: FC + Sigmoid
        z1 = X @ self.w1 + self.b1          # (B, hidden)
        a1 = sigmoid(z1)                      # (B, hidden)

        # Layer 2: FC + Softmax
        z2 = a1 @ self.w2 + self.b2          # (B, 10)
        a2 = softmax(z2)                      # (B, 10)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2, cache

    # --------------------------------------------------------
    # (2) Backward Propagation
    # --------------------------------------------------------
    def backward(self, cache, labels):
        """
        Backward pass: compute gradients and update parameters with Nesterov momentum SGD.

        Parameters
        ----------
        cache : dict
            Intermediate values from forward pass.
        labels : np.ndarray, shape (batch_size,)
            Ground-truth class indices.

        Uses chain rule to manually derive gradients for both layers.
        Gradients are averaged over the batch.
        Uses Nesterov momentum and label smoothing.
        """
        X = cache["X"]
        a1 = cache["a1"]
        a2 = cache["a2"]
        batch_size = X.shape[0]
        num_classes = a2.shape[1]

        # Label smoothing: replace hard one-hot with smoothed targets
        eps = self.label_smoothing
        one_hot = np.zeros_like(a2)
        one_hot[np.arange(batch_size), labels] = 1.0
        one_hot = one_hot * (1.0 - eps) + eps / num_classes

        # --- Layer 2 gradients ---
        # dL/dz2 = a2 - one_hot  (derivative of cross-entropy + softmax)
        dz2 = (a2 - one_hot) / batch_size          # (B, 10)

        dw2 = a1.T @ dz2                            # (hidden, 10)
        db2 = np.sum(dz2, axis=0, keepdims=True)    # (1, 10)

        # --- Layer 1 gradients ---
        # dL/da1 = dz2 @ w2^T
        da1 = dz2 @ self.w2.T                       # (B, hidden)

        # Sigmoid derivative: σ'(z) = σ(z) * (1 - σ(z)) = a1 * (1 - a1)
        dz1 = da1 * a1 * (1.0 - a1)                 # (B, hidden)

        dw1 = X.T @ dz1                              # (784, hidden)
        db1 = np.sum(dz1, axis=0, keepdims=True)     # (1, hidden)

        # --- L2 regularization (weight decay) ---
        dw1 += self.weight_decay * self.w1
        dw2 += self.weight_decay * self.w2

        # --- Update parameters via Nesterov momentum SGD ---
        vw1_prev = self.vw1.copy()
        self.vw1 = self.momentum * self.vw1 - self.lr * dw1
        self.w1 += -self.momentum * vw1_prev + (1 + self.momentum) * self.vw1

        vb1_prev = self.vb1.copy()
        self.vb1 = self.momentum * self.vb1 - self.lr * db1
        self.b1 += -self.momentum * vb1_prev + (1 + self.momentum) * self.vb1

        vw2_prev = self.vw2.copy()
        self.vw2 = self.momentum * self.vw2 - self.lr * dw2
        self.w2 += -self.momentum * vw2_prev + (1 + self.momentum) * self.vw2

        vb2_prev = self.vb2.copy()
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.b2 += -self.momentum * vb2_prev + (1 + self.momentum) * self.vb2

    # --------------------------------------------------------
    # (3) Train Function
    # --------------------------------------------------------
    def train(self, train_loader):
        """
        Train the MLP for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Iterable yielding (images, labels) batches.

        Returns
        -------
        avg_loss : float
            Average training loss over all batches.
        """
        total_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            # Forward pass
            predictions, cache = self.forward(images)

            # Compute loss
            loss = cross_entropy(predictions, labels)
            total_loss += loss
            num_batches += 1

            # Backward pass (computes gradients and updates parameters)
            self.backward(cache, labels)

        avg_loss = total_loss / num_batches
        return avg_loss

    def predict(self, X):
        """Return predicted class indices."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, test_loader):
    """
    Evaluate the model on the test set.

    Returns
    -------
    accuracy : float
        Classification accuracy (0 to 1).
    """
    correct = 0
    total = 0
    for images, labels in test_loader:
        preds = model.predict(images)
        correct += np.sum(preds == labels)
        total += len(labels)
    return correct / total


def evaluate_with_tta(model, test_loader):
    """
    Evaluate with Test-Time Augmentation (TTA).

    Predict on 9 versions of each image (original + 8 shifts from
    {-1,0,+1} x {-1,0,+1}), average the softmax outputs, then argmax.
    For MLP: unflatten to 28x28, shift, re-flatten.
    """
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),  (0, 0),  (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

    correct = 0
    total = 0
    for images, labels in test_loader:
        # images shape: (B, 784) for MLP — unflatten to (B, 28, 28)
        imgs_2d = images.reshape(-1, 28, 28)
        avg_probs = np.zeros((images.shape[0], 10))
        for dy, dx in shifts:
            shifted = np.roll(np.roll(imgs_2d, dx, axis=2), dy, axis=1)
            # Zero out wrapped edges
            if dy > 0:
                shifted[:, :dy, :] = 0
            elif dy < 0:
                shifted[:, dy:, :] = 0
            if dx > 0:
                shifted[:, :, :dx] = 0
            elif dx < 0:
                shifted[:, :, dx:] = 0
            flat = shifted.reshape(-1, 784)
            probs, _ = model.forward(flat)
            avg_probs += probs
        avg_probs /= len(shifts)
        preds = np.argmax(avg_probs, axis=1)
        correct += np.sum(preds == labels)
        total += len(labels)
    return correct / total


# ============================================================
# (7) Main Function
# ============================================================
def main():
    """
    Main function: train and test the MLP on MNIST.
    """
    np.random.seed(42)

    # Hyperparameters
    hidden_size = 1024
    learning_rate = 0.15
    momentum = 0.9
    weight_decay = 1e-4
    label_smoothing = 0.0
    epochs = 30
    batch_size = 128
    swa_start = 20  # Start SWA averaging from this epoch

    print("=" * 60)
    print("Task 1: MLP for MNIST Classification")
    print("=" * 60)
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate} (decay 0.97/epoch)")
    print(f"Momentum: {momentum} (Nesterov)")
    print(f"Weight decay: {weight_decay}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Data augmentation: random shifts (max 2px)")
    print(f"SWA: epochs {swa_start}-{epochs}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()

    # Load data (flattened for MLP, with augmentation on train)
    print("Loading MNIST data...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size, flatten=True)
    train_loader.augment = True
    train_loader.max_shift = 2
    print(f"Training samples: {train_loader.n}")
    print(f"Test samples: {test_loader.n}")
    print()

    # Global normalization (zero mean, unit variance)
    train_mean = np.mean(train_loader.images)
    train_std = np.std(train_loader.images) + 1e-8
    train_loader.images = (train_loader.images - train_mean) / train_std
    test_loader.images = (test_loader.images - train_mean) / train_std

    # Initialize model
    model = MLP(input_size=784, hidden_size=hidden_size, output_size=10,
                learning_rate=learning_rate, momentum=momentum,
                weight_decay=weight_decay, label_smoothing=label_smoothing)

    # SWA: accumulate weight averages
    swa_count = 0
    swa_w1 = np.zeros_like(model.w1)
    swa_b1 = np.zeros_like(model.b1)
    swa_w2 = np.zeros_like(model.w2)
    swa_b2 = np.zeros_like(model.b2)

    # Training loop with exponential LR decay
    print("Training...")
    lr_decay = 0.97
    for epoch in range(1, epochs + 1):
        # Exponential decay: same schedule as baseline
        model.lr = learning_rate * (lr_decay ** epoch)

        avg_loss = model.train(train_loader)

        # SWA: accumulate weights from epoch swa_start onward
        if epoch >= swa_start:
            swa_w1 += model.w1
            swa_b1 += model.b1
            swa_w2 += model.w2
            swa_b2 += model.b2
            swa_count += 1

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader)
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | LR: {model.lr:.4f} | Test Accuracy: {acc:.4f}")
        else:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | LR: {model.lr:.4f}")

    # Apply SWA averaged weights
    if swa_count > 0:
        model.w1 = swa_w1 / swa_count
        model.b1 = swa_b1 / swa_count
        model.w2 = swa_w2 / swa_count
        model.b2 = swa_b2 / swa_count
        print(f"\nSWA: averaged weights from {swa_count} epochs ({swa_start}-{epochs})")

    # Final evaluation
    print()
    print("-" * 60)
    test_accuracy = evaluate(model, test_loader)
    print(f"Final Test Accuracy (no TTA): {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Test-Time Augmentation
    tta_accuracy = evaluate_with_tta(model, test_loader)
    print(f"Final Test Accuracy (with TTA): {tta_accuracy:.4f} ({tta_accuracy * 100:.2f}%)")
    print("-" * 60)


if __name__ == "__main__":
    main()
