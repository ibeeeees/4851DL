"""
Task 2: Manual Implementation of CNN for MNIST Classification
=============================================================
2-layer CNN implemented from scratch using only NumPy.

Architecture:
  Layer 1: Convolutional Layer (1 kernel, 5x5, valid mode) + ReLU
  Layer 2: Fully Connected (flattened conv output -> 10) + Softmax

Training:
  - Batch size: 128
  - Max epochs: 5
  - Optimizer: Momentum SGD
  - Loss: Cross-Entropy
"""

import numpy as np
from dataloader import get_mnist_dataloaders


# ============================================================
# (4) ReLU Activation
# ============================================================
def relu(z):
    """
    ReLU activation function.
    ReLU(z) = max(0, z)
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU.
    ReLU'(z) = 1 if z > 0, else 0
    """
    return (z > 0).astype(np.float64)


# ============================================================
# (5) Softmax Activation
# ============================================================
def softmax(z):
    """
    Softmax activation function applied row-wise.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================
# (6) Cross-Entropy Loss
# ============================================================
def cross_entropy(predictions, labels):
    """
    Compute mean cross-entropy loss over the batch.
    """
    batch_size = predictions.shape[0]
    preds_clipped = np.clip(predictions, 1e-12, 1.0)
    log_probs = -np.log(preds_clipped[np.arange(batch_size), labels])
    return np.mean(log_probs)


# ============================================================
# 2D Convolution helpers (valid mode, single kernel)
# ============================================================
def conv2d_forward(images, kernel):
    """
    Perform valid-mode 2D convolution with a single kernel.

    Parameters
    ----------
    images : np.ndarray, shape (B, H, W)
        Batch of input images.
    kernel : np.ndarray, shape (kH, kW)
        Convolution kernel.

    Returns
    -------
    output : np.ndarray, shape (B, H - kH + 1, W - kW + 1)
        Convolution output.
    """
    B, H, W = images.shape
    kH, kW = kernel.shape
    outH = H - kH + 1
    outW = W - kW + 1

    # Use im2col-style vectorization for speed
    strides = images.strides
    patches = np.lib.stride_tricks.as_strided(
        images,
        shape=(B, outH, outW, kH, kW),
        strides=(strides[0], strides[1], strides[2], strides[1], strides[2]),
    )
    output = np.einsum("bijkl,kl->bij", patches, kernel)
    return output


def conv2d_kernel_gradient(images, d_out, kernel_shape):
    """
    Compute the gradient of the loss w.r.t. the convolution kernel.

    Parameters
    ----------
    images : np.ndarray, shape (B, H, W)
        Input images to the conv layer.
    d_out : np.ndarray, shape (B, outH, outW)
        Gradient of loss w.r.t. conv output.
    kernel_shape : tuple (kH, kW)

    Returns
    -------
    d_kernel : np.ndarray, shape (kH, kW)
        Gradient averaged over the batch.
    """
    B, H, W = images.shape
    kH, kW = kernel_shape
    outH = H - kH + 1
    outW = W - kW + 1

    strides = images.strides
    patches = np.lib.stride_tricks.as_strided(
        images,
        shape=(B, outH, outW, kH, kW),
        strides=(strides[0], strides[1], strides[2], strides[1], strides[2]),
    )
    d_kernel = np.einsum("bijkl,bij->kl", patches, d_out)
    return d_kernel


# ============================================================
# CNN Class
# ============================================================
class CNN:
    def __init__(self, kernel_size=5, output_size=10, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        """
        Initialize the 2-layer CNN.

        Architecture:
          Conv(1 kernel, kernel_size x kernel_size, valid) + ReLU
          FC(flattened -> 10) + Softmax
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size

        # Convolution kernel: He initialization for ReLU
        fan_in = kernel_size * kernel_size
        self.kernel = np.random.randn(kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b_conv = 0.0  # scalar bias for the single feature map

        # After valid convolution on 28x28 with kernel_size:
        conv_out_dim = 28 - kernel_size + 1
        self.fc_input_size = conv_out_dim * conv_out_dim

        # Fully connected layer: He initialization
        self.w_fc = np.random.randn(self.fc_input_size, output_size) * np.sqrt(2.0 / self.fc_input_size)
        self.b_fc = np.zeros((1, output_size))

        # Velocity terms for momentum SGD
        self.v_kernel = np.zeros_like(self.kernel)
        self.v_b_conv = 0.0
        self.v_w_fc = np.zeros_like(self.w_fc)
        self.v_b_fc = np.zeros_like(self.b_fc)

    # --------------------------------------------------------
    # (1) Forward Propagation
    # --------------------------------------------------------
    def forward(self, X):
        """
        Forward pass through the CNN.

        Parameters
        ----------
        X : np.ndarray, shape (batch_size, 28, 28)
            MNIST images.

        Returns
        -------
        predictions : np.ndarray, shape (batch_size, 10)
            Softmax output probabilities.
        cache : dict
            Intermediate values for backpropagation.
        """
        B = X.shape[0]

        # Layer 1: Convolution + bias + ReLU
        conv_out = conv2d_forward(X, self.kernel) + self.b_conv  # (B, outH, outW)
        relu_out = relu(conv_out)                                  # (B, outH, outW)

        # Flatten for FC layer
        flat = relu_out.reshape(B, -1)                             # (B, fc_input_size)

        # Layer 2: FC + Softmax
        z_fc = flat @ self.w_fc + self.b_fc                        # (B, 10)
        a_fc = softmax(z_fc)                                       # (B, 10)

        cache = {
            "X": X,
            "conv_out": conv_out,
            "relu_out": relu_out,
            "flat": flat,
            "z_fc": z_fc,
            "a_fc": a_fc,
        }
        return a_fc, cache

    # --------------------------------------------------------
    # (2) Backward Propagation
    # --------------------------------------------------------
    def backward(self, cache, labels):
        """
        Backward pass: compute gradients and update parameters.

        Uses chain rule to manually derive gradients for both layers.
        Gradients are averaged over the batch.
        """
        X = cache["X"]
        conv_out = cache["conv_out"]
        flat = cache["flat"]
        a_fc = cache["a_fc"]
        B = X.shape[0]

        # One-hot encode labels
        one_hot = np.zeros_like(a_fc)
        one_hot[np.arange(B), labels] = 1.0

        # --- FC layer gradients ---
        dz_fc = (a_fc - one_hot) / B                  # (B, 10)

        dw_fc = flat.T @ dz_fc                         # (fc_input_size, 10)
        db_fc = np.sum(dz_fc, axis=0, keepdims=True)   # (1, 10)

        # --- Backprop through flatten ---
        d_flat = dz_fc @ self.w_fc.T                    # (B, fc_input_size)
        d_relu_out = d_flat.reshape(conv_out.shape)     # (B, outH, outW)

        # --- ReLU gradient ---
        d_conv_out = d_relu_out * relu_derivative(conv_out)  # (B, outH, outW)

        # --- Conv kernel gradient ---
        d_kernel = conv2d_kernel_gradient(X, d_conv_out, self.kernel.shape)
        d_b_conv = np.sum(d_conv_out)

        # --- L2 regularization (weight decay) ---
        dw_fc += self.weight_decay * self.w_fc
        d_kernel += self.weight_decay * self.kernel

        # --- Update parameters via momentum SGD ---
        self.v_kernel = self.momentum * self.v_kernel - self.lr * d_kernel
        self.v_b_conv = self.momentum * self.v_b_conv - self.lr * d_b_conv
        self.v_w_fc = self.momentum * self.v_w_fc - self.lr * dw_fc
        self.v_b_fc = self.momentum * self.v_b_fc - self.lr * db_fc

        self.kernel += self.v_kernel
        self.b_conv += self.v_b_conv
        self.w_fc += self.v_w_fc
        self.b_fc += self.v_b_fc

    # --------------------------------------------------------
    # (3) Train Function
    # --------------------------------------------------------
    def train(self, train_loader):
        """
        Train the CNN for one epoch.

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

            # Backward pass
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
    """Evaluate the model on the test set and return accuracy."""
    correct = 0
    total = 0
    for images, labels in test_loader:
        preds = model.predict(images)
        correct += np.sum(preds == labels)
        total += len(labels)
    return correct / total


# ============================================================
# (7) Main Function
# ============================================================
def main():
    """
    Main function: train and test the CNN on MNIST.
    """
    np.random.seed(42)

    # Hyperparameters
    kernel_size = 5
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 5
    batch_size = 128

    print("=" * 60)
    print("Task 2: CNN for MNIST Classification")
    print("=" * 60)
    print(f"Kernel size: {kernel_size}x{kernel_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Weight decay: {weight_decay}")
    print(f"Data augmentation: disabled (too few epochs)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()

    # Load data (NOT flattened for CNN, with augmentation on train)
    print("Loading MNIST data...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size, flatten=False)
    # Note: augmentation disabled for CNN — with only 5 epochs and 1 kernel,
    # the model doesn't have enough capacity/time to benefit from it
    train_loader.augment = False
    print(f"Training samples: {train_loader.n}")
    print(f"Test samples: {test_loader.n}")
    print()

    # Normalize input data (zero mean, unit variance) for better gradient flow
    train_mean = np.mean(train_loader.images)
    train_std = np.std(train_loader.images) + 1e-8
    train_loader.images = (train_loader.images - train_mean) / train_std
    test_loader.images = (test_loader.images - train_mean) / train_std

    # Initialize model
    model = CNN(kernel_size=kernel_size, output_size=10, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)
    print(f"Conv output size: {28 - kernel_size + 1}x{28 - kernel_size + 1}")
    print(f"FC input size: {model.fc_input_size}")
    print()

    # Training loop
    print("Training...")
    for epoch in range(1, epochs + 1):
        avg_loss = model.train(train_loader)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {acc:.4f}")

    # Final evaluation
    print()
    print("-" * 60)
    test_accuracy = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("-" * 60)


if __name__ == "__main__":
    main()
