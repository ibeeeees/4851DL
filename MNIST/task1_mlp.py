import numpy as np
from read_MNIST import load_data

def sigmoid(x):  # manually define the sigmoid
    x = np.clip(x, -500, 500)
    return 1/(1 + np.exp(-x))

def softmax(x):  # define the softmax
    shift = x - np.max(x, axis=1, keepdims=True)
    numer = np.exp(shift)
    denom = np.sum(numer, axis=1, keepdims=True)
    return numer/denom

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr, momentum=0.9, weight_decay=1e-4):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Xavier initialization for sigmoid
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))

        # Velocity terms for Nesterov momentum
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, x):  # forward propagation to get predictions
        self.x = x
        self.Z1 = x @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        Z2 = self.A1 @ self.W2 + self.b2
        A2 = softmax(Z2)
        outputs = A2
        return outputs

    def backward(self, x, y, pred):
        # one-hot encode the labels
        N = x.shape[0]
        Y = np.zeros((N, 10))
        Y[np.arange(N), y] = 1

        # compute the gradients
        dZ2 = (pred - Y) / N
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        dW1 = x.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # L2 regularization (weight decay)
        dW1 += self.weight_decay * self.W1
        dW2 += self.weight_decay * self.W2

        # update the weights and biases via Nesterov momentum SGD
        vW1_prev = self.vW1.copy()
        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.W1 += -self.momentum * vW1_prev + (1 + self.momentum) * self.vW1

        vb1_prev = self.vb1.copy()
        self.vb1 = self.momentum * self.vb1 - self.lr * db1
        self.b1 += -self.momentum * vb1_prev + (1 + self.momentum) * self.vb1

        vW2_prev = self.vW2.copy()
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.W2 += -self.momentum * vW2_prev + (1 + self.momentum) * self.vW2

        vb2_prev = self.vb2.copy()
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.b2 += -self.momentum * vb2_prev + (1 + self.momentum) * self.vb2

    def train(self, x, y):
        N = x.shape[0]
        # call forward function
        pred = self.forward(x)
        # calculate loss
        loss = -np.mean(np.log(np.clip(pred[np.arange(N), y], 1e-12, 1.0)))
        # call backward function
        self.backward(x, y, pred)
        return loss

def evaluate_with_tta(model, test_loader, input_size):
    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        imgs = inputs.numpy().reshape(-1, 28, 28)
        labels = labels.numpy()
        avg_probs = np.zeros((imgs.shape[0], 10))
        for dy, dx in shifts:
            shifted = np.roll(np.roll(imgs, dx, axis=2), dy, axis=1)
            if dy > 0:  shifted[:, :dy, :] = 0
            elif dy < 0: shifted[:, dy:, :] = 0
            if dx > 0:  shifted[:, :, :dx] = 0
            elif dx < 0: shifted[:, :, dx:] = 0
            flat = shifted.reshape(-1, input_size)
            avg_probs += model.forward(flat)
        avg_probs /= len(shifts)
        correct += np.sum(np.argmax(avg_probs, axis=1) == labels)
        total += len(labels)
    return correct / total

def main():
    np.random.seed(42)

    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    num_epochs = 30
    output_size = 10
    hidden_size = 1024
    lr = 0.15
    momentum = 0.9
    lr_decay = 0.97
    swa_start = 20  # start averaging weights from this epoch

    model = MLP(input_size, hidden_size, output_size, lr, momentum)

    # SWA accumulators
    swa_W1 = np.zeros_like(model.W1)
    swa_b1 = np.zeros_like(model.b1)
    swa_W2 = np.zeros_like(model.W2)
    swa_b2 = np.zeros_like(model.b2)
    swa_count = 0

    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0
        model.lr = lr * (lr_decay ** (epoch + 1))  # exponential decay

        for inputs, lables in train_loader:  # define training phase for training model
            x = inputs.view(-1, input_size).numpy()
            y = lables.numpy()
            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, LR: {model.lr:.4f}")

        # SWA: accumulate weights from swa_start onward
        if (epoch + 1) >= swa_start:
            swa_W1 += model.W1
            swa_b1 += model.b1
            swa_W2 += model.W2
            swa_b2 += model.b2
            swa_count += 1

    # Apply averaged weights (SWA)
    model.W1 = swa_W1 / swa_count
    model.b1 = swa_b1 / swa_count
    model.W2 = swa_W2 / swa_count
    model.b2 = swa_b2 / swa_count
    print(f"\nSWA applied: averaged weights from epochs {swa_start}-{num_epochs}")

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the training phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred:.4f}")

    # Test-Time Augmentation (TTA): average predictions over 9 pixel shifts
    tta_accuracy = evaluate_with_tta(model, test_loader, input_size)
    print(f"Test Accuracy (TTA): {tta_accuracy:.4f}")

if __name__ == "__main__":  # Program entry
    main()
