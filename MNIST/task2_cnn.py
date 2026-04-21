import numpy as np
from read_MNIST import load_data

# ===================== Utility Functions ===================== #

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    shift = x - np.max(x, axis=1, keepdims=True)
    numer = np.exp(shift)
    denom = np.sum(numer, axis=1, keepdims=True)
    return numer/denom


# ===================== CNN Structure ===================== #
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.kernel_size = kernel_size

        # He initialization for ReLU: sqrt(2 / fan_in)
        fan_in = kernel_size * kernel_size
        self.K = np.random.randn(kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b_conv = np.zeros(1)
        self.fc_input_size = (input_size - kernel_size + 1) ** 2
        self.W_fc = np.random.randn(self.fc_input_size, 10) * np.sqrt(2.0 / self.fc_input_size)
        self.b_fc = np.zeros(10)

        # Velocity terms for Nesterov momentum
        self.vK = np.zeros_like(self.K)
        self.vb_conv = np.zeros(1)
        self.vW_fc = np.zeros_like(self.W_fc)
        self.vb_fc = np.zeros_like(self.b_fc)

    def forward(self, x):
        """ Forward propagation """
        N = x.shape[0]
        self.x = x
        k = self.kernel_size
        H_out = x.shape[1] - k + 1

        # Vectorized convolution using stride tricks
        strides = x.strides
        patches = np.lib.stride_tricks.as_strided(
            x,
            shape=(N, H_out, H_out, k, k),
            strides=(strides[0], strides[1], strides[2], strides[1], strides[2])
        )
        self.Z_conv = np.einsum("bijkl,kl->bij", patches, self.K) + self.b_conv

        self.A_conv = relu(self.Z_conv)
        self.A_flat = self.A_conv.reshape(N, -1)

        Z_fc = self.A_flat @ self.W_fc + self.b_fc
        outputs = softmax(Z_fc)

        return outputs

    def backward(self, x, y, pred):
        """ Backward propagation """
        # 1. one-hot encode the labels
        N = x.shape[0]
        Y = np.zeros((N, 10))
        Y[np.arange(N), y] = 1

        # 2. Calculate softmax cross-entropy loss gradient
        dZ_fc = (pred - Y) / N

        # 3. Calculate fully connected layer gradient
        dW_fc = self.A_flat.T @ dZ_fc
        db_fc = np.sum(dZ_fc, axis=0)
        dA_flat = dZ_fc @ self.W_fc.T

        # 4. Backpropagate through ReLU
        k = self.kernel_size
        H_out = x.shape[1] - k + 1
        dA_conv = dA_flat.reshape(N, H_out, H_out)
        dZ_conv = dA_conv * (self.Z_conv > 0)

        # 5. Calculate convolution kernel gradient (vectorized)
        strides = x.strides
        patches = np.lib.stride_tricks.as_strided(
            x,
            shape=(N, H_out, H_out, k, k),
            strides=(strides[0], strides[1], strides[2], strides[1], strides[2])
        )
        dK = np.einsum("bijkl,bij->kl", patches, dZ_conv) / N
        db_conv = np.sum(dZ_conv) / N

        # 6. Update parameters via Nesterov momentum SGD
        vK_prev = self.vK.copy()
        self.vK = self.momentum * self.vK - self.lr * dK
        self.K += -self.momentum * vK_prev + (1 + self.momentum) * self.vK

        vb_prev = self.vb_conv.copy()
        self.vb_conv = self.momentum * self.vb_conv - self.lr * db_conv
        self.b_conv += -self.momentum * vb_prev + (1 + self.momentum) * self.vb_conv

        vW_prev = self.vW_fc.copy()
        self.vW_fc = self.momentum * self.vW_fc - self.lr * dW_fc
        self.W_fc += -self.momentum * vW_prev + (1 + self.momentum) * self.vW_fc

        vb_fc_prev = self.vb_fc.copy()
        self.vb_fc = self.momentum * self.vb_fc - self.lr * db_fc
        self.b_fc += -self.momentum * vb_fc_prev + (1 + self.momentum) * self.vb_fc
        

    def train(self, x, y):
        N = x.shape[0]
        # call forward function
        pred = self.forward(x)
        # calculate loss
        loss = -np.mean(np.log(np.clip(pred[np.arange(N), y], 1e-12, 1.0)))
        # call backward function
        self.backward(x, y, pred)
        return loss

# ===================== Training Process ===================== #
def evaluate_with_tta(model, test_loader):
    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    bg_fill = np.full((28, 28), -1.0)  # normalized background (pixel=0 maps to -1)
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        imgs = inputs.squeeze(1).numpy()
        labels = labels.numpy()
        avg_probs = np.zeros((imgs.shape[0], 10))
        for dy, dx in shifts:
            shifted = np.roll(np.roll(imgs, dx, axis=2), dy, axis=1)
            if dy > 0:  shifted[:, :dy, :] = bg_fill[:dy, :]
            elif dy < 0: shifted[:, dy:, :] = bg_fill[dy:, :]
            if dx > 0:  shifted[:, :, :dx] = bg_fill[:, :dx]
            elif dx < 0: shifted[:, :, dx:] = bg_fill[:, dx:]
            avg_probs += model.forward(shifted)
        avg_probs /= len(shifts)
        correct += np.sum(np.argmax(avg_probs, axis=1) == labels)
        total += len(labels)
    return correct / total

def main():
    np.random.seed(42)

    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28
    num_epochs = 5 # dont change
    num_filters = 1 #dont change
    kernel_size = 5
    lr = 0.02
    fc_output_size = (input_size - kernel_size + 1) ** 2
    momentum = 0.9

    model = CNN(input_size, num_filters, kernel_size, fc_output_size, lr, momentum)
    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, lables in train_loader:  # define training phase for training model
            x = inputs.squeeze(1).numpy()
            y = lables.numpy()
            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.squeeze(1).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the training phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred:.4f}")

    # Test-Time Augmentation (TTA): average predictions over 9 pixel shifts
    tta_accuracy = evaluate_with_tta(model, test_loader)
    print(f"Test Accuracy (TTA): {tta_accuracy:.4f}")

if __name__ == "__main__":
    main()
