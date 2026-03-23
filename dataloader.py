"""
MNIST Dataloader
Loads and preprocesses MNIST data for MLP and CNN tasks.
"""

import numpy as np
import struct
import os
import urllib.request
import gzip
import ssl

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url, filepath):
    """Download a file if it doesn't exist."""
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Downloading {url} ...")
        # Bypass SSL verification for environments without proper certs
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, filepath)


def _load_images(filepath):
    """Load MNIST image file and return numpy array of shape (N, 28, 28), normalized to [0, 1]."""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols).astype(np.float64) / 255.0


def _load_labels(filepath):
    """Load MNIST label file and return numpy array of shape (N,)."""
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _download_all():
    """Download all MNIST files."""
    files = {}
    for key, url in URLS.items():
        filename = url.split("/")[-1]
        filepath = os.path.join(DATA_DIR, filename)
        _download(url, filepath)
        files[key] = filepath
    return files


class DataLoader:
    """
    Iterable dataloader that yields batches of (images, labels).

    Parameters
    ----------
    images : np.ndarray
        Image data.
    labels : np.ndarray
        Label data.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle data each epoch.
    flatten : bool
        If True, images are flattened to (batch_size, 784) for MLP.
        If False, images remain as (batch_size, 28, 28) for CNN.
    """

    def __init__(self, images, labels, batch_size=128, shuffle=True, flatten=False, augment=False, max_shift=2):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flatten = flatten
        self.augment = augment
        self.max_shift = max_shift
        self.n = len(images)

    @staticmethod
    def _shift_images(images, max_shift):
        """Apply random pixel shifts to a batch of 28x28 images using NumPy roll + zero-fill."""
        B = images.shape[0]
        shifted = images.copy()
        for i in range(B):
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)
            shifted[i] = np.roll(np.roll(images[i], dx, axis=1), dy, axis=0)
            # Zero out wrapped-around edges
            if dy > 0:
                shifted[i, :dy, :] = 0
            elif dy < 0:
                shifted[i, dy:, :] = 0
            if dx > 0:
                shifted[i, :, :dx] = 0
            elif dx < 0:
                shifted[i, :, dx:] = 0
        return shifted

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, self.n, self.batch_size):
            idx = indices[start : start + self.batch_size]
            batch_images = self.images[idx]
            if self.augment:
                batch_images = self._shift_images(batch_images, self.max_shift)
            if self.flatten:
                batch_images = batch_images.reshape(len(idx), -1)  # (B, 784)
            yield batch_images, self.labels[idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def get_mnist_dataloaders(batch_size=128, flatten=False):
    """
    Returns train and test DataLoader objects.

    Parameters
    ----------
    batch_size : int
        Batch size.
    flatten : bool
        If True, flatten images to 784-d vectors (for MLP).
        If False, keep as 28x28 (for CNN).

    Returns
    -------
    train_loader, test_loader : DataLoader, DataLoader
    """
    files = _download_all()
    train_images = _load_images(files["train_images"])
    train_labels = _load_labels(files["train_labels"])
    test_images = _load_images(files["test_images"])
    test_labels = _load_labels(files["test_labels"])

    train_loader = DataLoader(train_images, train_labels, batch_size=batch_size, shuffle=True, flatten=flatten)
    test_loader = DataLoader(test_images, test_labels, batch_size=batch_size, shuffle=False, flatten=flatten)

    return train_loader, test_loader
