"""
Dataset loader and preprocessing utilities for MNIST Handwritten Digit Recognition.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any


def load_mnist_data(val_split: float = 0.16666, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Loads MNIST dataset and preprocesses features and targets.
    
    Returns:
        Dict containing:
            - x_train, y_train: Training images and labels
            - x_val, y_val:     Validation images and labels
            - x_test, y_test:   Test images and labels
            - x_train_flat, x_val_flat, x_test_flat: Flattened (N, 784) arrays for DNN/ML
    """
    # Load dataset via tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Create validation split from training set
    np.random.seed(seed)
    num_train_full = x_train_full.shape[0]
    val_size = int(num_train_full * val_split)
    indices = np.random.permutation(num_train_full)
    
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]
    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]

    # Flatten images to 1D vectors for Fully Connected (Dense) layers
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_val_flat = x_val.reshape((x_val.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "x_train_flat": x_train_flat,
        "x_val_flat": x_val_flat,
        "x_test_flat": x_test_flat,
    }


if __name__ == "__main__":
    data = load_mnist_data()
    print("MNIST Dataset Summary:")
    print(f"Training set:   {data['x_train'].shape}, labels: {data['y_train'].shape}")
    print(f"Validation set: {data['x_val'].shape}, labels: {data['y_val'].shape}")
    print(f"Test set:       {data['x_test'].shape}, labels: {data['y_test'].shape}")
    print(f"Flattened train features: {data['x_train_flat'].shape}")
    print(f"Pixel range: min={data['x_train'].min():.1f}, max={data['x_train'].max():.1f}")
