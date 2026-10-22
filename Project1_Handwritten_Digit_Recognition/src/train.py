"""
Training script for MNIST Deep Neural Network (DNN).
Trains TensorFlow/Keras and PyTorch models, saves weights, and exports training history.
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from dataset import load_mnist_data
from model import build_dnn_model, PyTorchMNISTDNN


def train_keras_dnn(
    data: dict,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    model_save_path: str = "../models/mnist_dnn.keras"
) -> tf.keras.callbacks.History:
    """
    Trains the Keras DNN model on MNIST.
    """
    print("=" * 60)
    print(">>> Starting TensorFlow/Keras DNN Training on MNIST")
    print("=" * 60)

    model = build_dnn_model(
        input_shape=(784,),
        num_classes=10,
        hidden_units=[256, 128, 64],
        activation="relu",
        dropout_rate=0.2,
        learning_rate=learning_rate
    )
    model.summary()

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    start_time = time.time()
    history = model.fit(
        data["x_train_flat"],
        data["y_train"],
        validation_data=(data["x_val_flat"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    elapsed_time = time.time() - start_time
    print(f"\n[Keras] Training completed in {elapsed_time:.2f} seconds.")

    # Save trained Keras model
    model.save(model_save_path)
    print(f"[Keras] Model saved successfully to {model_save_path}")

    # Save history as JSON
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict["training_time_seconds"] = elapsed_time
    history_save_path = os.path.join(os.path.dirname(model_save_path), "keras_training_history.json")
    with open(history_save_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[Keras] History saved to {history_save_path}")

    return history


def train_pytorch_dnn(
    data: dict,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    model_save_path: str = "../models/mnist_dnn.pth"
) -> dict:
    """
    Trains the PyTorch DNN model on MNIST.
    """
    print("\n" + "=" * 60)
    print(">>> Starting PyTorch DNN Training on MNIST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Using device: {device}")

    train_tensor_x = torch.tensor(data["x_train_flat"], dtype=torch.float32)
    train_tensor_y = torch.tensor(data["y_train"], dtype=torch.long)
    val_tensor_x = torch.tensor(data["x_val_flat"], dtype=torch.float32)
    val_tensor_y = torch.tensor(data["y_val"], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_tensor_x, train_tensor_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y), batch_size=batch_size, shuffle=False)

    model = PyTorchMNISTDNN(input_dim=784, num_classes=10, hidden_units=[256, 128, 64], dropout_rate=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation phase
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} - "
              f"loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} - "
              f"val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

    elapsed_time = time.time() - start_time
    history["training_time_seconds"] = elapsed_time
    print(f"\n[PyTorch] Best Model saved to {model_save_path} in {elapsed_time:.2f} seconds.")

    torch_history_path = os.path.join(os.path.dirname(model_save_path), "pytorch_training_history.json")
    with open(torch_history_path, "w") as f:
        json.dump(history, f, indent=4)

    return history


def main():
    data = load_mnist_data()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    keras_path = os.path.join(base_dir, "models", "mnist_dnn.keras")
    pytorch_path = os.path.join(base_dir, "models", "mnist_dnn.pth")

    train_keras_dnn(data, epochs=12, batch_size=128, model_save_path=keras_path)
    train_pytorch_dnn(data, epochs=12, batch_size=128, model_save_path=pytorch_path)


if __name__ == "__main__":
    main()
