"""
Evaluation script for MNIST Handwritten Digit Recognition DNN.
Generates learning curves, confusion matrix, classification report, and prediction visualizations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_mnist_data
from model import PyTorchMNISTDNN


def plot_learning_curves(history_path: str, output_path: str):
    """
    Plots training and validation loss and accuracy curves across epochs.
    """
    if not os.path.exists(history_path):
        print(f"Warning: History file not found at {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, history["loss"], "bo-", label="Training Loss", linewidth=2, markersize=5)
    axes[0].plot(epochs, history["val_loss"], "ro-", label="Validation Loss", linewidth=2, markersize=5)
    axes[0].set_title("Training and Validation Loss vs Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Accuracy Curve
    axes[1].plot(epochs, history["accuracy"], "bo-", label="Training Accuracy", linewidth=2, markersize=5)
    axes[1].plot(epochs, history["val_accuracy"], "ro-", label="Validation Accuracy", linewidth=2, markersize=5)
    axes[1].set_title("Training and Validation Accuracy vs Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Learning curves plot: {output_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    """
    Plots a 10x10 confusion matrix heatmap for MNIST digits 0-9.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
        cbar=True,
        ax=ax
    )
    ax.set_title("MNIST DNN - Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Digit", fontsize=12)
    ax.set_ylabel("Ground Truth Digit", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion matrix plot: {output_path}")


def plot_sample_predictions(x_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, output_path: str, num_samples: int = 25):
    """
    Visualizes a grid of sample test predictions highlighting correct and incorrect predictions.
    """
    fig, axes = plt.subplots(5, 5, figsize=(11, 11))
    fig.suptitle("MNIST Test Sample Predictions (Green = Correct, Red = Error)", fontsize=14, fontweight="bold")

    # Select both correct and incorrect samples to provide educational insight
    incorrect_indices = np.where(y_true != y_pred)[0]
    correct_indices = np.where(y_true == y_pred)[0]

    num_incorrect = min(5, len(incorrect_indices))
    num_correct = num_samples - num_incorrect

    np.random.seed(42)
    chosen_correct = np.random.choice(correct_indices, num_correct, replace=False)
    chosen_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False) if num_incorrect > 0 else []

    selected = np.concatenate([chosen_correct, chosen_incorrect])
    np.random.shuffle(selected)

    for i, idx in enumerate(selected[:25]):
        ax = axes[i // 5, i % 5]
        ax.imshow(x_test[idx], cmap="gray")
        ax.axis("off")

        true_lbl = y_true[idx]
        pred_lbl = y_pred[idx]
        color = "green" if true_lbl == pred_lbl else "red"

        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Sample predictions grid: {output_path}")


def evaluate_models():
    """
    Main evaluation pipeline.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "mnist_dnn.keras")
    history_path = os.path.join(base_dir, "models", "keras_training_history.json")
    output_dir = os.path.join(base_dir, "outputs")

    data = load_mnist_data()

    print("=" * 60)
    print(">>> Evaluating Keras Deep Neural Network on Test Set")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(data["x_test_flat"], data["y_test"], verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Predictions
    y_prob = model.predict(data["x_test_flat"], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Classification Report
    print("\nClassification Report (Per-Digit Performance):")
    report = classification_report(data["y_test"], y_pred, digits=4)
    print(report)

    # Save classification report to text file
    report_save_path = os.path.join(output_dir, "classification_report.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_save_path, "w") as f:
        f.write("MNIST Deep Neural Network - Test Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report)

    # Visualizations
    plot_learning_curves(history_path, os.path.join(output_dir, "learning_curves.png"))
    plot_confusion_matrix(data["y_test"], y_pred, os.path.join(output_dir, "confusion_matrix.png"))
    plot_sample_predictions(data["x_test"], data["y_test"], y_pred, os.path.join(output_dir, "sample_predictions.png"))

    # Also evaluate PyTorch model if available
    pytorch_path = os.path.join(base_dir, "models", "mnist_dnn.pth")
    if os.path.exists(pytorch_path):
        print("\n" + "=" * 60)
        print(">>> Evaluating PyTorch Deep Neural Network on Test Set")
        print("=" * 60)
        torch_model = PyTorchMNISTDNN(input_dim=784, num_classes=10)
        torch_model.load_state_dict(torch.load(pytorch_path, map_location=torch.device("cpu")))
        torch_model.eval()

        with torch.no_grad():
            x_test_t = torch.tensor(data["x_test_flat"], dtype=torch.float32)
            outputs = torch_model(x_test_t)
            _, torch_preds = torch.max(outputs, 1)
            torch_acc = (torch_preds.numpy() == data["y_test"]).mean()
            print(f"[PyTorch Model] Test Accuracy: {torch_acc * 100:.2f}%")


if __name__ == "__main__":
    evaluate_models()
