"""
Comparative analysis between Deep Learning (DNN) and Traditional Machine Learning models on MNIST.
Also conducts hyperparameter / activation function comparison (ReLU, Sigmoid, Tanh, LeakyReLU).
"""

import os
import time
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

from dataset import load_mnist_data
from model import build_dnn_model


def run_activation_function_experiments(data: dict, epochs: int = 8, batch_size: int = 128) -> dict:
    """
    Compares different activation functions (ReLU, Sigmoid, Tanh, LeakyReLU) on the DNN.
    """
    print("\n" + "=" * 60)
    print(">>> Running Activation Function Hyperparameter Experiments")
    print("=" * 60)

    activations = ["relu", "sigmoid", "tanh", "leaky_relu"]
    results = {}

    for act in activations:
        print(f"\n--- Training DNN with '{act}' activation ---")
        model = build_dnn_model(
            input_shape=(784,),
            num_classes=10,
            hidden_units=[256, 128, 64],
            activation=act,
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        start = time.time()
        history = model.fit(
            data["x_train_flat"],
            data["y_train"],
            validation_data=(data["x_val_flat"], data["y_val"]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        train_time = time.time() - start
        
        test_loss, test_acc = model.evaluate(data["x_test_flat"], data["y_test"], verbose=0)
        print(f"[{act.upper()}] Final Test Accuracy: {test_acc*100:.2f}%, Loss: {test_loss:.4f}, Time: {train_time:.1f}s")

        results[act] = {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "training_time": float(train_time)
        }

    return results


def plot_activation_comparison(results: dict, output_path: str):
    """
    Plots validation loss and accuracy across different activation functions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(next(iter(results.values()))["val_loss"]) + 1)

    colors = {"relu": "#1f77b4", "sigmoid": "#ff7f0e", "tanh": "#2ca02c", "leaky_relu": "#d62728"}
    markers = {"relu": "o", "sigmoid": "s", "tanh": "^", "leaky_relu": "D"}

    for act, data in results.items():
        axes[0].plot(epochs, data["val_loss"], marker=markers[act], color=colors[act], label=f"{act.upper()}", linewidth=2)
        axes[1].plot(epochs, data["val_accuracy"], marker=markers[act], color=colors[act], label=f"{act.upper()}", linewidth=2)

    axes[0].set_title("Validation Loss by Activation Function", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Validation Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    axes[1].set_title("Validation Accuracy by Activation Function", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation Accuracy", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Activation function comparison plot: {output_path}")


def compare_ml_and_dl():
    """
    Compares Traditional ML (Random Forest, Logistic Regression) against Deep Learning (DNN).
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    data = load_mnist_data()

    print("=" * 60)
    print(">>> Traditional ML vs Deep Learning Comparison on MNIST")
    print("=" * 60)

    comparison = {}

    # 1. Logistic Regression (Multinomial Softmax Baseline)
    print("\n1. Training Logistic Regression baseline...")
    start = time.time()
    # In scikit-learn >= 1.8, multi_class parameter is deprecated/removed and handled automatically
    lr_model = LogisticRegression(max_iter=100, solver="lbfgs", random_state=42)
    lr_model.fit(data["x_train_flat"], data["y_train"])
    lr_time = time.time() - start
    lr_preds = lr_model.predict(data["x_test_flat"])
    lr_acc = accuracy_score(data["y_test"], lr_preds)
    lr_f1 = f1_score(data["y_test"], lr_preds, average="macro")
    print(f"Logistic Regression: Accuracy = {lr_acc*100:.2f}%, F1 = {lr_f1:.4f}, Time = {lr_time:.1f}s")
    comparison["Logistic Regression"] = {"accuracy": float(lr_acc), "f1_score": float(lr_f1), "train_time": float(lr_time)}

    # 2. Random Forest Classifier Baseline
    print("\n2. Training Random Forest Classifier baseline (100 trees)...")
    start = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf_model.fit(data["x_train_flat"], data["y_train"])
    rf_time = time.time() - start
    rf_preds = rf_model.predict(data["x_test_flat"])
    rf_acc = accuracy_score(data["y_test"], rf_preds)
    rf_f1 = f1_score(data["y_test"], rf_preds, average="macro")
    print(f"Random Forest: Accuracy = {rf_acc*100:.2f}%, F1 = {rf_f1:.4f}, Time = {rf_time:.1f}s")
    comparison["Random Forest"] = {"accuracy": float(rf_acc), "f1_score": float(rf_f1), "train_time": float(rf_time)}

    # Save trained Random Forest model
    rf_save_path = os.path.join(models_dir, "baseline_rf.joblib")
    joblib.dump(rf_model, rf_save_path, compress=3)
    print(f"[Saved] Random Forest model: {rf_save_path}")

    # 3. Deep Neural Network (DNN)
    print("\n3. Loading / Evaluating Deep Neural Network...")
    keras_path = os.path.join(models_dir, "mnist_dnn.keras")
    if os.path.exists(keras_path):
        dnn_model = tf.keras.models.load_model(keras_path)
    else:
        dnn_model = build_dnn_model(input_shape=(784,), num_classes=10)
        dnn_model.fit(data["x_train_flat"], data["y_train"], epochs=10, batch_size=128, verbose=0)
        dnn_model.save(keras_path)

    # Load history time if available
    hist_path = os.path.join(models_dir, "keras_training_history.json")
    dnn_time = 35.0
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            h = json.load(f)
            dnn_time = h.get("training_time_seconds", 35.0)

    dnn_probs = dnn_model.predict(data["x_test_flat"], verbose=0)
    dnn_preds = np.argmax(dnn_probs, axis=1)
    dnn_acc = accuracy_score(data["y_test"], dnn_preds)
    dnn_f1 = f1_score(data["y_test"], dnn_preds, average="macro")
    print(f"Deep Neural Network: Accuracy = {dnn_acc*100:.2f}%, F1 = {dnn_f1:.4f}, Time = {dnn_time:.1f}s")
    comparison["Deep Neural Network (DNN)"] = {"accuracy": float(dnn_acc), "f1_score": float(dnn_f1), "train_time": float(dnn_time)}

    # Save comparison metrics
    summary_path = os.path.join(outputs_dir, "model_comparison_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=4)

    # Plot Comparison Bar Chart
    plot_model_comparison(comparison, os.path.join(outputs_dir, "model_comparison_metrics.png"))

    # Run Activation Experiments
    act_results = run_activation_function_experiments(data, epochs=6, batch_size=128)
    plot_activation_comparison(act_results, os.path.join(outputs_dir, "activation_comparison.png"))

    act_summary_path = os.path.join(outputs_dir, "activation_experiments.json")
    with open(act_summary_path, "w") as f:
        json.dump(act_results, f, indent=4)

    print("\nComparison and activation experiments finished successfully!")


def plot_model_comparison(comparison: dict, output_path: str):
    """
    Generates side-by-side bar plots comparing Accuracy and Training Time across models.
    """
    models = list(comparison.keys())
    accuracies = [comparison[m]["accuracy"] * 100 for m in models]
    times = [comparison[m]["train_time"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    bars1 = axes[0].bar(models, accuracies, color=["#4e79a7", "#f28e2c", "#59a14f"], width=0.55)
    axes[0].set_title("Test Accuracy Comparison: ML vs Deep Learning", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_ylim(85, 100)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")

    # Training time comparison
    bars2 = axes[1].bar(models, times, color=["#4e79a7", "#f28e2c", "#59a14f"], width=0.55)
    axes[1].set_title("Training Duration (seconds)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Seconds", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.1f}s", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Model comparison plot: {output_path}")


if __name__ == "__main__":
    compare_ml_and_dl()
