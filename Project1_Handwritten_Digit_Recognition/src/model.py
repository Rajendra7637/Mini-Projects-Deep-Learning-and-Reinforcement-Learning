"""
Deep Neural Network (DNN) architectures for MNIST Handwritten Digit Classification.
Provides both TensorFlow/Keras and PyTorch implementations.
"""

from typing import List, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import torch
import torch.nn as nn


def build_dnn_model(
    input_shape: tuple = (784,),
    num_classes: int = 10,
    hidden_units: List[int] = [256, 128, 64],
    activation: str = "relu",
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Builds and compiles a Deep Neural Network (DNN) using TensorFlow/Keras.
    
    Architecture:
        - Input: Flattened (784,)
        - Dense(256) -> BatchNorm -> Activation -> Dropout(0.2)
        - Dense(128) -> BatchNorm -> Activation -> Dropout(0.2)
        - Dense(64)  -> BatchNorm -> Activation -> Dropout(0.2)
        - Output: Dense(10, activation='softmax')
    """
    model = models.Sequential(name="MNIST_Deep_Neural_Network")
    model.add(layers.Input(shape=input_shape))

    for idx, units in enumerate(hidden_units):
        model.add(layers.Dense(units, name=f"dense_layer_{idx+1}"))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f"batch_norm_{idx+1}"))
        if activation.lower() == "leaky_relu":
            model.add(layers.LeakyReLU(negative_slope=0.1, name=f"leaky_relu_{idx+1}"))
        else:
            model.add(layers.Activation(activation, name=f"{activation}_{idx+1}"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{idx+1}"))

    model.add(layers.Dense(num_classes, activation="softmax", name="output_layer"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


class PyTorchMNISTDNN(nn.Module):
    """
    Equivalent Deep Neural Network implementation using PyTorch nn.Module.
    """
    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        hidden_units: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2
    ):
        super(PyTorchMNISTDNN, self).__init__()
        layers_list = []
        in_features = input_dim

        for units in hidden_units:
            layers_list.append(nn.Linear(in_features, units))
            layers_list.append(nn.BatchNorm1d(units))
            layers_list.append(nn.ReLU())
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
            in_features = units

        layers_list.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input has shape (B, 1, 28, 28) or (B, 28, 28), flatten to (B, 784)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


if __name__ == "__main__":
    keras_model = build_dnn_model()
    keras_model.summary()
    
    torch_model = PyTorchMNISTDNN()
    print("\nPyTorch Model Structure:")
    print(torch_model)
