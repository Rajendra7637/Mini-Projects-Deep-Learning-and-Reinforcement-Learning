"""
Script to generate well-documented Jupyter Notebooks for Project 1 and Project 2.
"""

import os
import nbformat as nbf

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def create_project1_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# COER UNIVERSITY, ROORKEE, UTTARAKHAND
## Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning
### MCA III Semester (Session 2026–2027)

---
# Project 1: Handwritten Digit Recognition using Neural Networks

**Objectives:**
1. Understand neural network architectures and backpropagation.
2. Build intelligent digit classification systems using TensorFlow/Keras and PyTorch.
3. Compare Traditional Machine Learning (Logistic Regression, Random Forest) with Deep Learning (DNN).
4. Train and evaluate models, plot learning curves, save trained weights, and tune hyperparameters (activation functions, learning rate).
"""),
        nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch

# Ensure local source modules are discoverable
sys.path.append(os.path.abspath('../src'))
from dataset import load_mnist_data
from model import build_dnn_model, PyTorchMNISTDNN

print(f"TensorFlow Version: {tf.__version__}")
print(f"PyTorch Version:    {torch.__version__}")
"""),
        nbf.v4.new_markdown_cell("""## 1. Dataset Loading and Exploratory Data Analysis (EDA)
The MNIST dataset contains 70,000 grayscale images of handwritten digits (28x28 pixels).
We normalize all pixel values from [0, 255] to [0.0, 1.0] and partition them into train, validation, and test splits.
"""),
        nbf.v4.new_code_cell("""data = load_mnist_data()
print(f"Training samples:   {data['x_train'].shape[0]} images, shape: {data['x_train'].shape[1:]}")
print(f"Validation samples: {data['x_val'].shape[0]} images")
print(f"Test samples:       {data['x_test'].shape[0]} images")

# Visualize 10 sample digits from training set
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(data['x_train'][i], cmap='gray')
    axes[i].set_title(f"Label: {data['y_train'][i]}")
    axes[i].axis('off')
plt.suptitle("Sample MNIST Handwritten Digits", fontsize=14, y=1.05)
plt.show()
"""),
        nbf.v4.new_markdown_cell("""## 2. Deep Neural Network Architecture
We construct a Deep Neural Network with:
- **Input:** 784 flattened dimensions
- **Layer 1:** Dense(256) + Batch Normalization + ReLU + Dropout(0.2)
- **Layer 2:** Dense(128) + Batch Normalization + ReLU + Dropout(0.2)
- **Layer 3:** Dense(64) + Batch Normalization + ReLU + Dropout(0.2)
- **Output:** Dense(10) with Softmax activation
"""),
        nbf.v4.new_code_cell("""model = build_dnn_model(
    input_shape=(784,),
    num_classes=10,
    hidden_units=[256, 128, 64],
    activation='relu',
    dropout_rate=0.2,
    learning_rate=0.001
)
model.summary()
"""),
        nbf.v4.new_markdown_cell("""## 3. Training the Deep Neural Network
We utilize Adam optimizer, Sparse Categorical Crossentropy loss, and EarlyStopping callbacks.
"""),
        nbf.v4.new_code_cell("""from train import train_keras_dnn

history = train_keras_dnn(
    data,
    epochs=12,
    batch_size=128,
    model_save_path='../models/mnist_dnn.keras'
)
"""),
        nbf.v4.new_markdown_cell("""## 4. Evaluation and Learning Curves
Evaluating test accuracy and visualizing loss and accuracy curves across training epochs.
"""),
        nbf.v4.new_code_cell("""from evaluate import plot_learning_curves, plot_confusion_matrix, plot_sample_predictions
from sklearn.metrics import classification_report

test_loss, test_acc = model.evaluate(data['x_test_flat'], data['y_test'], verbose=0)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print(f"Final Test Loss:     {test_loss:.4f}")

y_pred = np.argmax(model.predict(data['x_test_flat'], verbose=0), axis=1)
print("\\nPer-Digit Classification Report:")
print(classification_report(data['y_test'], y_pred, digits=4))

# Generate output charts
plot_learning_curves('../models/keras_training_history.json', '../outputs/learning_curves.png')
plot_confusion_matrix(data['y_test'], y_pred, '../outputs/confusion_matrix.png')
plot_sample_predictions(data['x_test'], data['y_test'], y_pred, '../outputs/sample_predictions.png')
"""),
        nbf.v4.new_markdown_cell("""## 5. Comparative Analysis: Traditional ML vs Deep Learning
To address the core objective: 'To compare traditional machine learning with deep learning',
we evaluate Logistic Regression, Random Forest Classifier, and the Deep Neural Network.
"""),
        nbf.v4.new_code_cell("""from compare_ml import compare_ml_and_dl

compare_ml_and_dl()
""")
    ]

    target = os.path.join(BASE_DIR, "Project1_Handwritten_Digit_Recognition", "notebooks", "Project1_Handwritten_Digit_Recognition.ipynb")
    with open(target, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Created] {target}")


def create_project2_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# COER UNIVERSITY, ROORKEE, UTTARAKHAND
## Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning
### MCA III Semester (Session 2026–2027)

---
# Project 2: Q-Learning based Maze Solver

**Objectives:**
1. Understand Reinforcement Learning concepts: Agent, Environment, States, Actions, and Rewards.
2. Formulate the Maze Navigation problem as a Markov Decision Process (MDP).
3. Implement the Tabular Q-Learning algorithm using the Bellman Optimality Equation.
4. Perform reward optimization and exploration tuning (Epsilon-Greedy policy).
5. Extract and visualize the optimal shortest path and state-value landscape.
"""),
        nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Ensure local source modules are discoverable
sys.path.append(os.path.abspath('../src'))
from maze_env import MazeEnv
from q_learning_agent import QLearningAgent

print("Gymnasium Version:", gym.__version__)
"""),
        nbf.v4.new_markdown_cell("""## 1. Maze Environment Setup
The environment is an 8x8 GridWorld featuring:
- **Start (S):** Initial state (0, 0)
- **Goal (G):** Terminal state (7, 7) yielding +100 reward
- **Walls (#):** Obstacles yielding -5 penalty and restricting movement
- **Traps (T):** Hazard cells yielding -15 penalty
- **Step Penalty:** -1 per movement to encourage finding the strictly shortest path
"""),
        nbf.v4.new_code_cell("""env = MazeEnv()
obs, info = env.reset()
print("Initial Maze Rendering:")
print(env.render())
print(f"Number of States: {env.num_states} (Grid size: {env.rows}x{env.cols})")
print(f"Number of Actions: {env.action_space.n} {MazeEnv.ACTION_NAMES}")
"""),
        nbf.v4.new_markdown_cell("""## 2. Train Q-Learning Agent
The agent updates its Q-table using the Bellman equation:
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
"""),
        nbf.v4.new_code_cell("""from train import train_q_learning

history = train_q_learning(
    episodes=600,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.992,
    save_dir='../models'
)
"""),
        nbf.v4.new_markdown_cell("""## 3. Evaluate Learned Policy & Find Shortest Path
We run the trained agent with a pure greedy policy (epsilon = 0).
"""),
        nbf.v4.new_code_cell("""from evaluate import evaluate_greedy_policy, plot_training_metrics, plot_maze_solution, plot_q_value_heatmap

agent = QLearningAgent(num_states=env.num_states, num_actions=env.action_space.n)
agent.load('../models/q_table.npy')

path, actions, total_reward, success = evaluate_greedy_policy(env, agent)
print(f"Goal Reached:         {success}")
print(f"Shortest Path Length: {len(path) - 1} steps")
print(f"Total Test Reward:    {total_reward:.2f}")
print(f"Action Sequence:      {' -> '.join(actions)}")

# Generate visualization plots
plot_training_metrics('../models/q_learning_history.json', '../outputs/training_metrics.png')
plot_maze_solution(env, path, '../outputs/maze_solution_path.png')
plot_q_value_heatmap(env, agent, '../outputs/q_table_heatmap.png')
"""),
        nbf.v4.new_markdown_cell("""## 4. Hyperparameter Tuning Comparison
We evaluate the sensitivity of Q-Learning to Learning Rate (alpha), Discount Factor (gamma), and Exploration Decay (epsilon).
"""),
        nbf.v4.new_code_cell("""from compare_hyperparams import run_hyperparameter_experiments

run_hyperparameter_experiments()
""")
    ]

    target = os.path.join(BASE_DIR, "Project2_QLearning_Maze_Solver", "notebooks", "Project2_QLearning_Maze_Solver.ipynb")
    with open(target, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Created] {target}")


if __name__ == "__main__":
    create_project1_notebook()
    create_project2_notebook()
