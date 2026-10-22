# Mini Projects on Deep Learning and Reinforcement Learning
### COER University, Roorkee, Uttarakhand
**Rajendra Singh Bisht**
**Master of Computer Applications (MCA) — III Semester | Session 2026–2027**  
**Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning**

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.4.0-brightgreen.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Overview

This repository contains the complete implementation, documentation, and trained models for **Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning** for the MCA III Semester curriculum at COER University.

The project encompasses two comprehensive sub-projects addressing both perceptual vision modeling with deep architectures and sequential optimal decision-making with reinforcement learning:

1. **Project 1: Handwritten Digit Recognition using Neural Networks**
   - End-to-end Deep Neural Network (DNN) developed in **TensorFlow/Keras** and **PyTorch**.
   - Benchmarking and empirical comparison against Traditional Machine Learning models (**Random Forest Classifier** and **Multinomial Logistic Regression**).
   - Hyperparameter tuning comparing activation functions (**ReLU**, **Leaky ReLU**, **Tanh**, **Sigmoid**).
   - Full evaluation metrics: Confusion Matrix, Learning Curves, Per-Digit Classification Report, and Sample Prediction grids.

2. **Project 2: Q-Learning based Maze Solver**
   - Custom **Gymnasium** environment (`gymnasium.Env`) modeling a hazardous 2D GridWorld with obstacle walls and traps.
   - Tabular **Q-Learning** implementing the **Bellman Optimality Equation** with dynamic reward optimization.
   - $\epsilon$-greedy exploration-exploitation balancing with geometric decay.
   - Extraction and visualization of the globally **shortest path (14 steps)** avoiding traps and walls.
   - State-value function ($V(s) = \max_a Q(s, a)$) heatmaps and multi-parameter ablation analysis ($\alpha$, $\gamma$, $\epsilon_{\text{decay}}$).

---

## Project Repository Structure

```
Mini-Projects-DL-RL/                              # Single-command runner for entire suite
├── README.md
│
├── Project1_Handwritten_Digit_Recognition/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataset.py                        # MNIST dataset loading and normalization
│   │   ├── model.py                          # Keras and PyTorch DNN architectures
│   │   ├── train.py                          # Training pipeline with early stopping & callbacks
│   │   ├── evaluate.py                       # Evaluation, confusion matrix & prediction plots
│   │   └── compare_ml.py                     # Traditional ML vs DL & Activation tuning
│   ├── notebooks/
│   │   └── Project1_Handwritten_Digit_Recognition.ipynb # Interactive Jupyter Notebook
│   ├── models/
│   │   ├── mnist_dnn.keras                   # Trained Keras DNN weights
│   │   ├── mnist_dnn.pth                     # Trained PyTorch DNN weights
│   │   ├── baseline_rf.joblib                # Trained Random Forest baseline
│   │   ├── keras_training_history.json
│   │   └── pytorch_training_history.json
│   └── outputs/
│       ├── learning_curves.png               # Loss & Accuracy vs Epochs
│       ├── confusion_matrix.png              # 10x10 digit classification heatmap
│       ├── sample_predictions.png            # 25-sample test prediction grid
│       ├── model_comparison_metrics.png      # ML vs DL Accuracy and Training Time
│       ├── activation_comparison.png         # ReLU vs LeakyReLU vs Tanh vs Sigmoid
│       ├── classification_report.txt
│       └── model_comparison_metrics.json
│
├── Project2_QLearning_Maze_Solver/
    ├── src/
    │   ├── __init__.py
    │   ├── maze_env.py                       # Gymnasium-compliant 2D Maze Environment
│   │   ├── q_learning_agent.py               # Tabular Q-Learning Agent & Bellman updates
│   │   ├── train.py                          # Episodic training loop & metric logging
│   │   ├── evaluate.py                       # Policy evaluation & path plotting
│   │   └── compare_hyperparams.py            # Alpha, Gamma, and Decay sensitivity studies
│   ├── notebooks/
│   │   └── Project2_QLearning_Maze_Solver.ipynb # Interactive Jupyter Notebook
│   ├── models/
│   │   ├── q_table.npy                       # Learned 64x4 Q-Value matrix
│   │   ├── optimal_policy.json               # Extracted optimal policy per state
│   │   └── q_learning_history.json
│   └── outputs/
│       ├── training_metrics.png              # Rewards, Steps, and Epsilon decay curves
│       ├── maze_solution_path.png            # 2D Grid with optimal trajectory arrows
│       ├── q_table_heatmap.png               # State-value function V(s) heatmap
│       ├── hyperparameter_comparison.png     # Multi-panel ablation curves
│       ├── evaluation_summary.json
│       └── hyperparameter_comparison.json
```

---


