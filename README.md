# Mini Projects on Deep Learning and Reinforcement Learning
### COER University, Roorkee, Uttarakhand
**Master of Computer Applications (MCA) — III Semester | Session 2026–2027**  
**Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning**

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.4.0-brightgreen.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Overview

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

## 📂 Project Repository Structure

```
Mini-Projects-DL-RL/
├── .gitignore
├── requirements.txt
├── run_all.py                                # Single-command runner for entire suite
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
│   ├── src/
│   │   ├── __init__.py
│   │   ├── maze_env.py                       # Gymnasium-compliant 2D Maze Environment
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
│
└── docs/
    ├── Lab_Sheet_08_Report.md                # Comprehensive Academic Lab Report
    ├── Viva_Questions_and_Answers.md         # 35+ Viva preparation Q&A for MCA examiners
    └── Observations_and_Interpretations.md   # Empirical findings & engineering insights
```

---

## 🔬 Project 1: Handwritten Digit Recognition (DNN)

### Architecture
- **Input Layer:** 784 nodes (Flattened $28 \times 28$ grayscale pixels, normalized to $[0.0, 1.0]$)
- **Hidden Layer 1:** 256 units + Batch Normalization + ReLU + Dropout(0.2)
- **Hidden Layer 2:** 128 units + Batch Normalization + ReLU + Dropout(0.2)
- **Hidden Layer 3:** 64 units + Batch Normalization + ReLU + Dropout(0.2)
- **Output Layer:** 10 units + Softmax activation (Digits 0–9)
- **Total Parameters:** 243,658 trainable weights

### Key Experimental Results
- **TensorFlow/Keras Test Accuracy:** **97.93%** (Test Loss: 0.0758)
- **PyTorch Test Accuracy:** **98.24%**

#### Traditional ML vs. Deep Learning Comparison
| Model | Paradigm | Accuracy | Macro F1 | Training Time |
|:---|:---|:---:|:---:|:---:|
| **Logistic Regression** | Linear Softmax Baseline | 92.56% | 0.9245 | 14.9 s |
| **Random Forest (100 Trees)** | Ensemble Bagging | 96.89% | 0.9687 | 12.5 s |
| **Deep Neural Network (DNN)** | Deep Learning (TensorFlow) | **97.93%** | **0.9792** | 49.6 s |
| **PyTorch DNN** | Deep Learning (PyTorch) | **98.24%** | **0.9823** | 90.0 s |

#### Activation Functions Hyperparameter Ablation
| Activation | Final Test Accuracy | Test Loss | Characteristics |
|:---|:---:|:---:|:---|
| **ReLU** | **97.83%** | **0.0682** | Fastest convergence; lowest loss |
| **Leaky ReLU** | 97.66% | 0.0757 | Zero dead units; high stability |
| **Tanh** | 96.85% | 0.1096 | Zero-centered, moderate convergence |
| **Sigmoid** | 95.85% | 0.1388 | Gradient saturation in deep layers |

---

## 🤖 Project 2: Q-Learning based Maze Solver

### Environment & Reward Shaping
- **Gymnasium Grid:** $8 \times 8$ GridWorld ($64$ states, $4$ discrete actions: `UP`, `RIGHT`, `DOWN`, `LEFT`).
- **Goal Reward ($G$):** $+100.0$ upon reaching terminal cell $(7, 7)$.
- **Step Penalty:** $-1.0$ per valid move (Reward shaping compelling the agent to discover the strictly shortest trajectory).
- **Wall Collision:** $-5.0$ penalty (agent remains in place).
- **Trap Penalty ($T$):** $-15.0$ penalty at hazard coordinate $(6, 3)$.

### Bellman Update Equation
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### Results Under Pure Greedy Policy ($\epsilon = 0$)
- **Goal Reached:** `True`
- **Shortest Path Length:** **14 steps** (Theoretical optimum)
- **Total Test Reward:** **+87.00**
- **Action Sequence:**
  `RIGHT -> RIGHT -> RIGHT -> DOWN -> DOWN -> DOWN -> DOWN -> RIGHT -> RIGHT -> RIGHT -> DOWN -> DOWN -> RIGHT -> DOWN`
- **Learned Coordinates:**
  `[(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3), (4,3), (4,4), (4,5), (4,6), (5,6), (6,6), (6,7), (7,7)]`

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.11+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/Rajendra7637/Mini-Projects-Deep-Learning-and-Reinforcement-Learning.git
cd Mini-Projects-Deep-Learning-and-Reinforcement-Learning
pip install -r requirements.txt
```

### 2. Execute Entire Suite (Single Command)
Run the automated unified pipeline to train all models, generate comparison charts, and evaluate both projects:

```bash
python run_all.py
```

### 3. Run Projects Individually

#### Project 1: Handwritten Digit Recognition
```bash
cd Project1_Handwritten_Digit_Recognition/src
python train.py      # Train Keras and PyTorch models
python evaluate.py   # Generate confusion matrix and learning curves
python compare_ml.py # Run Traditional ML comparison and activation tuning
```

#### Project 2: Q-Learning Maze Solver
```bash
cd Project2_QLearning_Maze_Solver/src
python train.py               # Train Q-Learning agent
python evaluate.py            # Evaluate greedy policy & render shortest path
python compare_hyperparams.py # Run alpha, gamma, and epsilon sensitivity studies
```

### 4. Interactive Jupyter Notebooks
Launch Jupyter to explore interactive step-by-step walkthroughs:

```bash
jupyter lab
```
Navigate to:
- `Project1_Handwritten_Digit_Recognition/notebooks/Project1_Handwritten_Digit_Recognition.ipynb`
- `Project2_QLearning_Maze_Solver/notebooks/Project2_QLearning_Maze_Solver.ipynb`

---

## 📚 Academic Submissions & Documentation

- [Full Academic Laboratory Report](docs/Lab_Sheet_08_Report.md)
- [Comprehensive Viva Questions & Answers Guide](docs/Viva_Questions_and_Answers.md)
- [Empirical Observations and Interpretations](docs/Observations_and_Interpretations.md)

---

## 👨‍💻 Author & Attribution

- **Student / Developer:** Rajendra Singh ([@Rajendra7637](https://github.com/Rajendra7637))
- **Program:** Master of Computer Applications (MCA) – Semester III
- **Academic Session:** 2026–2027
- **Institution:** COER University, Roorkee, Uttarakhand, India
- **Coursework:** Lab Sheet-08 (Mini Projects on Deep Learning and Reinforcement Learning)

---
*Licensed under the [MIT License](LICENSE).*
