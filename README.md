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

## Overview

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




