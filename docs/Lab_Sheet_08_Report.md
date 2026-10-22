# COER UNIVERSITY, ROORKEE, UTTARAKHAND
## Department of Computer Science & Applications
### MCA III Semester (Session 2026–2027)

---

# Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning

- **Course:** Master of Computer Applications (MCA) – III Semester
- **Subject:** Deep Learning and Reinforcement Learning Laboratory
- **Author / GitHub User:** Rajendra7637
- **Institution:** COER University, Roorkee-Haridwar Highway, Vardhmanpuram, Roorkee-247 667, District Haridwar, Uttarakhand, India

---

## 1. Laboratory Objectives

1. **Neural Network Fundamentals:** To understand multi-layer neural network architectures, forward propagation, loss computation, backpropagation, and weight optimization.
2. **Deep Learning System Engineering:** To construct and evaluate robust end-to-end intelligent systems utilizing modern deep learning frameworks (**TensorFlow/Keras** and **PyTorch**).
3. **Machine Learning Comparison:** To empirically compare traditional machine learning classifiers (Logistic Regression, Random Forest) against Deep Neural Networks (DNN) on benchmark vision data.
4. **Reinforcement Learning Foundations:** To formulate sequential decision-making problems using the **Markov Decision Process (MDP)** framework.
5. **Q-Learning Implementation:** To implement tabular Q-Learning with the **Bellman Optimality Equation** in a custom **Gymnasium** environment.
6. **Reward Optimization:** To engineer reward functions and exploration-exploitation strategies ($\epsilon$-greedy) to enable an agent to discover the globally shortest path in a hazardous maze.
7. **Empirical Analysis & Hyperparameter Tuning:** To plot learning curves, evaluate convergence behavior, save trained model weights, and document engineering observations.

---

## 2. Theoretical Foundations

### 2.1 Deep Neural Networks (DNN)

A Deep Neural Network (DNN) is an artificial neural network with multiple hidden layers between the input and output layers. Each layer consists of artificial neurons (nodes) connected by trainable weights $W$ and biases $b$:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = \sigma(z^{[l]})$$

where:
- $l$ denotes layer index.
- $\sigma(\cdot)$ is a non-linear activation function.
- $a^{[0]} = x$ represents the input feature vector.

#### Activation Functions
Non-linear activation functions enable neural networks to learn arbitrary complex non-linear decision boundaries:
- **Rectified Linear Unit (ReLU):** $f(z) = \max(0, z)$. Resolves vanishing gradient issues and facilitates sparse activations.
- **Leaky ReLU:** $f(z) = \max(\alpha z, z)$ where $\alpha \approx 0.1$. Prevents the "dying ReLU" problem by maintaining a small gradient for negative inputs.
- **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$. Squeezes activations into $(0, 1)$, prone to gradient saturation when $|z|$ is large.
- **Hyperbolic Tangent (Tanh):** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Zero-centered activation ranging within $(-1, 1)$.

#### Loss Function & Optimization
For multi-class classification with mutually exclusive classes ($K = 10$), the network output is normalized via the **Softmax** function:

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}$$

The network is trained by minimizing the **Sparse Categorical Cross-Entropy Loss**:

$$\mathcal{L} = -\sum_{i=1}^{N} \log(p_{y_i})$$

Optimization is conducted using the **Adam (Adaptive Moment Estimation)** algorithm, combining the benefits of AdaGrad and RMSProp with exponentially decaying running averages of past gradients ($m_t$) and squared gradients ($v_t$).

---

### 2.2 Reinforcement Learning & Q-Learning

Reinforcement Learning (RL) formalizes goal-directed learning where an autonomous agent interacts with an environment modeled as a **Markov Decision Process (MDP)** defined by the 5-tuple $(S, A, P, R, \gamma)$:
- $S$: Finite set of environmental states.
- $A$: Finite set of discrete actions $\{ \text{UP, RIGHT, DOWN, LEFT} \}$.
- $P(s' | s, a)$: Transition probability function.
- $R(s, a, s')$: Reward function returned by the environment upon transition.
- $\gamma \in [0, 1)$: Discount factor balancing immediate versus future returns.

The cumulative expected return from time step $t$ is:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

#### Bellman Optimality Equation
The optimal action-value function $Q^*(s, a)$ satisfies the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \;\middle|\; S_t = s, A_t = a \right]$$

#### Tabular Q-Learning Algorithm
Q-Learning is a model-free, off-policy Temporal Difference (TD) control algorithm. At every step, the Q-table entry is updated via:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where:
- $\alpha \in (0, 1]$ is the learning rate.
- $TD_{\text{target}} = r + \gamma \max_{a'} Q(s', a')$.
- $TD_{\text{error}} = TD_{\text{target}} - Q(s, a)$.

#### Exploration vs. Exploitation ($\epsilon$-Greedy Policy)
The agent balances exploration (discovering new trajectories) and exploitation (utilizing current knowledge) through an $\epsilon$-greedy policy:

$$\pi(a|s) = \begin{cases} 
\text{random action from } A & \text{with probability } \epsilon \\ 
\arg\max_{a} Q(s, a) & \text{with probability } 1 - \epsilon 
\end{cases}$$

The exploration parameter $\epsilon$ undergoes exponential decay:

$$\epsilon_{t+1} = \max(\epsilon_{\min}, \epsilon_t \cdot \lambda_{\text{decay}})$$

---

## 3. Project 1: Handwritten Digit Recognition using Neural Networks

### 3.1 Dataset Specification
- **Dataset:** MNIST (Modified National Institute of Standards and Technology)
- **Samples:** 70,000 grayscale images ($28 \times 28$ pixels)
  - Training Set: 50,000 samples
  - Validation Set: 10,000 samples
  - Test Set: 10,000 samples
- **Feature Preprocessing:** Pixel intensities scaled to $[0.0, 1.0]$ via $X_{\text{norm}} = X / 255.0$. Reshaped from $(28, 28)$ to flat 784-dimensional vectors.

### 3.2 Deep Neural Network Architecture

```
=================================================================
Layer (type)                Output Shape              Param #
=================================================================
InputLayer                  (None, 784)               0
Dense (Layer 1)             (None, 256)               200,960
BatchNormalization          (None, 256)               1,024
Activation (ReLU)           (None, 256)               0
Dropout (rate = 0.2)        (None, 256)               0
Dense (Layer 2)             (None, 128)               32,896
BatchNormalization          (None, 128)               512
Activation (ReLU)           (None, 128)               0
Dropout (rate = 0.2)        (None, 128)               0
Dense (Layer 3)             (None, 64)                8,256
BatchNormalization          (None, 64)                256
Activation (ReLU)           (None, 64)                0
Dropout (rate = 0.2)        (None, 64)                0
Dense (Output Layer)        (None, 10)                650
=================================================================
Total Trainable Parameters: 243,658 (951.79 KB)
Optimizer: Adam (lr = 0.001) | Loss: Sparse Categorical Cross-Entropy
=================================================================
```

### 3.3 Training & Regularization Strategy
- **Batch Size:** 128
- **Maximum Epochs:** 12 (with Early Stopping patience = 5, monitoring `val_loss`)
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor = 0.5, patience = 2, min_lr = 1e-5)
- **PyTorch Dual Implementation:** An equivalent PyTorch module (`nn.Module`) was trained using cross-entropy loss and Adam optimizer to validate framework parity.

### 3.4 Experimental Results & Evaluation

#### Classification Performance on 10,000 Unseen Test Images
- **Keras DNN Test Accuracy:** **97.93%** (Test Loss: 0.0758)
- **PyTorch DNN Test Accuracy:** **98.24%**

#### Detailed Per-Digit Classification Report:
| Digit Class | Precision | Recall | F1-Score | Support |
|:-----------:|:---------:|:------:|:--------:|:-------:|
| 0 | 0.9799 | 0.9929 | 0.9863 | 980 |
| 1 | 0.9868 | 0.9912 | 0.9890 | 1135 |
| 2 | 0.9595 | 0.9874 | 0.9733 | 1032 |
| 3 | 0.9724 | 0.9782 | 0.9753 | 1010 |
| 4 | 0.9917 | 0.9756 | 0.9836 | 982 |
| 5 | 0.9809 | 0.9776 | 0.9792 | 892 |
| 6 | 0.9832 | 0.9781 | 0.9806 | 958 |
| 7 | 0.9795 | 0.9737 | 0.9766 | 1028 |
| 8 | 0.9801 | 0.9600 | 0.9699 | 974 |
| 9 | 0.9801 | 0.9762 | 0.9782 | 1009 |
| **Macro Average** | **0.9794** | **0.9791** | **0.9792** | **10,000** |
| **Weighted Average** | **0.9794** | **0.9793** | **0.9793** | **10,000** |

### 3.5 Comparative Study: Traditional ML vs. Deep Learning

| Model Architecture | Model Paradigm | Test Accuracy | Macro F1-Score | Training Time | Model Artifact Saved |
|:---|:---|:---:|:---:|:---:|:---|
| **Logistic Regression** | Linear Softmax Baseline | 92.56% | 0.9245 | 14.9 s | In memory |
| **Random Forest (100 Trees)** | Ensemble Bagging | 96.89% | 0.9687 | 12.5 s | `baseline_rf.joblib` |
| **Deep Neural Network (DNN)** | Deep Learning (3 Hidden Layers) | **97.93%** | **0.9792** | 49.6 s | `mnist_dnn.keras` |
| **PyTorch DNN** | Deep Learning (Equivalent) | **98.24%** | **0.9823** | 90.0 s | `mnist_dnn.pth` |

#### Key Insights:
1. The **Deep Neural Network** outperforms Logistic Regression by **+5.37%** in accuracy because linear hyperplanes cannot capture complex non-linear stroke variations.
2. The DNN improves upon Random Forest by **+1.04%** in accuracy while generalizing better across subtle rotations and handwriting quirks.
3. Batch Normalization combined with Dropout (0.2) effectively neutralized overfitting, evidenced by the validation loss curve mirroring the training loss smoothly.

### 3.6 Hyperparameter Tuning: Activation Functions

An ablation experiment was conducted comparing activation functions under identical architectures:

| Activation Function | Final Test Accuracy | Final Test Loss | Convergence Characteristics |
|:---|:---:|:---:|:---|
| **ReLU** | **97.83%** | **0.0682** | Fastest convergence; lowest final loss. |
| **Leaky ReLU** | 97.66% | 0.0757 | Highly stable; effectively prevents dead units. |
| **Tanh** | 96.85% | 0.1096 | Zero-centered, but slower convergence in deeper layers. |
| **Sigmoid** | 95.85% | 0.1388 | Suffered noticeable gradient saturation; lowest accuracy. |

---

## 4. Project 2: Q-Learning based Maze Solver

### 4.1 Environment Design & MDP Formulation

The environment is an $8 \times 8$ grid world implementing the official `gymnasium.Env` interface:
- **Total States:** $8 \times 8 = 64$ discrete states ($s = r \times 8 + c$).
- **Actions:** Discrete(4):
  - `0`: UP $(\Delta r = -1, \Delta c = 0)$
  - `1`: RIGHT $(\Delta r = 0, \Delta c = +1)$
  - `2`: DOWN $(\Delta r = +1, \Delta c = 0)$
  - `3`: LEFT $(\Delta r = 0, \Delta c = -1)$
- **Maze Layout:**
  - Start position: $(0, 0)$
  - Goal position: $(7, 7)$
  - Interior Walls: Impassable barriers forcing path deviations.
  - Trap cell: Dangerous hazard at coordinate $(6, 3)$.

### 4.2 Reward Engineering for Shortest Path Optimization

To ensure the agent converges strictly to the **shortest path** without unnecessary wandering, the reward structure was engineered as follows:
- **Goal Reached ($G$):** $+100.0$ (Episode terminates successfully).
- **Step Penalty:** $-1.0$ per valid transition into path cell (Penalizes lengthy trajectories, compelling minimal path length).
- **Wall Collision / Boundary:** $-5.0$ (Agent stays in place, penalizing invalid motions).
- **Trap Cell ($T$):** $-15.0$ (Strong negative feedback discouraging hazardous navigation).

### 4.3 Q-Learning Parameters
- **Episodes:** 600
- **Learning Rate ($\alpha$):** 0.1
- **Discount Factor ($\gamma$):** 0.99
- **Initial Epsilon ($\epsilon_0$):** 1.0 (100% random exploration initially)
- **Minimum Epsilon ($\epsilon_{\min}$):** 0.01 (1% residual exploration)
- **Epsilon Decay Rate:** 0.992 per episode
- **Max Steps per Episode:** 150

### 4.4 Training Dynamics & Learning Curves

| Episode Interval | 50-Ep Avg Reward | 50-Ep Avg Steps | Epsilon ($\epsilon$) | Success Rate |
|:---:|:---:|:---:|:---:|:---:|
| 1 – 50 | -217.78 | 107.0 | 0.669 | 58.0% |
| 51 – 100 | +35.14 | 33.8 | 0.448 | 100.0% |
| 101 – 150 | +65.32 | 22.2 | 0.300 | 100.0% |
| 151 – 200 | +73.18 | 18.9 | 0.201 | 100.0% |
| 201 – 300 | +82.66 | 15.9 | 0.090 | 100.0% |
| 301 – 450 | +85.10 | 14.9 | 0.027 | 100.0% |
| 451 – 600 | **+86.40** | **14.3** | **0.010** | **100.0%** |

### 4.5 Evaluation with Pure Greedy Policy ($\epsilon = 0$)

Upon executing the trained Q-table using the pure greedy policy:
- **Goal Reached:** `True`
- **Shortest Path Length:** **14 steps**
- **Total Test Reward:** **+87.00** ($100.0 \text{ [goal]} - 13 \times 1.0 \text{ [step penalties]} = 87.00$)
- **Action Sequence:**
  `RIGHT -> RIGHT -> RIGHT -> DOWN -> DOWN -> DOWN -> DOWN -> RIGHT -> RIGHT -> RIGHT -> DOWN -> DOWN -> RIGHT -> DOWN`
- **Optimal Coordinate Path:**
  `[(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3), (4,3), (4,4), (4,5), (4,6), (5,6), (6,6), (6,7), (7,7)]`

The agent completely avoids the trap at $(6, 3)$ and maneuvers cleanly around all internal walls.

### 4.6 Hyperparameter Tuning Experiments

Three systematic ablation studies were executed:

1. **Learning Rate ($\alpha \in \{0.02, 0.1, 0.5\}$):**
   - $\alpha = 0.02$: Slower learning; requires more episodes to propagate target values from the goal backward.
   - $\alpha = 0.1$: **Optimal convergence**; smooth TD-error reduction without oscillations.
   - $\alpha = 0.5$: High variance during early exploration; occasional overshooting of Q-values.
2. **Discount Factor ($\gamma \in \{0.70, 0.90, 0.99\}$):**
   - $\gamma = 0.70$: Agent heavily discounts future rewards, making it myopic and prone to stagnating in loops.
   - $\gamma = 0.99$: **Optimal performance**; ensures the terminal $+100$ goal reward propagates across all 14 steps back to the start cell $(0, 0)$.
3. **Epsilon Decay Strategy ($\lambda_{\text{decay}} \in \{0.97, 0.992, 0.998\}$):**
   - Fast Decay (0.97): Explores insufficiently; risks settling into sub-optimal longer routes.
   - Balanced Decay (0.992): **Optimal balance** between discovering alternate routes and exploiting optimal policies.
   - Slow Decay (0.998): Prolonged random actions; slower average reward convergence.

---

## 5. Summary of Saved Artifacts

```
Mini-Projects-DL-RL/
├── Project1_Handwritten_Digit_Recognition/
│   ├── models/
│   │   ├── mnist_dnn.keras            # Trained Keras DNN Model weights
│   │   ├── mnist_dnn.pth              # Trained PyTorch DNN Model weights
│   │   ├── baseline_rf.joblib         # Trained Random Forest Baseline
│   │   ├── keras_training_history.json
│   │   └── pytorch_training_history.json
│   └── outputs/
│       ├── learning_curves.png        # Training vs Val Loss & Accuracy
│       ├── confusion_matrix.png       # 10x10 Digit Confusion Heatmap
│       ├── sample_predictions.png     # 25-sample Grid (Correct vs Misclassified)
│       ├── model_comparison_metrics.png # DNN vs Random Forest vs Logistic Regression
│       └── activation_comparison.png  # ReLU vs Sigmoid vs Tanh vs LeakyReLU
│
└── Project2_QLearning_Maze_Solver/
    ├── models/
    │   ├── q_table.npy                # 64x4 Optimal Q-Value Matrix
    │   ├── optimal_policy.json        # Greedy Action per State
    │   └── q_learning_history.json    # Episodic Training Metrics
    └── outputs/
        ├── training_metrics.png       # Rewards, Steps, and Epsilon Decay Curves
        ├── maze_solution_path.png     # 2D Grid with Directional Shortest Path
        ├── q_table_heatmap.png        # State-Value Landscape V(s) = max Q(s,a)
        └── hyperparameter_comparison.png # Alpha, Gamma, and Decay Ablation Curves
```

---

## 6. Academic Conclusion

Lab Sheet-08 successfully demonstrated the end-to-end design, implementation, and empirical verification of both Deep Learning and Reinforcement Learning paradigms:
1. **Deep Learning:** Proved superior to traditional machine learning on high-dimensional raw pixel representations, achieving **97.93% (Keras)** and **98.24% (PyTorch)** accuracy. Regularization techniques (Batch Normalization and Dropout) were essential to prevent overfitting.
2. **Reinforcement Learning:** Tabular Q-Learning utilizing the Bellman optimality equation successfully solved the navigation task, converging to the exact globally shortest path of **14 steps** while avoiding traps and wall obstacles through reward shaping.
