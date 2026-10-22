# Empirical Observations and Interpretations
## Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning
### MCA III Semester (Session 2026–2027) — COER University

---

## 1. Project 1 Observations: Handwritten Digit Recognition (DNN)

### 1.1 Architecture & Convergence Behavior
- **Deep Neural Network Convergence:** The 3-hidden-layer DNN (256 $\to$ 128 $\to$ 64) with Batch Normalization and Dropout converged rapidly within 10–12 epochs.
- **Validation Loss:** Validation loss decreased steadily from $0.132$ in epoch 1 down to $0.058$ in epoch 10, tracking the training loss closely without divergence. This proves that the combination of **Batch Normalization** and **Dropout ($p = 0.2$)** successfully prevented overfitting.
- **Accuracy Milestones:**
  - TensorFlow/Keras Test Accuracy: **97.93%**
  - PyTorch Test Accuracy: **98.24%**

### 1.2 Per-Digit Confusion Analysis
- **Top Performing Classes:**
  - Digit `1` achieved the highest F1-score (**0.9890**) and recall (**0.9912**) due to its distinct vertical stroke morphology.
  - Digit `0` achieved high recall (**0.9929**) due to its closed circular loop structure.
- **Confusion Hotspots:**
  - Slight confusion occurred between digit `4` and `9` due to similar ascenders and upper loops.
  - Digit `8` exhibited the lowest recall (**0.9600**), occasionally being misclassified as `3` or `5` when handwriting left loops unclosed or disconnected.

### 1.3 Machine Learning Baseline Comparison
1. **Logistic Regression (92.56% Accuracy, 14.9s):**
   - The linear decision boundaries failed to model non-linear stroke curvatures and spatial variations.
   - It reached max iterations without complete optimization convergence.
2. **Random Forest (96.89% Accuracy, 12.5s):**
   - An ensemble of 100 deep decision trees achieved respectable accuracy by capturing non-linear feature splits.
   - However, because tree splits are orthogonal and axis-aligned in pixel space, they could not smoothly generalize across continuous pixel transformations.
3. **Deep Neural Network (97.93% / 98.24% Accuracy):**
   - Superior representation learning through composite matrix multiplications and non-linear activations.
   - Demonstrated superior generalizability on noisy and skewed test digits.

### 1.4 Activation Function Hyperparameter Tuning
- **ReLU (97.83%, Loss: 0.0682):** Fastest training dynamics and lowest cross-entropy loss due to constant gradient ($1.0$) for positive activations.
- **LeakyReLU (97.66%, Loss: 0.0757):** Closely matched ReLU, with zero dead units throughout training.
- **Tanh (96.85%, Loss: 0.1096):** Slower convergence as gradients shrank in deeper layers.
- **Sigmoid (95.85%, Loss: 0.1388):** Showed noticeable gradient saturation, resulting in the slowest convergence and highest test loss.

---

## 2. Project 2 Observations: Q-Learning Maze Solver

### 2.1 Learning Dynamics Across Episodes
- **Episodes 1 – 50 (Initial Exploration Phase):**
  - $\epsilon$ decayed from $1.0$ down to $0.67$.
  - The agent wandered pseudo-randomly, colliding frequently with boundary walls and traps (average reward: $-217.78$). Success rate was only $58\%$.
- **Episodes 51 – 150 (Knowledge Propagation Phase):**
  - Success rate reached **100%**.
  - The positive $+100$ goal reward began propagating backward through the Q-table via the Bellman update equation:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
  - Average steps to reach the goal dropped precipitously from $107$ steps down to $22.2$ steps.
- **Episodes 200 – 600 (Exploitation and Fine-Tuning Phase):**
  - $\epsilon \to 0.01$.
  - The agent transitioned almost entirely to exploiting the learned Q-values.
  - Steps stabilized at **14 steps** (the theoretical minimum path length). Average episode reward plateaued at $+86.40$.

### 2.2 Shortest Path & Reward Optimization
- **Discovered Shortest Path:**
  - Length: **14 steps**
  - Total Accumulated Test Reward: **+87.00**
  - Trajectory:
    $$(0,0) \to (0,1) \to (0,2) \to (0,3) \to (1,3) \to (2,3) \to (3,3) \to (4,3) \to (4,4) \to (4,5) \to (4,6) \to (5,6) \to (6,6) \to (6,7) \to (7,7)$$
- **Obstacle & Hazard Avoidance:**
  - The agent bypassed the deadly trap cell at $(6, 3)$ with a safety margin of multiple cells.
  - The $-1.0$ step penalty acted as a strong regularizer that penalized unnecessary loops or detours.

### 2.3 Hyperparameter Sensitivities
1. **Learning Rate ($\alpha$):**
   - $\alpha = 0.1$ achieved smooth, monotonic value convergence. $\alpha = 0.5$ introduced instability and transient policy degradation during mid-exploration.
2. **Discount Factor ($\gamma$):**
   - $\gamma = 0.99$ was critical for propagating rewards across long horizons (14+ steps). Smaller values ($\gamma = 0.70$) caused the agent to discount the goal reward so heavily that it prioritized immediate short-term survival over reaching the distant goal.
3. **Epsilon Decay ($\lambda_{\text{decay}}$):**
   - Balanced decay ($0.992$) ensured sufficient early exploration while guaranteeing that the policy converged stably to the greedy optimum before episode 400.
