# Comprehensive Viva Questions and Answers
## Lab Sheet-08: Mini Projects on Deep Learning and Reinforcement Learning
### MCA III Semester (Session 2026–2027) — COER University

---

## Part A: Deep Learning & Neural Networks (Project 1)

### Q1: What is a Deep Neural Network (DNN) and how does it differ from a Single-Layer Perceptron?
**Answer:** A Single-Layer Perceptron consists only of an input layer and an output layer with no hidden layers. Because of this, it can only solve linearly separable problems (e.g., AND, OR) and fundamentally fails on non-linear problems like XOR (as shown by Minsky and Papert). A Deep Neural Network (DNN) incorporates two or more hidden layers between the input and output. According to the Universal Approximation Theorem, a multi-layer network with non-linear activation functions can approximate any continuous function, allowing it to capture complex, non-linear manifolds such as handwritten digits.

---

### Q2: Why did we flatten the 28x28 MNIST images into a 784-dimensional vector?
**Answer:** Fully connected (Dense) layers require 1D input feature vectors. A 2D grayscale image of size $28 \times 28$ contains $28 \times 28 = 784$ total pixels. By flattening the array along spatial dimensions, each pixel coordinate becomes an independent input feature $x_1, x_2, \dots, x_{784}$ fed into the initial Dense layer.

---

### Q3: Why is normalization ($X / 255.0$) necessary before feeding pixels into the network?
**Answer:** Raw pixel intensities range from integer values $0$ to $255$. Feeding large unnormalized integers causes the weighted inputs $z = \sum w_i x_i + b$ to become very large, which can:
1. Saturate activation functions and cause vanishing or exploding gradients.
2. Produce an ill-conditioned, elongated loss landscape, slowing gradient descent or causing numerical instability.
Normalizing pixel values to the range $[0.0, 1.0]$ ensures zero/near-zero mean, uniform feature scale, stable variance, and faster, smoother convergence.

---

### Q4: Explain the role and mathematical definition of the ReLU activation function.
**Answer:** ReLU (Rectified Linear Unit) is defined as:
$$f(z) = \max(0, z)$$
Its derivative is:
$$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$
**Advantages:**
- It does not saturate in the positive region, effectively solving the vanishing gradient problem during backpropagation.
- It is computationally very inexpensive (a simple thresholding operation).
- It introduces sparsity in representations since negative activations become exactly zero.

---

### Q5: What is the "Dying ReLU" problem, and how does Leaky ReLU address it?
**Answer:** If a large gradient pushes the weights of a neuron such that it produces negative inputs for all training examples, the gradient of ReLU becomes $0$ permanently. The neuron ceases to update and "dies".  
**Leaky ReLU** solves this by assigning a small non-zero slope $\alpha$ (typically $0.01$ or $0.1$) to negative inputs:
$$f(z) = \max(\alpha z, z)$$
Because the derivative for $z < 0$ is $\alpha > 0$, gradients continue to flow and the neuron can recover during subsequent training iterations.

---

### Q6: What is Softmax activation and why is it placed at the output layer?
**Answer:** Softmax converts a raw logit vector $z = [z_1, z_2, \dots, z_K]$ into a discrete probability distribution over $K$ mutually exclusive classes:
$$p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
It ensures:
1. All outputs are strictly non-negative: $p_k > 0$.
2. The sum of all class probabilities equals exactly $1$: $\sum_{k=1}^K p_k = 1.0$.
This allows the output to be interpreted directly as posterior class probabilities $P(Y = k | X)$.

---

### Q7: Explain Sparse Categorical Cross-Entropy loss.
**Answer:** Categorical Cross-Entropy measures the distance between the true distribution $y$ and predicted probability $\hat{y}$:
$$\mathcal{L} = -\sum_{k=1}^K y_k \log(\hat{y}_k)$$
In standard Categorical Cross-Entropy, targets are one-hot encoded vectors (e.g., `[0, 0, 1, 0]`).  
In **Sparse Categorical Cross-Entropy**, class labels are provided directly as integer indices (e.g., `y = 2`). The loss simplifies to:
$$\mathcal{L} = -\log(\hat{y}_{y_{\text{true}}})$$
This saves significant memory and computation by avoiding sparse one-hot matrix allocations.

---

### Q8: What is Batch Normalization and what advantages does it provide?
**Answer:** Batch Normalization (Ioffe & Szegedy, 2015) normalizes layer inputs across each mini-batch:
$$\hat{z} = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{z} + \beta$$
where $\mu_B, \sigma_B^2$ are mini-batch mean and variance, and $\gamma, \beta$ are learnable scale and shift parameters.  
**Benefits:**
- Stabilizes internal covariate shift.
- Enables higher learning rates.
- Acts as a mild regularizer, reducing sensitivity to weight initialization.

---

### Q9: How does Dropout prevent overfitting?
**Answer:** During each training forward pass, Dropout randomly deactivates (sets to zero) a fraction $p$ (e.g., $0.2$) of neurons. This prevents co-adaptation among units—no single neuron can rely on specific companion units. It simulates training an ensemble of an exponentially large number of thinned subnetworks, resulting in superior generalization on unseen data. At inference time, Dropout is disabled, and activations are scaled by $(1 - p)$.

---

### Q10: What is the difference between Gradient Descent, Stochastic Gradient Descent (SGD), and Adam?
**Answer:**
- **Batch Gradient Descent:** Computes gradients over the entire training set per step. Exact, but extremely slow and memory-intensive for large datasets.
- **Stochastic Gradient Descent (SGD):** Updates weights using a single random sample (or mini-batch) at a time. Fast, but updates exhibit high variance and oscillate.
- **Adam (Adaptive Moment Estimation):** Computes individual adaptive learning rates for each parameter by maintaining exponentially decaying averages of past gradients ($m_t$, first moment) and past squared gradients ($v_t$, second raw moment):
  $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
  It accelerates through ravines and dampens oscillations, outperforming standard SGD on vision tasks.

---

### Q11: Why did Deep Neural Networks outperform Traditional Machine Learning (Random Forest and Logistic Regression)?
**Answer:**
1. **Hierarchical Feature Extraction:** DNNs automatically learn representations at multiple levels of abstraction (edges $\rightarrow$ strokes $\rightarrow$ loops $\rightarrow$ digit structures).
2. **Non-linear Capacity:** Logistic regression creates linear decision hyperplanes, which are too rigid for hand-drawn digit variances.
3. **End-to-end Optimization:** Random forest relies on orthogonal recursive partitioning (decision trees), which cannot compute smooth gradient-based parameter adjustments over continuous pixel manifolds.

---

## Part B: Reinforcement Learning & Q-Learning (Project 2)

### Q12: Define Reinforcement Learning and compare it with Supervised and Unsupervised Learning.
**Answer:**
- **Supervised Learning:** Learns from labeled pairs $(X, Y)$ provided by an external supervisor. Goal is error minimization.
- **Unsupervised Learning:** Discovers hidden structures, clusters, or distributions within unlabeled data $X$.
- **Reinforcement Learning:** An agent learns through trial-and-error by taking actions in an environment to maximize cumulative scalar rewards. There are no explicit labels, feedback is delayed, and actions influence future observations.

---

### Q13: What is a Markov Decision Process (MDP)?
**Answer:** An MDP is a mathematical framework for modeling sequential decision problems where outcomes are partly random and partly under agent control. It satisfies the **Markov Property**:
$$P(S_{t+1} = s' | S_t = s, A_t = a, S_{t-1} = s_{t-1}, \dots) = P(S_{t+1} = s' | S_t = s, A_t = a)$$
The future state depends exclusively on the current state and action, independent of prior history.

---

### Q14: What is the Bellman Optimality Equation for $Q^*(s, a)$?
**Answer:** The Bellman optimality equation states that the value of an action in a state under the optimal policy equals the expected immediate reward plus the discounted value of the optimal action in the next state:
$$Q^*(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

---

### Q15: Explain the Q-Learning update rule and identify its components.
**Answer:**
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
- $Q(s, a)$: Current estimated action-value.
- $\alpha$: Learning rate ($0 < \alpha \le 1$), determining the weight given to new information.
- $r$: Immediate scalar reward received after taking action $a$ in state $s$.
- $\gamma$: Discount factor ($0 \le \gamma < 1$), weighting the importance of future rewards.
- $\max_{a'} Q(s', a')$: Estimated maximum future value from state $s'$.
- $\left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$: **Temporal Difference (TD) Error** ($\delta_t$).

---

### Q16: Why is Q-Learning classified as an "Off-Policy" algorithm?
**Answer:** Because the policy used to generate behavior (the **behavior policy**, e.g., $\epsilon$-greedy which includes random exploratory actions) is different from the policy being evaluated and improved (the **target policy**, which is pure greedy $\max_{a'} Q(s', a')$). This allows Q-Learning to learn optimal policies while simultaneously exploring suboptimal actions.

---

### Q17: Contrast Q-Learning with SARSA.
**Answer:**
- **Q-Learning (Off-Policy):** Updates with respect to the maximum Q-value of the next state regardless of the action actually taken:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
- **SARSA (On-Policy):** Updates with respect to the action $a'$ actually selected by the behavior policy in state $s'$:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$$
SARSA is more conservative around hazards, whereas Q-Learning learns the optimal path directly.

---

### Q18: What is the Exploration vs. Exploitation dilemma?
**Answer:**
- **Exploration:** Trying untried or less frequented actions to discover potentially superior strategies or navigate unseen states.
- **Exploitation:** Choosing the action currently known to yield the highest expected return.  
Excessive exploitation leads the agent to get trapped in local optima; excessive exploration wastes time taking random actions. $\epsilon$-Greedy balances the two by starting with high exploration ($\epsilon = 1.0$) and gradually decaying $\epsilon$ to favor exploitation ($\epsilon \to 0.01$).

---

### Q19: Why was a step penalty ($-1.0$) introduced in the Maze environment?
**Answer:** Without a step penalty, the agent would receive $0$ reward for wandering aimlessly and $+100$ when eventually hitting the goal, so any path (even one with 100 loops) would yield the same total return ($+100$).  
By penalizing every valid step with $-1.0$, a path of length $L$ yields total return:
$$R_{\text{total}} = 100 - (L - 1) \times 1.0$$
To maximize total return, the agent is mathematically forced to minimize $L$, converging strictly to the **shortest path**.

---

### Q20: What is the role of the discount factor $\gamma$? What happens if $\gamma = 0$ or $\gamma \to 1$?
**Answer:**
- $\gamma$ determines the present value of future rewards.
- If $\gamma = 0$, the agent is completely myopic (short-sighted), only caring about immediate rewards $r_t$. In our maze, it would never learn to reach the goal because steps to the goal give $-1.0$ while the goal is 14 steps away.
- If $\gamma \to 1$ (e.g., $0.99$), the agent values far-future rewards almost as much as immediate ones, enabling the $+100$ goal reward to propagate back through intermediate states to the starting position.

---

### Q21: What is Gymnasium and how does its API function?
**Answer:** Gymnasium (formerly OpenAI Gym) is the standard open-source library for developing and comparing reinforcement learning algorithms. Its primary interface methods are:
- `reset(seed=None)`: Reinitializes environment and returns `(observation, info)`.
- `step(action)`: Applies action and transitions state, returning `(next_state, reward, terminated, truncated, info)`.
- `observation_space` & `action_space`: Formal descriptions of valid inputs/outputs (e.g., `Discrete(4)`).

---

### Q22: What is the curse of dimensionality in Tabular Q-Learning and how does Deep Q-Learning (DQN) solve it?
**Answer:** In tabular Q-learning, the Q-table requires storing a separate value for every state-action pair $|S| \times |A|$. In problems with continuous or high-dimensional states (such as raw video frames or Atari games), the state space is infinite or exponentially huge ($10^{20}+$ states), making table storage and exploration impossible.  
**Deep Q-Networks (DQN)** replace the lookup table with a deep neural network function approximator $Q(s, a; \theta) \approx Q^*(s, a)$, generalizing across similar states.
