"""
Tabular Q-Learning Agent implementation for Maze Navigation.
Implements the Bellman optimality equation and Epsilon-Greedy exploration strategy.
"""

from typing import Tuple, Optional, Dict, Any
import json
import numpy as np


class QLearningAgent:
    """
    Q-Learning Agent using tabular representation.
    
    Update Rule (Bellman Equation):
        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    def __init__(
        self,
        num_states: int,
        num_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = 42
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)

    def choose_action(self, state: int, greedy: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy.
        If greedy=True, strictly selects the action with highest Q-value.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))

        # Greedy choice with random tie-breaking
        q_values = self.q_table[state]
        max_val = np.max(q_values)
        best_actions = np.where(q_values == max_val)[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool
    ) -> float:
        """
        Performs Q-value update based on the Bellman equation.
        Returns the TD error magnitude for monitoring.
        """
        current_q = self.q_table[state, action]
        best_next_q = 0.0 if terminated else np.max(self.q_table[next_state])
        td_target = reward + (self.gamma * best_next_q)
        td_error = td_target - current_q

        # Q-update
        self.q_table[state, action] += self.alpha * td_error
        return float(abs(td_error))

    def decay_epsilon(self):
        """Decays the exploration rate epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Saves Q-table to a NumPy file (.npy)."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str):
        """Loads Q-table from a NumPy file (.npy)."""
        self.q_table = np.load(filepath)

    def get_policy(self) -> np.ndarray:
        """Returns the greedy action for each state."""
        return np.argmax(self.q_table, axis=1)

    def get_state_values(self) -> np.ndarray:
        """Returns the maximum Q-value for each state: V(s) = max_a Q(s, a)."""
        return np.max(self.q_table, axis=1)


if __name__ == "__main__":
    agent = QLearningAgent(num_states=64, num_actions=4)
    print("Q-Table initialized with shape:", agent.q_table.shape)
    action = agent.choose_action(0)
    print("Sample action chosen for state 0:", action)
