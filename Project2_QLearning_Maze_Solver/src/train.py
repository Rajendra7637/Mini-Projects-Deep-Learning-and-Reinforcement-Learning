"""
Training script for the Q-Learning Maze Solver agent.
Trains agent over episodes, records learning statistics, and saves the learned Q-table.
"""

import os
import json
import time
import numpy as np

from maze_env import MazeEnv
from q_learning_agent import QLearningAgent


def train_q_learning(
    episodes: int = 600,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.992,
    max_steps_per_episode: int = 150,
    save_dir: str = "../models"
) -> dict:
    """
    Executes the episodic Q-Learning training loop.
    """
    print("=" * 60)
    print(">>> Starting Q-Learning Agent Training on Maze Environment")
    print(f"Episodes: {episodes} | Alpha: {alpha} | Gamma: {gamma} | Epsilon Decay: {epsilon_decay}")
    print("=" * 60)

    env = MazeEnv(max_steps=max_steps_per_episode)
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.action_space.n,
        learning_rate=alpha,
        discount_factor=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=42
    )

    history = {
        "episode_rewards": [],
        "episode_steps": [],
        "episode_epsilons": [],
        "goal_reached": [],
        "rolling_reward_50": []
    }

    start_time = time.time()

    for ep in range(1, episodes + 1):
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        history["episode_rewards"].append(float(total_reward))
        history["episode_steps"].append(int(steps))
        history["episode_epsilons"].append(float(agent.epsilon))
        history["goal_reached"].append(bool(step_info["reached_goal"]))

        # 50-episode rolling average reward
        recent_avg = np.mean(history["episode_rewards"][-50:])
        history["rolling_reward_50"].append(float(recent_avg))

        if ep % 50 == 0 or ep == episodes:
            success_rate = np.mean(history["goal_reached"][-50:]) * 100
            avg_steps = np.mean(history["episode_steps"][-50:])
            print(f"Episode {ep:03d}/{episodes:03d} | "
                  f"Avg Reward: {recent_avg:6.2f} | "
                  f"Avg Steps: {avg_steps:5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Success: {success_rate:5.1f}%")

    elapsed_time = time.time() - start_time
    history["training_time_seconds"] = elapsed_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")

    # Save Q-table and metadata
    os.makedirs(save_dir, exist_ok=True)
    q_table_path = os.path.join(save_dir, "q_table.npy")
    agent.save(q_table_path)
    print(f"[Saved] Q-table: {q_table_path}")

    # Export policy
    policy = agent.get_policy().tolist()
    policy_path = os.path.join(save_dir, "optimal_policy.json")
    with open(policy_path, "w") as f:
        json.dump({
            "rows": env.rows,
            "cols": env.cols,
            "actions": MazeEnv.ACTION_NAMES,
            "policy": policy
        }, f, indent=4)
    print(f"[Saved] Optimal policy: {policy_path}")

    # Export history
    history_path = os.path.join(save_dir, "q_learning_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[Saved] Training history: {history_path}")

    return history


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    train_q_learning(episodes=600, save_dir=models_dir)


if __name__ == "__main__":
    main()
