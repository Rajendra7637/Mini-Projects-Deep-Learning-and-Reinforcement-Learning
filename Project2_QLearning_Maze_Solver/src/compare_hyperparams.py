"""
Hyperparameter tuning and comparative analysis for Q-Learning Maze Solver.
Studies variations in Learning Rate (alpha), Discount Factor (gamma), and Epsilon Decay.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from maze_env import MazeEnv
from q_learning_agent import QLearningAgent


def train_agent_experiment(
    env: MazeEnv,
    episodes: int = 350,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_decay: float = 0.992,
    seed: int = 42
) -> dict:
    """Trains an agent under a specific hyperparameter configuration."""
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.action_space.n,
        learning_rate=alpha,
        discount_factor=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
        seed=seed
    )

    rewards = []
    steps_list = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_r = 0.0
        step = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_r += reward
            step += 1

        agent.decay_epsilon()
        rewards.append(total_r)
        steps_list.append(step)

    # Compute rolling averages
    rolling_rewards = np.convolve(rewards, np.ones(30)/30, mode="valid").tolist()
    rolling_steps = np.convolve(steps_list, np.ones(30)/30, mode="valid").tolist()

    return {
        "final_reward": float(np.mean(rewards[-30:])),
        "final_steps": float(np.mean(steps_list[-30:])),
        "rolling_rewards": rolling_rewards,
        "rolling_steps": rolling_steps
    }


def run_hyperparameter_experiments():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    env = MazeEnv(max_steps=120)
    episodes = 350

    print("=" * 60)
    print(">>> Running Q-Learning Hyperparameter Tuning Experiments")
    print("=" * 60)

    # 1. Learning Rate (Alpha) Comparison
    print("\n[Experiment 1] Comparing Learning Rates (alpha = 0.02, 0.1, 0.5)...")
    alpha_results = {}
    for a in [0.02, 0.1, 0.5]:
        print(f"  Training with alpha = {a}...")
        alpha_results[f"alpha_{a}"] = train_agent_experiment(env, episodes=episodes, alpha=a, gamma=0.99, epsilon_decay=0.992)

    # 2. Discount Factor (Gamma) Comparison
    print("\n[Experiment 2] Comparing Discount Factors (gamma = 0.70, 0.90, 0.99)...")
    gamma_results = {}
    for g in [0.70, 0.90, 0.99]:
        print(f"  Training with gamma = {g}...")
        gamma_results[f"gamma_{g}"] = train_agent_experiment(env, episodes=episodes, alpha=0.1, gamma=g, epsilon_decay=0.992)

    # 3. Epsilon Decay Comparison
    print("\n[Experiment 3] Comparing Epsilon Decay (Fast=0.97, Balanced=0.992, Slow=0.998)...")
    decay_results = {}
    for d, label in [(0.97, "Fast (0.97)"), (0.992, "Balanced (0.992)"), (0.998, "Slow (0.998)")]:
        print(f"  Training with decay = {label}...")
        decay_results[label] = train_agent_experiment(env, episodes=episodes, alpha=0.1, gamma=0.99, epsilon_decay=d)

    # Plot multi-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Learning Rate Impact
    for key, data in alpha_results.items():
        val = key.split("_")[1]
        axes[0].plot(data["rolling_rewards"], label=f"α = {val}", linewidth=2)
    axes[0].set_title("Impact of Learning Rate (α)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode (30-ep rolling)", fontsize=11)
    axes[0].set_ylabel("Rolling Reward", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Panel 2: Discount Factor Impact
    for key, data in gamma_results.items():
        val = key.split("_")[1]
        axes[1].plot(data["rolling_rewards"], label=f"γ = {val}", linewidth=2)
    axes[1].set_title("Impact of Discount Factor (γ)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode (30-ep rolling)", fontsize=11)
    axes[1].set_ylabel("Rolling Reward", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    # Panel 3: Epsilon Decay Impact
    for label, data in decay_results.items():
        axes[2].plot(data["rolling_rewards"], label=label, linewidth=2)
    axes[2].set_title("Impact of Exploration Decay (ε)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Episode (30-ep rolling)", fontsize=11)
    axes[2].set_ylabel("Rolling Reward", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    comp_plot_path = os.path.join(outputs_dir, "hyperparameter_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"[Saved] Hyperparameter comparison plot: {comp_plot_path}")

    # Export metrics
    all_experiments = {
        "learning_rate_experiments": {k: {"final_reward": v["final_reward"], "final_steps": v["final_steps"]} for k, v in alpha_results.items()},
        "discount_factor_experiments": {k: {"final_reward": v["final_reward"], "final_steps": v["final_steps"]} for k, v in gamma_results.items()},
        "epsilon_decay_experiments": {k: {"final_reward": v["final_reward"], "final_steps": v["final_steps"]} for k, v in decay_results.items()}
    }
    with open(os.path.join(outputs_dir, "hyperparameter_comparison.json"), "w") as f:
        json.dump(all_experiments, f, indent=4)

    print("Hyperparameter tuning experiments completed successfully!")


if __name__ == "__main__":
    run_hyperparameter_experiments()
