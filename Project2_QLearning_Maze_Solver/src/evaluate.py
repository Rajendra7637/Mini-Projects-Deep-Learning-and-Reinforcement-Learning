"""
Evaluation and visualization script for Q-Learning Maze Solver.
Generates learning curves, solution path visualization, and Q-table state-value heatmaps.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from maze_env import MazeEnv
from q_learning_agent import QLearningAgent


def evaluate_greedy_policy(env: MazeEnv, agent: QLearningAgent):
    """
    Evaluates the greedy policy (epsilon=0) from Start to Goal.
    Returns:
        path: List of (row, col) coordinates traversed.
        actions_taken: List of action names taken.
        total_reward: Accumulated reward.
        success: Boolean flag indicating if goal was reached.
    """
    state, info = env.reset()
    path = [env.start_pos]
    actions_taken = []
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < env.max_steps:
        action = agent.choose_action(state, greedy=True)
        actions_taken.append(MazeEnv.ACTION_NAMES[action])
        next_state, reward, terminated, truncated, step_info = env.step(action)
        path.append(step_info["pos"])
        total_reward += reward
        state = next_state
        step += 1
        done = terminated or truncated

    success = (path[-1] == env.goal_pos)
    return path, actions_taken, total_reward, success


def plot_training_metrics(history_path: str, output_path: str):
    """
    Plots Reward curves, Steps-to-Goal curves, and Epsilon decay.
    """
    if not os.path.exists(history_path):
        print(f"Warning: History file not found at {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    episodes = range(1, len(history["episode_rewards"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Total Reward per Episode
    axes[0].plot(episodes, history["episode_rewards"], color="#a0c4ff", alpha=0.4, label="Episode Reward")
    axes[0].plot(episodes, history["rolling_reward_50"], color="#0056b3", linewidth=2.5, label="50-Ep Rolling Avg")
    axes[0].set_title("Reward per Episode (Learning Convergence)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Total Reward", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # 2. Steps per Episode (Convergence to shortest path)
    axes[1].plot(episodes, history["episode_steps"], color="#ffadad", alpha=0.4, label="Steps per Ep")
    # Compute rolling average of steps
    rolling_steps = np.convolve(history["episode_steps"], np.ones(50)/50, mode="valid")
    axes[1].plot(range(50, len(history["episode_steps"]) + 1), rolling_steps, color="#d90429", linewidth=2.5, label="50-Ep Rolling Avg")
    axes[1].set_title("Steps to Reach Goal vs Episodes", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].set_ylabel("Steps", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    # 3. Epsilon Decay Curve
    axes[2].plot(episodes, history["episode_epsilons"], color="#2b9348", linewidth=2.5)
    axes[2].set_title("Exploration Rate (Epsilon) Decay", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Episode", fontsize=11)
    axes[2].set_ylabel("Epsilon (ε)", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Training metrics plot: {output_path}")


def plot_maze_solution(env: MazeEnv, path: list, output_path: str):
    """
    Renders 2D visual representation of the maze, obstacle walls, traps,
    and overlays the optimal shortest path discovered by Q-Learning.
    """
    rows, cols = env.rows, env.cols
    fig, ax = plt.subplots(figsize=(8, 8))

    # Cell color mapping
    for r in range(rows):
        for c in range(cols):
            cell = env.grid[r][c]
            if cell == "#":
                color = "#2d3436"  # Wall (Dark)
                text = "WALL"
                text_color = "white"
            elif cell == "S":
                color = "#00b894"  # Start (Green)
                text = "START"
                text_color = "white"
            elif cell == "G":
                color = "#fdcb6e"  # Goal (Gold)
                text = "GOAL"
                text_color = "black"
            elif cell == "T":
                color = "#d63031"  # Trap (Red)
                text = "TRAP"
                text_color = "white"
            else:
                color = "#f8f9fa"  # Path (Light)
                text = ""
                text_color = "black"

            rect = patches.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor="#636e72", linewidth=1.5)
            ax.add_patch(rect)
            if text:
                ax.text(c + 0.5, rows - 1 - r + 0.5, text, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    # Plot solution path with directional arrows
    path_coords = [(c + 0.5, rows - 1 - r + 0.5) for r, c in path]
    xs, ys = zip(*path_coords)

    # Draw connecting path line
    ax.plot(xs, ys, color="#0984e3", linewidth=4, alpha=0.85, zorder=5, label="Optimal Path")

    # Draw step index & arrows
    for i in range(len(path_coords) - 1):
        x1, y1 = path_coords[i]
        x2, y2 = path_coords[i + 1]
        dx, dy = (x2 - x1) * 0.45, (y2 - y1) * 0.45
        ax.arrow(x1, y1, dx, dy, head_width=0.15, head_length=0.12, fc="#0984e3", ec="#0984e3", zorder=6)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(f"Q-Learning Learned Shortest Path ({len(path)-1} Steps to Goal)", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Maze solution path plot: {output_path}")


def plot_q_value_heatmap(env: MazeEnv, agent: QLearningAgent, output_path: str):
    """
    Plots the state-value function V(s) = max_a Q(s, a) across the grid.
    Illustrates how the value gradient directs the agent smoothly toward the goal.
    """
    rows, cols = env.rows, env.cols
    value_grid = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            if env.grid[r][c] == "#":
                value_grid[r, c] = np.nan
            else:
                state = env.pos_to_state((r, c))
                value_grid[r, c] = np.max(agent.q_table[state])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        value_grid,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar=True,
        linewidths=1,
        linecolor="white",
        ax=ax,
        mask=np.isnan(value_grid)
    )
    ax.set_title("Q-Learning State-Value Function V(s) = max_a Q(s,a)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Grid Column", fontsize=11)
    ax.set_ylabel("Grid Row", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Q-table state-value heatmap: {output_path}")


def evaluate_maze():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    q_table_path = os.path.join(models_dir, "q_table.npy")
    history_path = os.path.join(models_dir, "q_learning_history.json")

    env = MazeEnv()
    agent = QLearningAgent(num_states=env.num_states, num_actions=env.action_space.n)
    agent.load(q_table_path)

    print("=" * 60)
    print(">>> Evaluating Q-Learning Agent with Greedy Policy")
    print("=" * 60)

    path, actions, total_reward, success = evaluate_greedy_policy(env, agent)
    print(f"Start Position:       {env.start_pos}")
    print(f"Goal Position:        {env.goal_pos}")
    print(f"Goal Reached:         {success}")
    print(f"Shortest Path Length: {len(path) - 1} steps")
    print(f"Total Test Reward:    {total_reward:.2f}")
    print(f"Action Sequence:      {' -> '.join(actions)}")
    print(f"Path Coordinates:     {path}")

    # Visualizations
    plot_training_metrics(history_path, os.path.join(outputs_dir, "training_metrics.png"))
    plot_maze_solution(env, path, os.path.join(outputs_dir, "maze_solution_path.png"))
    plot_q_value_heatmap(env, agent, os.path.join(outputs_dir, "q_table_heatmap.png"))

    # Save summary
    summary = {
        "start_pos": env.start_pos,
        "goal_pos": env.goal_pos,
        "goal_reached": success,
        "path_length_steps": len(path) - 1,
        "total_test_reward": total_reward,
        "action_sequence": actions,
        "path_coordinates": [list(p) for p in path]
    }
    with open(os.path.join(outputs_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    evaluate_maze()
