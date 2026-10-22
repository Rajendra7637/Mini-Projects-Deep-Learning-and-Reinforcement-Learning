"""
Custom Gymnasium-compatible Maze Environment for Reinforcement Learning.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Default 8x8 Maze Layout:
# 'S': Start
# 'G': Goal
# '#': Wall / Obstacle
# 'T': Trap / Penalty Hazard
# '.': Valid Empty Cell
DEFAULT_MAZE = [
    "S . . . # . . .",
    ". # # . # . # .",
    ". . . . . . # .",
    "# # # . # # # .",
    ". . . . . . . .",
    ". # # # # # . #",
    ". . . T . # . .",
    "# # . . . . . G"
]


class MazeEnv(gym.Env):
    """
    Gymnasium-compliant 2D Grid Maze Environment.
    
    Actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT

    Rewards:
        +100.0 : Reaching the goal 'G' (Terminates episode)
        -15.0  : Stepping into a trap 'T'
        -5.0   : Attempting to walk into a wall '#' or boundary (agent stays in place)
        -1.0   : Normal step into valid path '.' (Reward shaping to incentivize shortest path)
    """
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    ACTION_MAP = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1),  # LEFT
    }
    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

    def __init__(
        self,
        maze_layout: Optional[List[str]] = None,
        max_steps: int = 150,
        step_penalty: float = -1.0,
        wall_penalty: float = -5.0,
        trap_penalty: float = -15.0,
        goal_reward: float = 100.0,
        render_mode: Optional[str] = "ansi"
    ):
        super().__init__()

        layout = maze_layout or DEFAULT_MAZE
        self.grid = [row.split() if " " in row else list(row) for row in layout]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.max_steps = max_steps

        # Reward structure
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.trap_penalty = trap_penalty
        self.goal_reward = goal_reward
        self.render_mode = render_mode

        # Define Observation and Action spaces
        self.num_states = self.rows * self.cols
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(4)

        # Locate Start and Goal positions
        self.start_pos = None
        self.goal_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "S":
                    self.start_pos = (r, c)
                elif self.grid[r][c] == "G":
                    self.goal_pos = (r, c)

        if not self.start_pos or not self.goal_pos:
            raise ValueError("Maze must contain both a Start ('S') and a Goal ('G')!")

        self.current_pos = self.start_pos
        self.step_count = 0

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Converts (row, col) coordinate to 1D scalar state index."""
        return pos[0] * self.cols + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Converts 1D scalar state index to (row, col) coordinate."""
        return state // self.cols, state % self.cols

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        self.step_count = 0

        info = {
            "pos": self.current_pos,
            "steps": self.step_count
        }
        return self.pos_to_state(self.current_pos), info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        dr, dc = self.ACTION_MAP[action]
        nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc

        terminated = False
        truncated = self.step_count >= self.max_steps

        # Check out of bounds
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            reward = self.wall_penalty
            # Agent stays in place
        elif self.grid[nr][nc] == "#":
            reward = self.wall_penalty
            # Hit a wall, stays in place
        else:
            # Valid move
            self.current_pos = (nr, nc)
            cell_type = self.grid[nr][nc]

            if cell_type == "G":
                reward = self.goal_reward
                terminated = True
            elif cell_type == "T":
                reward = self.trap_penalty
            else:
                reward = self.step_penalty

        state = self.pos_to_state(self.current_pos)
        info = {
            "pos": self.current_pos,
            "steps": self.step_count,
            "reached_goal": (self.current_pos == self.goal_pos)
        }

        return state, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        output = []
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if (r, c) == self.current_pos:
                    row_str.append("A")  # Agent
                else:
                    row_str.append(self.grid[r][c])
            output.append(" ".join(row_str))
        rendered = "\n".join(output)
        if self.render_mode == "ansi":
            return rendered
        else:
            print(rendered)
            return None


if __name__ == "__main__":
    env = MazeEnv()
    obs, info = env.reset()
    print("Initial Maze State:")
    print(env.render())
    print(f"Num States: {env.num_states}, Action Space: {env.action_space}")
