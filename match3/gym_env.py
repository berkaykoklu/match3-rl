"""The Gymnasium face of the game.

Kept apart from env.py on purpose: the rules of match-3 do not depend on any
reinforcement learning library, and a failure in one should never be mistaken
for a failure in the other.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from match3.env import COLOURS, COLS, ROWS, SWAPS
from match3.levels import LEVELS, Episode, Level

OBS_SIZE = ROWS * COLS * COLOURS + 2

# Progress reward over a solved level sums to 1.0, so a bonus of the same size
# makes finishing worth as much as everything collected on the way there.
# Without it the agent learns to clear tiles, which is not the same as learning
# to finish -- the last move of a level is worth more than its arithmetic.
WIN_BONUS = 1.0


def encode(episode: Episode) -> npt.NDArray[np.float32]:
    """One-hot board, then how much budget and how much goal is left.

    The board is one-hot rather than raw colour indices because the indices are
    names, not quantities: a network fed 0..3 would infer that orange exceeds
    blue and look for an ordering that is not there.
    """
    one_hot = np.zeros((ROWS * COLS, COLOURS), dtype=np.float32)
    one_hot[np.arange(ROWS * COLS), episode.board.ravel()] = 1.0
    remaining = max(episode.level.target - episode.collected, 0) / episode.level.target
    budget = episode.moves_left / episode.level.moves
    return np.concatenate([one_hot.ravel(), [budget, remaining]]).astype(np.float32)


class Match3Env(gym.Env[npt.NDArray[np.float32], np.int64]):
    """One level per episode, drawn from the pool given at construction."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, levels: list[Level] | None = None, seed: int = 0) -> None:
        super().__init__()
        self._levels = levels if levels is not None else LEVELS
        self._rng = np.random.default_rng(seed)
        self._episode: Episode | None = None
        self.action_space = spaces.Discrete(len(SWAPS))
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_SIZE,), dtype=np.float32)

    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("reset() must be called before step()")
        return self._episode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        level = self._levels[int(self._rng.integers(len(self._levels)))]
        self._episode = Episode(level, seed=int(self._rng.integers(2**31)))
        return encode(self._episode), {"level": level.number}

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """MaskablePPO looks for this exact method name."""
        return self.episode.legal()

    def step(
        self, action: np.int64
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        episode = self.episode
        gained = episode.step(int(action))
        reward = gained / episode.level.target
        if episode.won:
            reward += WIN_BONUS
        # A board with no legal move is over, however many moves are left.
        stuck = not episode.legal().any()
        terminated = episode.done or stuck
        return encode(episode), reward, terminated, False, {"won": episode.won}
