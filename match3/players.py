"""Players, and the floor a trained agent has to clear.

The random player is not a baseline in the decorative sense. An agent that
cannot beat it has learned nothing, and the difficulty numbers that follow
would be measuring the random player wearing a neural network.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from match3.levels import Episode, Level

Player = Callable[[Episode, np.random.Generator], int]


def random_player(episode: Episode, rng: np.random.Generator) -> int:
    """Pick uniformly among the swaps that actually produce a match."""
    legal = np.flatnonzero(episode.legal())
    return int(rng.choice(legal))


def play(level: Level, player: Player, seed: int) -> bool:
    """Play one episode to the end. Returns whether it was won."""
    episode = Episode(level, seed=seed)
    rng = np.random.default_rng(seed)
    while not episode.done:
        if not episode.legal().any():
            # A board with no legal move is a dead end, not a win.
            return False
        episode.step(player(episode, rng))
    return episode.won


def solve_rate(level: Level, player: Player, episodes: int, seed: int) -> float:
    """The share of episodes this player wins on this level."""
    wins = sum(play(level, player, seed=seed * 10_000 + i) for i in range(episodes))
    return wins / episodes
