"""Players, and the floor a trained agent has to clear.

The random player is not a baseline in the decorative sense. An agent that
cannot beat it has learned nothing, and the difficulty numbers that follow
would be measuring the random player wearing a neural network.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from match3.env import SWAPS, apply_swap, settle
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


def greedy_player(episode: Episode, rng: np.random.Generator) -> int:
    """Take whichever legal swap clears the most target-colour tiles right now.

    A one-move horizon with no plan behind it. It exists to bound the trained
    agent from above the way the random player bounds it from below: an agent
    that cannot beat pure greed has learned tactics but no strategy, and saying
    so is more useful than reporting a win over random alone.
    """
    best, best_gain = -1, -1
    for action in np.flatnonzero(episode.legal()):
        # settle() consumes randomness, so each trial gets its own generator
        # seeded the same way; otherwise the comparison measures luck.
        _, cleared = settle(
            apply_swap(episode.board, SWAPS[int(action)]),
            np.random.default_rng(0),
        )
        gain = int(cleared[episode.level.colour])
        if gain > best_gain:
            best, best_gain = int(action), gain
    return best
