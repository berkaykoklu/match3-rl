"""Turning solve rates into a difficulty curve, and finding the walls in it."""

from __future__ import annotations

from match3.levels import LEVELS
from match3.players import Player, solve_rate

# A drop this large between consecutive levels is a wall rather than a slope.
# Provisional until calibrate.py has been run against a real curve; the comment
# is rewritten with that run's numbers rather than left as a round guess.
SPIKE_THRESHOLD = 0.20


def difficulty(rate: float) -> float:
    return 1.0 - rate


def spikes(rates: list[float], threshold: float = SPIKE_THRESHOLD) -> list[int]:
    """Indices where the solve rate falls by more than `threshold`."""
    return [i for i in range(1, len(rates)) if rates[i - 1] - rates[i] > threshold]


def curve(player: Player, episodes: int, seed: int) -> list[float]:
    """Solve rate for every level, in order."""
    return [solve_rate(level, player, episodes, seed) for level in LEVELS]
