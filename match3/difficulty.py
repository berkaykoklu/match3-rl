"""Turning solve rates into a difficulty curve, and finding the walls in it."""

from __future__ import annotations

import math

from match3.levels import LEVELS
from match3.players import Player, solve_rate

# How many standard errors a drop must clear before it counts as a wall.
# Three is the usual bar for "not chance". A fixed threshold was the first
# design and it was wrong: measured with 60 episodes a curve wobbles by about
# 7 points between neighbours, with 200 episodes by about 5, so one number
# flags noise on the noisy curve and nothing on the quiet one. See calibrate.py.
SPIKE_SIGMAS = 3.0


def difficulty(rate: float) -> float:
    return 1.0 - rate


def noise_threshold(rates: list[float], episodes: int, sigmas: float = SPIKE_SIGMAS) -> float:
    """The smallest level-to-level drop that measurement noise cannot explain.

    Each rate is a proportion out of `episodes`, so it carries a standard error
    of sqrt(p(1-p)/n); a drop involves two of them, hence the sqrt(2).
    """
    mean = sum(rates) / len(rates)
    standard_error = math.sqrt(mean * (1 - mean) / episodes)
    return sigmas * standard_error * math.sqrt(2)


def spikes(rates: list[float], threshold: float) -> list[int]:
    """Indices where the solve rate falls by more than `threshold`."""
    return [i for i in range(1, len(rates)) if rates[i - 1] - rates[i] > threshold]


def curve(player: Player, episodes: int, seed: int) -> list[float]:
    """Solve rate for every level, in order."""
    return [solve_rate(level, player, episodes, seed) for level in LEVELS]


def skill_sensitivity(weak: list[float], strong: list[float]) -> list[float]:
    """How much a level separates a considered player from a careless one.

    Difficulty is not one number per level: it is one number per level *and*
    player. The spread between two players is the part a designer can act on --
    a level everyone clears teaches nothing, and a level nobody clears teaches
    nothing either. Only the gap says the level rewards thinking.
    """
    return [s - w for w, s in zip(weak, strong, strict=True)]
