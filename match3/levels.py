"""Levels, and one playthrough of one.

A level is a goal and a move budget. Everything that makes a level harder --
a rarer colour, a bigger target, fewer moves -- lives in these four numbers,
which is what makes the difficulty curve something the agent can measure
rather than something the designer asserts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from match3.env import SWAPS, Board, apply_swap, legal_moves, new_board, settle


@dataclass(frozen=True)
class Level:
    number: int
    colour: int
    target: int
    moves: int


def _build_levels() -> list[Level]:
    """Forty levels whose targets rise and whose move budgets tighten.

    The progression is deliberately smooth: no level is designed to be a wall.
    Any spike the agent finds is therefore a property of the game, not of the
    level table -- which is the whole point of measuring instead of asserting.
    """
    return [Level(number=i + 1, colour=i % 4, target=8 + i, moves=25 - i // 4) for i in range(40)]


LEVELS: list[Level] = _build_levels()


class Episode:
    """One playthrough of one level."""

    def __init__(self, level: Level, seed: int) -> None:
        self.level = level
        self._rng = np.random.default_rng(seed)
        self.board: Board = new_board(self._rng)
        self.moves_left: int = level.moves
        self.collected: int = 0

    @property
    def done(self) -> bool:
        return self.won or self.moves_left <= 0

    @property
    def won(self) -> bool:
        return self.collected >= self.level.target

    def legal(self) -> npt.NDArray[np.bool_]:
        return legal_moves(self.board)

    def step(self, action: int) -> int:
        """Play a swap. Returns how many target-colour tiles it cleared."""
        if not self.legal()[action]:
            raise ValueError(f"action {action} does not produce a match")
        swapped = apply_swap(self.board, SWAPS[action])
        self.board, cleared = settle(swapped, self._rng)
        self.moves_left -= 1
        gained = int(cleared[self.level.colour])
        self.collected += gained
        return gained
