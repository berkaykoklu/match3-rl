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

from match3.env import SWAPS, Board, Mask, apply_swap, legal_moves, new_board, settle


@dataclass(frozen=True)
class Level:
    """A goal and a budget -- not a particular board.

    Each attempt deals a fresh arrangement, the way a real match-3 deals fresh
    colours onto a fixed layout. Pinning the board instead would make every
    attempt identical, and a deterministic player would then score 0% or 100%
    on every level: a verdict rather than the rate a difficulty curve needs.
    """

    number: int
    colour: int
    target: int
    moves: int


# Target tiles a random player collects per move, measured over 30 episodes.
# Cascades are why this is far above three-quarters of a single match: one swap
# often sets off several. The level table is built against this number because
# a table built against a guess turned out to be trivially easy -- every level
# was clearable by chance, which leaves no difficulty curve to measure.
RANDOM_TILES_PER_MOVE = 2.38

# Share of what a random player could collect that each level actually demands,
# swept from comfortable to out of reach. The sweep is deliberately linear: the
# demand curve is smooth by construction, so any step in the *solve* curve is a
# property of the game rather than something the table put there.
EASIEST_DEMAND = 0.35
HARDEST_DEMAND = 1.35


def _build_levels() -> list[Level]:
    """Forty levels whose targets rise and whose move budgets tighten."""
    levels: list[Level] = []
    for i in range(40):
        moves = 25 - i // 4
        demand = EASIEST_DEMAND + (HARDEST_DEMAND - EASIEST_DEMAND) * i / 39
        target = round(moves * RANDOM_TILES_PER_MOVE * demand)
        levels.append(Level(number=i + 1, colour=i % 4, target=target, moves=moves))
    return levels


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

    def step(self, action: int, rounds: list[tuple[Mask, Board]] | None = None) -> int:
        """Play a swap. Returns how many target-colour tiles it cleared.

        `rounds` is for replays: pass a list and each cascade step lands in it.
        """
        if not self.legal()[action]:
            raise ValueError(f"action {action} does not produce a match")
        swapped = apply_swap(self.board, SWAPS[action])
        self.board, cleared = settle(swapped, self._rng, rounds)
        self.moves_left -= 1
        gained = int(cleared[self.level.colour])
        self.collected += gained
        return gained
