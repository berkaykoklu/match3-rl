"""The match-3 game itself.

This module deliberately imports no reinforcement learning library. The rules of
the game are worth testing on their own, and a bug here would otherwise look
like a bug in the agent.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ROWS = 6
COLS = 6
COLOURS = 4
RUN = 3  # tiles in a row needed to clear

Board = npt.NDArray[np.int8]
Mask = npt.NDArray[np.bool_]


def find_matches(board: Board) -> Mask:
    """Mark every cell belonging to a run of RUN or more of the same colour."""
    matched = np.zeros(board.shape, dtype=np.bool_)
    for axis in (0, 1):
        # Walking one axis at a time keeps this readable; a crossing run is
        # simply marked twice, which the boolean OR collapses.
        rolled = board if axis == 1 else board.T
        out = matched if axis == 1 else matched.T
        for r, row in enumerate(rolled):
            start = 0
            for c in range(1, len(row) + 1):
                if c == len(row) or row[c] != row[start]:
                    if c - start >= RUN:
                        out[r, start:c] = True
                    start = c
    return matched


def collapse(board: Board, matched: Mask, rng: np.random.Generator) -> Board:
    """Clear matched cells, drop survivors into the gaps, refill from the top."""
    out = board.copy()
    rows, cols = board.shape
    for c in range(cols):
        survivors = out[:, c][~matched[:, c]]
        gaps = rows - len(survivors)
        if gaps == 0:
            continue
        fresh = rng.integers(0, COLOURS, size=gaps, dtype=np.int8)
        out[:, c] = np.concatenate([fresh, survivors])
    return out


Swap = tuple[int, int, int, int]

SWAPS: list[Swap] = [
    *[(r, c, r, c + 1) for r in range(ROWS) for c in range(COLS - 1)],
    *[(r, c, r + 1, c) for r in range(ROWS - 1) for c in range(COLS)],
]


def apply_swap(board: Board, swap: Swap) -> Board:
    r1, c1, r2, c2 = swap
    out = board.copy()
    out[r1, c1], out[r2, c2] = board[r2, c2], board[r1, c1]
    return out


def settle(board: Board, rng: np.random.Generator) -> tuple[Board, npt.NDArray[np.int64]]:
    """Resolve cascades until the board is stable, counting what was cleared."""
    cleared = np.zeros(COLOURS, dtype=np.int64)
    out = board
    while True:
        matched = find_matches(out)
        if not matched.any():
            return out, cleared
        for colour in range(COLOURS):
            cleared[colour] += int((matched & (out == colour)).sum())
        out = collapse(out, matched, rng)


def legal_moves(board: Board) -> Mask:
    """Mark the swaps that produce at least one match."""
    return np.array(
        [find_matches(apply_swap(board, swap)).any() for swap in SWAPS],
        dtype=np.bool_,
    )


def new_board(rng: np.random.Generator) -> Board:
    """A starting board with nothing already matched on it."""
    board = rng.integers(0, COLOURS, size=(ROWS, COLS), dtype=np.int8)
    settled, _ = settle(board, rng)
    return settled
