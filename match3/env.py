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
