import numpy as np

from match3.env import COLS, ROWS, Board, find_matches


def plain() -> Board:
    """A board with no runs in either direction, to place test shapes onto.

    A striped board (every row identical) looks empty read across and is solid
    read down, which silently matches everything. Alternating on row+column is
    the cheapest pattern that is genuinely run-free both ways.
    """
    rows, cols = np.indices((ROWS, COLS))
    return ((rows + cols) % 2).astype(np.int8)


def test_the_plain_board_really_has_no_matches() -> None:
    assert not find_matches(plain()).any()


def test_a_horizontal_run_of_three_is_a_match() -> None:
    board = plain()
    board[2, 1:4] = 3

    matched = find_matches(board)

    assert matched[2, 1:4].all()
    assert matched.sum() == 3


def test_a_vertical_run_of_three_is_a_match() -> None:
    board = plain()
    board[1:4, 2] = 3

    matched = find_matches(board)

    assert matched[1:4, 2].all()
    assert matched.sum() == 3


def test_a_run_of_two_is_not_a_match() -> None:
    board = plain()
    board[2, 1:3] = 3

    assert not find_matches(board).any()


def test_a_run_of_five_matches_every_cell_in_it() -> None:
    board = plain()
    board[0, 0:5] = 2

    matched = find_matches(board)

    assert matched[0, 0:5].all()
    assert matched.sum() == 5


def test_crossing_runs_are_both_matched() -> None:
    board = plain()
    board[2, 1:4] = 3
    board[1:4, 2] = 3

    matched = find_matches(board)

    assert matched.sum() == 5  # three across plus three down, sharing one cell
