import numpy as np

from match3.env import (
    COLOURS,
    COLS,
    ROWS,
    SWAPS,
    Board,
    apply_swap,
    collapse,
    find_matches,
    legal_moves,
    new_board,
    settle,
)


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


def test_cleared_cells_are_refilled_so_the_board_stays_full() -> None:
    rng = np.random.default_rng(0)
    board = plain()
    board[5, 0:3] = 3
    matched = find_matches(board)

    result = collapse(board, matched, rng)

    assert result.shape == board.shape
    assert ((result >= 0) & (result < COLOURS)).all()


def test_tiles_above_a_cleared_cell_fall_into_it() -> None:
    rng = np.random.default_rng(0)
    board = plain()
    board[5, 0:3] = 3          # bottom row will clear
    board[4, 0] = 2            # this specific tile should land on the bottom row
    matched = find_matches(board)

    result = collapse(board, matched, rng)

    assert result[5, 0] == 2


def test_collapse_does_not_mutate_the_board_it_was_given() -> None:
    rng = np.random.default_rng(0)
    board = plain()
    board[5, 0:3] = 3
    before = board.copy()

    collapse(board, find_matches(board), rng)

    assert (board == before).all()


def test_collapsing_nothing_leaves_the_board_alone() -> None:
    rng = np.random.default_rng(0)
    board = plain()
    empty = np.zeros(board.shape, dtype=np.bool_)

    assert (collapse(board, empty, rng) == board).all()


def test_there_is_one_swap_for_every_adjacent_pair() -> None:
    assert len(SWAPS) == 2 * ROWS * (COLS - 1)  # 60 on a 6x6 board


def test_a_swap_exchanges_exactly_two_cells() -> None:
    board = plain()
    swapped = apply_swap(board, (0, 0, 0, 1))

    assert swapped[0, 0] == board[0, 1]
    assert swapped[0, 1] == board[0, 0]
    assert (swapped[1:] == board[1:]).all()


def test_settling_leaves_no_matches_behind() -> None:
    rng = np.random.default_rng(3)
    board = rng.integers(0, COLOURS, size=(ROWS, COLS), dtype=np.int8)

    settled, _ = settle(board, rng)

    assert not find_matches(settled).any()


def test_settling_counts_the_tiles_it_cleared() -> None:
    rng = np.random.default_rng(0)
    board = plain()
    board[5, 0:3] = 3

    _, cleared = settle(board, rng)

    assert cleared.shape == (COLOURS,)
    assert cleared[3] >= 3


def test_a_new_board_starts_with_no_matches() -> None:
    for seed in range(20):
        board = new_board(np.random.default_rng(seed))
        assert not find_matches(board).any()


def test_a_legal_move_actually_produces_a_match() -> None:
    rng = np.random.default_rng(7)
    board = new_board(rng)
    legal = legal_moves(board)

    assert legal.shape == (len(SWAPS),)
    for i, is_legal in enumerate(legal):
        produced = find_matches(apply_swap(board, SWAPS[i])).any()
        assert produced == is_legal
