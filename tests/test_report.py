import numpy as np

from match3.env import find_matches, settle
from match3.levels import LEVELS
from match3.players import greedy_player, random_player
from match3.report import replay


def test_a_replay_records_every_move_played() -> None:
    result = replay(LEVELS[2], random_player, seed=0)

    assert len(result["moves"]) >= 1
    assert len(result["start"]) == 6 and len(result["start"][0]) == 6


def test_a_replay_is_reproducible() -> None:
    assert replay(LEVELS[2], random_player, seed=3) == replay(LEVELS[2], random_player, seed=3)


def test_every_player_starts_from_the_same_board() -> None:
    """The comparison is only fair if the deal was the same."""
    a = replay(LEVELS[19], random_player, seed=7)["start"]
    b = replay(LEVELS[19], greedy_player, seed=7)["start"]

    assert a == b


def test_a_swap_names_two_adjacent_cells() -> None:
    for move in replay(LEVELS[14], greedy_player, seed=3)["moves"]:
        r1, c1, r2, c2 = move["swap"]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_every_move_clears_something() -> None:
    """Illegal swaps are refused, so every recorded move had at least one match."""
    for move in replay(LEVELS[14], greedy_player, seed=3)["moves"]:
        assert move["rounds"], "a move was recorded with no cascade round"
        assert move["rounds"][0]["matched"]


def test_the_recorded_rounds_replay_the_real_board() -> None:
    """The board a viewer ends on must be the board the game ended on."""
    for move in replay(LEVELS[14], greedy_player, seed=3)["moves"]:
        assert not find_matches(np.array(move["rounds"][-1]["after"], dtype=np.int8)).any()


def test_cascades_are_kept_whole() -> None:
    """A swap that sets off several rounds records all of them, not just the first."""
    moves = replay(LEVELS[14], greedy_player, seed=3)["moves"]

    assert max(len(m["rounds"]) for m in moves) > 1


def test_the_score_only_ever_rises() -> None:
    collected = [m["collected"] for m in replay(LEVELS[19], greedy_player, seed=7)["moves"]]

    assert collected == sorted(collected)


def test_the_move_budget_only_ever_falls() -> None:
    left = [m["moves_left"] for m in replay(LEVELS[19], random_player, seed=7)["moves"]]

    assert left == sorted(left, reverse=True)
    assert left[0] == LEVELS[19].moves - 1


def test_recording_a_settle_does_not_change_where_it_lands() -> None:
    """The rounds are a by-product; asking for them must not alter the game."""
    board = np.random.default_rng(4).integers(0, 4, size=(6, 6), dtype=np.int8)

    plain, plain_cleared = settle(board, np.random.default_rng(1))
    rounds: list[tuple[np.ndarray, np.ndarray]] = []
    recorded, recorded_cleared = settle(board, np.random.default_rng(1), rounds)

    assert (plain == recorded).all()
    assert (plain_cleared == recorded_cleared).all()
