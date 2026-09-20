from match3.levels import LEVELS
from match3.players import greedy_player, random_player
from match3.report import replay


def test_a_replay_records_a_board_for_every_move_played() -> None:
    result = replay(LEVELS[2], random_player, seed=0)

    assert len(result["frames"]) >= 2  # the starting board plus at least one move
    assert all(len(f["board"]) == 6 and len(f["board"][0]) == 6 for f in result["frames"])
    assert all(0 <= c < 4 for f in result["frames"] for row in f["board"] for c in row)


def test_a_replay_is_reproducible() -> None:
    assert replay(LEVELS[2], random_player, seed=3) == replay(LEVELS[2], random_player, seed=3)


def test_every_player_starts_from_the_same_board() -> None:
    """The comparison is only fair if the deal was the same."""
    a = replay(LEVELS[19], random_player, seed=7)["frames"][0]["board"]
    b = replay(LEVELS[19], greedy_player, seed=7)["frames"][0]["board"]

    assert a == b


def test_the_score_only_ever_rises() -> None:
    frames = replay(LEVELS[19], greedy_player, seed=7)["frames"]
    collected = [f["collected"] for f in frames]

    assert collected == sorted(collected)
    assert collected[0] == 0


def test_the_move_budget_only_ever_falls() -> None:
    frames = replay(LEVELS[19], random_player, seed=7)["frames"]
    left = [f["moves_left"] for f in frames]

    assert left == sorted(left, reverse=True)
    assert left[0] == LEVELS[19].moves


def test_a_won_replay_reached_the_target() -> None:
    result = replay(LEVELS[2], greedy_player, seed=7)

    if result["won"]:
        assert result["frames"][-1]["collected"] >= LEVELS[2].target
