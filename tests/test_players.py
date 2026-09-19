import numpy as np

from match3.levels import LEVELS, Episode, Level
from match3.players import play, random_player, solve_rate


def test_the_random_player_only_ever_picks_a_legal_move() -> None:
    rng = np.random.default_rng(0)
    episode = Episode(LEVELS[0], seed=0)
    for _ in range(5):
        if episode.done or not episode.legal().any():
            break
        action = random_player(episode, rng)
        assert episode.legal()[action]
        episode.step(action)


def test_a_playthrough_finishes_and_reports_a_result() -> None:
    assert isinstance(play(LEVELS[0], random_player, seed=0), bool)


def test_a_trivial_level_is_always_solved() -> None:
    trivial = Level(number=0, colour=0, target=1, moves=30)
    assert solve_rate(trivial, random_player, episodes=20, seed=0) == 1.0


def test_an_impossible_level_is_never_solved() -> None:
    impossible = Level(number=0, colour=0, target=10**6, moves=5)
    assert solve_rate(impossible, random_player, episodes=20, seed=0) == 0.0


def test_the_solve_rate_is_a_proportion() -> None:
    rate = solve_rate(LEVELS[10], random_player, episodes=30, seed=1)
    assert 0.0 <= rate <= 1.0


def test_the_same_seed_gives_the_same_answer() -> None:
    a = solve_rate(LEVELS[5], random_player, episodes=20, seed=42)
    b = solve_rate(LEVELS[5], random_player, episodes=20, seed=42)
    assert a == b


def test_a_different_seed_can_give_a_different_answer() -> None:
    """Guards against a seed that is accepted and then quietly ignored."""
    rates = {solve_rate(LEVELS[19], random_player, episodes=20, seed=s) for s in range(6)}
    assert len(rates) > 1
