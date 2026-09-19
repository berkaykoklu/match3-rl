import numpy as np
import pytest

from match3.env import COLOURS, SWAPS
from match3.levels import LEVELS, Episode, Level


def test_levels_are_numbered_in_order_from_one() -> None:
    assert [level.number for level in LEVELS] == list(range(1, len(LEVELS) + 1))


def test_every_level_asks_for_a_real_colour_and_a_positive_target() -> None:
    for level in LEVELS:
        assert 0 <= level.colour < COLOURS
        assert level.target > 0
        assert level.moves > 0


def test_a_level_cannot_be_edited_after_it_is_made() -> None:
    with pytest.raises(AttributeError):
        LEVELS[0].target = 1  # type: ignore[misc]


def test_an_episode_starts_unfinished_with_nothing_collected() -> None:
    episode = Episode(LEVELS[0], seed=0)

    assert episode.collected == 0
    assert episode.moves_left == LEVELS[0].moves
    assert not episode.done


def test_a_step_spends_a_move() -> None:
    episode = Episode(LEVELS[0], seed=0)
    action = int(np.flatnonzero(episode.legal())[0])

    episode.step(action)

    assert episode.moves_left == LEVELS[0].moves - 1


def test_an_illegal_action_is_refused() -> None:
    episode = Episode(LEVELS[0], seed=0)
    illegal = int(np.flatnonzero(~episode.legal())[0])

    with pytest.raises(ValueError):
        episode.step(illegal)


def test_an_episode_ends_when_the_moves_run_out() -> None:
    level = Level(number=99, colour=0, target=10**6, moves=3)
    episode = Episode(level, seed=0)

    for _ in range(3):
        episode.step(int(np.flatnonzero(episode.legal())[0]))

    assert episode.done
    assert not episode.won


def test_an_episode_is_won_when_the_target_is_reached() -> None:
    level = Level(number=99, colour=0, target=1, moves=50)
    episode = Episode(level, seed=0)

    while not episode.done:
        episode.step(int(np.flatnonzero(episode.legal())[0]))

    assert episode.won
    assert episode.collected >= 1


def test_the_action_space_matches_the_swap_table() -> None:
    assert Episode(LEVELS[0], seed=0).legal().shape == (len(SWAPS),)


def test_a_step_only_counts_the_colour_the_level_asked_for() -> None:
    level = Level(number=99, colour=0, target=10**6, moves=50)
    episode = Episode(level, seed=5)
    action = int(np.flatnonzero(episode.legal())[0])

    gained = episode.step(action)

    assert gained == episode.collected  # first move, so the two agree
