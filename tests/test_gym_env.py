import numpy as np
import pytest
from gymnasium import spaces

from match3.env import COLOURS, COLS, ROWS, SWAPS
from match3.gym_env import OBS_SIZE, Match3Env, encode
from match3.levels import LEVELS, Episode, Level


def test_the_observation_has_the_declared_shape_and_range() -> None:
    env = Match3Env(seed=0)
    obs, _ = env.reset(seed=0)

    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert (obs >= 0).all() and (obs <= 1).all()


def test_the_one_hot_board_marks_exactly_one_colour_per_cell() -> None:
    obs = encode(Episode(LEVELS[0], seed=0))
    board_part = obs[: ROWS * COLS * COLOURS].reshape(ROWS * COLS, COLOURS)

    assert (board_part.sum(axis=1) == 1.0).all()


def test_the_one_hot_board_marks_the_colour_that_is_actually_there() -> None:
    episode = Episode(LEVELS[0], seed=4)
    obs = encode(episode)
    board_part = obs[: ROWS * COLS * COLOURS].reshape(ROWS, COLS, COLOURS)

    assert (board_part.argmax(axis=2) == episode.board).all()


def test_the_action_space_matches_the_swap_table() -> None:
    space = Match3Env(seed=0).action_space

    assert isinstance(space, spaces.Discrete)  # one action per swap, not a box
    assert space.n == len(SWAPS)


def test_the_mask_marks_at_least_one_legal_move_on_a_fresh_board() -> None:
    env = Match3Env(seed=0)
    env.reset(seed=0)

    mask = env.action_masks()

    assert mask.shape == (len(SWAPS),)
    assert mask.any()


def test_a_masked_step_is_accepted_and_returns_a_finite_reward() -> None:
    env = Match3Env(seed=0)
    env.reset(seed=0)
    action = int(np.flatnonzero(env.action_masks())[0])

    obs, reward, terminated, truncated, _ = env.step(np.int64(action))

    assert obs.shape == (OBS_SIZE,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_an_episode_always_ends() -> None:
    env = Match3Env(seed=0)
    env.reset(seed=0)
    for _ in range(200):
        mask = env.action_masks()
        if not mask.any():
            break
        _, _, terminated, truncated, _ = env.step(np.int64(np.flatnonzero(mask)[0]))
        if terminated or truncated:
            break
    else:
        raise AssertionError("episode ran past its move budget")


def test_winning_pays_more_than_the_step_rewards_alone() -> None:
    env = Match3Env(levels=[Level(number=1, colour=0, target=1, moves=30)], seed=0)
    env.reset(seed=0)
    total = 0.0
    while True:
        mask = env.action_masks()
        if not mask.any():
            break
        _, reward, terminated, truncated, _ = env.step(np.int64(np.flatnonzero(mask)[0]))
        total += reward
        if terminated or truncated:
            break

    assert total > 1.0  # the win bonus is on top of progress reward


def test_stepping_before_reset_is_refused_rather_than_crashing_obscurely() -> None:
    with pytest.raises(RuntimeError):
        Match3Env(seed=0).action_masks()
