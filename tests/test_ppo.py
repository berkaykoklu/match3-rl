from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from match3.env import SWAPS
from match3.gym_env import OBS_SIZE
from match3.ppo import Policy, clipped_objective, gae, masked_logits, train


def test_the_policy_returns_one_logit_per_action_and_one_value() -> None:
    policy = Policy(OBS_SIZE, len(SWAPS))
    obs = torch.zeros((4, OBS_SIZE))

    logits, value = policy(obs)

    assert logits.shape == (4, len(SWAPS))
    assert value.shape == (4,)


def test_the_two_heads_share_a_trunk() -> None:
    """Both questions need the board read; reading it twice would be waste."""
    policy = Policy(OBS_SIZE, len(SWAPS))
    trunk_params = {id(p) for p in policy.trunk.parameters()}

    assert not trunk_params & {id(p) for p in policy.actor.parameters()}
    assert not trunk_params & {id(p) for p in policy.critic.parameters()}
    assert len(trunk_params) > 0


def test_the_value_head_returns_a_scalar_per_observation_not_a_column() -> None:
    """A trailing 1-dim here silently broadcasts later and ruins the loss."""
    value = Policy(OBS_SIZE, len(SWAPS))(torch.zeros((7, OBS_SIZE)))[1]

    assert value.ndim == 1


def test_with_no_discounting_the_return_is_the_sum_of_rewards() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.tensor([0.0, 0.0, 1.0])

    _, returns = gae(rewards, values, dones, torch.tensor(0.0), gamma=1.0, lam=1.0)

    assert abs(returns[0].item() - 3.0) < 1e-5


def test_an_episode_boundary_stops_the_return_bleeding_backwards() -> None:
    """Reward earned after a reset belongs to the next episode, not this one."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.tensor([0.0, 1.0, 1.0])  # episode ends at index 1

    _, returns = gae(rewards, values, dones, torch.tensor(0.0), gamma=1.0, lam=1.0)

    assert abs(returns[0].item() - 2.0) < 1e-5


def test_a_perfect_critic_leaves_no_advantage() -> None:
    """If the critic predicted exactly what happened, there is nothing to learn."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([1.0, 1.0, 1.0])
    dones = torch.tensor([0.0, 0.0, 1.0])

    advantages, _ = gae(rewards, values, dones, torch.tensor(0.0), gamma=1.0, lam=1.0)

    assert abs(advantages[2].item()) < 1e-5


def test_a_better_than_expected_step_gets_a_positive_advantage() -> None:
    rewards = torch.tensor([2.0])
    values = torch.tensor([0.5])
    dones = torch.tensor([1.0])

    advantages, _ = gae(rewards, values, dones, torch.tensor(0.0), gamma=1.0, lam=1.0)

    assert advantages[0].item() > 0


def test_a_worse_than_expected_step_gets_a_negative_advantage() -> None:
    rewards = torch.tensor([0.1])
    values = torch.tensor([0.9])
    dones = torch.tensor([1.0])

    advantages, _ = gae(rewards, values, dones, torch.tensor(0.0), gamma=1.0, lam=1.0)

    assert advantages[0].item() < 0


def test_discounting_makes_a_distant_reward_worth_less() -> None:
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.zeros(3)
    dones = torch.tensor([0.0, 0.0, 1.0])

    _, returns = gae(rewards, values, dones, torch.tensor(0.0), gamma=0.5, lam=1.0)

    assert abs(returns[0].item() - 0.25) < 1e-5  # 1.0 discounted twice


def test_masking_makes_illegal_actions_impossible_to_sample() -> None:
    probs = torch.softmax(
        masked_logits(torch.tensor([[1.0, 5.0, 1.0]]), torch.tensor([[True, False, True]])),
        dim=-1,
    )

    assert probs[0, 1].item() == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_masking_leaves_the_legal_actions_in_proportion() -> None:
    probs = torch.softmax(
        masked_logits(torch.tensor([[2.0, 9.0, 2.0]]), torch.tensor([[True, False, True]])),
        dim=-1,
    )

    assert abs(probs[0, 0].item() - probs[0, 2].item()) < 1e-6


def test_an_unchanged_policy_has_a_ratio_of_one_and_loses_nothing_to_clipping() -> None:
    ratio = torch.tensor([1.0])
    advantage = torch.tensor([2.0])

    assert clipped_objective(ratio, advantage).item() == -2.0


def test_a_huge_increase_on_a_good_action_is_capped() -> None:
    """Without the cap one lucky episode would swing the policy."""
    advantage = torch.tensor([1.0])

    capped = clipped_objective(torch.tensor([5.0]), advantage)
    modest = clipped_objective(torch.tensor([1.2]), advantage)

    assert abs(capped.item() - modest.item()) < 1e-6


def test_a_huge_decrease_on_a_bad_action_is_capped() -> None:
    advantage = torch.tensor([-1.0])

    capped = clipped_objective(torch.tensor([0.01]), advantage)
    modest = clipped_objective(torch.tensor([0.8]), advantage)

    assert abs(capped.item() - modest.item()) < 1e-6


def test_a_small_change_is_left_alone() -> None:
    """Inside the trust region the objective is the plain one."""
    assert abs(clipped_objective(torch.tensor([1.1]), torch.tensor([2.0])).item() + 2.2) < 1e-5


class _OneGoodAction(gym.Env[np.ndarray, np.int64]):
    """Three actions, one of them pays. The smallest task PPO should solve.

    Match-3 takes about a hundred updates before learning clears the noise,
    which is no use as a test. Here the right answer is unambiguous, so a run
    that fails to find it means the loop itself is wired wrong.
    """

    GOOD = 1

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):  # type: ignore[no-untyped-def]
        reward = 1.0 if int(action) == self.GOOD else 0.0
        return np.zeros(2, dtype=np.float32), reward, True, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(3, dtype=np.bool_)


def test_the_loop_learns_a_task_whose_answer_is_obvious(tmp_path: Path) -> None:
    out, curve = train(
        steps=4096, seed=0, out=tmp_path / "p.pt",
        make_env=lambda _seed: _OneGoodAction(), batch=512,
    )

    assert curve[-1] > 0.9, f"never found the paying action: {curve}"
    assert curve[-1] > curve[0], f"no improvement: {curve[0]:.2f} -> {curve[-1]:.2f}"
    assert out.exists()


def test_the_trained_policy_prefers_the_paying_action(tmp_path: Path) -> None:
    train(
        steps=4096, seed=0, out=tmp_path / "p.pt",
        make_env=lambda _seed: _OneGoodAction(), batch=512,
    )
    policy = Policy(2, 3)
    policy.load_state_dict(torch.load(tmp_path / "p.pt"))

    logits, _ = policy(torch.zeros((1, 2)))

    assert int(logits.argmax().item()) == _OneGoodAction.GOOD
