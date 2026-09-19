import torch

from match3.env import SWAPS
from match3.gym_env import OBS_SIZE
from match3.ppo import Policy, gae


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
