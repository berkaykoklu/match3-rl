import torch

from match3.env import SWAPS
from match3.gym_env import OBS_SIZE
from match3.ppo import Policy


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
