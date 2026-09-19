"""PPO, written out.

The point of this file is not to beat stable-baselines3. It is to be checked
against it: two implementations of the same algorithm on the same environment
should learn at the same rate, and if ours does not, ours is wrong.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Policy(nn.Module):
    """One trunk, two heads: what to do, and how good this looks.

    The heads share a trunk because both questions start by reading the board,
    and two separate networks would each learn that reading from scratch.
    """

    def __init__(self, obs_size: int, n_actions: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh()
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.trunk(obs)
        # squeeze keeps the value a scalar per observation; a trailing 1-dim
        # broadcasts against the advantage later and quietly corrupts the loss.
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


# PPO's published defaults, unchanged. Tuning these before the implementation
# is verified against the reference would confuse "our code is wrong" with
# "our settings are wrong".
GAMMA = 0.99  # a reward one step later is worth this much of one now
LAM = 0.95    # how far ahead the advantage looks before it stops caring


def gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float = GAMMA,
    lam: float = LAM,
) -> tuple[Tensor, Tensor]:
    """How much better each step went than the critic expected.

    Walked backwards, because a step's worth depends on what followed it. The
    `not_done` factor is what keeps one episode's ending from leaking into the
    step before it: at a boundary the future is cut off rather than carried.
    """
    advantages = torch.zeros_like(rewards)
    running = torch.tensor(0.0)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values
