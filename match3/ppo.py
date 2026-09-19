"""PPO, written out.

The point of this file is not to beat stable-baselines3. It is to be checked
against it: two implementations of the same algorithm on the same environment
should learn at the same rate, and if ours does not, ours is wrong.
"""

from __future__ import annotations

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
