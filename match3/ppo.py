"""PPO, written out.

The point of this file is not to beat stable-baselines3. It is to be checked
against it: two implementations of the same algorithm on the same environment
should learn at the same rate, and if ours does not, ours is wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor

from match3.gym_env import Match3Env, encode
from match3.levels import Episode
from match3.players import Player


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


# How far one update may move a single action's probability. 0.2 is PPO's
# published value: the whole point of the algorithm is that this is small.
CLIP = 0.2


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Push illegal actions to negative infinity so softmax gives them zero.

    Not a penalty the agent has to learn from -- a move it never sees. Learning
    which swaps are legal would cost thousands of steps that teach nothing
    about playing well.
    """
    return logits.masked_fill(~mask, float("-inf"))


def clipped_objective(ratio: Tensor, advantage: Tensor) -> Tensor:
    """PPO's loss: follow the advantage, but refuse to take a large step.

    The minimum is what makes this pessimistic. A ratio that ran away is worth
    no more than one that stopped at the edge of the trust region, so there is
    nothing to gain by running away -- in either direction.
    """
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage
    return -torch.min(unclipped, clipped).mean()


# Rollout and update settings, all PPO defaults.
BATCH = 2048      # steps collected before each update
EPOCHS = 4        # passes over that batch
LR = 3e-4
VALUE_COEF = 0.5    # how much the critic's error counts against the actor's
ENTROPY_COEF = 0.01  # pressure to stay undecided, so exploration survives
MAX_GRAD_NORM = 0.5


def train(
    steps: int,
    seed: int,
    out: Path,
    make_env: Callable[[int], Any] | None = None,
    batch: int = BATCH,
) -> tuple[Path, list[float]]:
    """Collect, score, update -- repeated until the step budget runs out.

    On-policy: every batch is thrown away after the update that used it,
    because the ratio in the clipped objective is only meaningful against the
    policy that actually collected the data.

    `make_env` exists so the loop can be checked on a task whose right answer
    is obvious. Match-3 takes roughly a hundred updates before learning shows
    above the noise, which is too slow to be a test and too vague to be
    evidence that the machinery -- advantage sign, ratio, update direction --
    is wired correctly.
    """
    torch.manual_seed(seed)
    env = make_env(seed) if make_env else Match3Env(seed=seed)
    obs_space, act_space = env.observation_space, env.action_space
    assert isinstance(obs_space, spaces.Box) and isinstance(act_space, spaces.Discrete)
    policy = Policy(int(obs_space.shape[0]), int(act_space.n))
    optimiser = torch.optim.Adam(policy.parameters(), lr=LR)

    obs = torch.from_numpy(env.reset(seed=seed)[0])
    curve: list[float] = []
    episode_reward, finished = 0.0, []

    for _ in range(max(steps // batch, 1)):
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, mask_buf = ([] for _ in range(7))

        for _ in range(batch):  # ---- collect
            mask = torch.from_numpy(env.action_masks())
            with torch.no_grad():
                logits, value = policy(obs.unsqueeze(0))
                dist = torch.distributions.Categorical(
                    logits=masked_logits(logits, mask.unsqueeze(0))
                )
                action = dist.sample()
            step_obs, reward, terminated, truncated, _ = env.step(np.int64(action.item()))

            obs_buf.append(obs)
            act_buf.append(action.squeeze(0))
            logp_buf.append(dist.log_prob(action).squeeze(0))
            val_buf.append(value.squeeze(0))
            mask_buf.append(mask)
            rew_buf.append(torch.tensor(float(reward)))
            done = terminated or truncated
            done_buf.append(torch.tensor(float(done)))

            episode_reward += float(reward)
            if done:
                finished.append(episode_reward)
                episode_reward = 0.0
                step_obs, _ = env.reset()
            obs = torch.from_numpy(step_obs)

        with torch.no_grad():  # ---- score
            last_value = policy(obs.unsqueeze(0))[1].squeeze(0)
        advantages, returns = gae(
            torch.stack(rew_buf), torch.stack(val_buf), torch.stack(done_buf), last_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_obs, batch_act = torch.stack(obs_buf), torch.stack(act_buf)
        batch_logp, batch_mask = torch.stack(logp_buf), torch.stack(mask_buf)

        for _ in range(EPOCHS):  # ---- update
            logits, values = policy(batch_obs)
            dist = torch.distributions.Categorical(logits=masked_logits(logits, batch_mask))
            ratio = torch.exp(dist.log_prob(batch_act) - batch_logp)
            loss = (
                clipped_objective(ratio, advantages)
                + VALUE_COEF * ((values - returns) ** 2).mean()
                - ENTROPY_COEF * dist.entropy().mean()
            )
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimiser.step()

        curve.append(float(np.mean(finished[-20:])) if finished else 0.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out)
    return out, curve


def ppo_player(policy: Policy) -> Player:
    """Adapt a trained policy to the plain Player signature."""

    def play(episode: Episode, rng: np.random.Generator) -> int:
        obs = torch.from_numpy(encode(episode)).unsqueeze(0)
        mask = torch.from_numpy(episode.legal()).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(obs)
        return int(masked_logits(logits, mask).argmax(dim=-1).item())

    return play
