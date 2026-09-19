"""Training with a reference implementation.

This runs first, before the hand-written PPO, and the order is the point. If the
agent fails to learn here, the environment is at fault; only once a known-good
algorithm has learned this game does a failure point at our own PPO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from match3.gym_env import Match3Env, encode
from match3.levels import Episode
from match3.players import Player

DEFAULT_STEPS = 200_000


def train(steps: int, seed: int, out: Path) -> Path:
    env = Match3Env(seed=seed)
    model = MaskablePPO("MlpPolicy", env, seed=seed, verbose=0)
    model.learn(total_timesteps=steps)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return out.with_suffix(".zip")


def sb3_player(model: MaskablePPO) -> Player:
    """Adapt a trained model to the plain Player signature."""

    def play(episode: Episode, rng: np.random.Generator) -> int:
        action, _ = model.predict(encode(episode), action_masks=episode.legal(), deterministic=True)
        return int(action)

    return play


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/sb3"))
    args = parser.parse_args()
    saved = train(args.steps, args.seed, args.out)
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
