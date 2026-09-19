"""Two implementations of PPO, one environment.

Agreement is the evidence. A hand-written algorithm that tracks a reference
implementation is probably right; one that does not is wrong somewhere, and the
gap says so before anyone has to trust the numbers built on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from match3.gym_env import Match3Env
from match3.ppo import BATCH
from match3.ppo import train as train_ours

# Both implementations update every this many steps, so a snapshot taken at the
# end of each rollout lands at the same point on both curves.
WINDOW = 20  # episodes averaged per snapshot, matching ppo.train


class _CurveRecorder(BaseCallback):
    """Snapshot the mean of the last WINDOW episode rewards after each rollout.

    ppo.train records exactly this, at exactly this cadence. Reading SB3's
    final buffer instead and slicing it into chunks would produce a curve of
    the same length that does not line up with ours point for point.
    """

    def __init__(self) -> None:
        super().__init__()
        self.curve: list[float] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        finished = [float(info["r"]) for info in (self.model.ep_info_buffer or [])]
        self.curve.append(float(np.mean(finished[-WINDOW:])) if finished else 0.0)


def sb3_curve(steps: int, seed: int, out: Path, verbose: int = 0) -> list[float]:
    env = Match3Env(seed=seed)
    model = MaskablePPO("MlpPolicy", env, seed=seed, verbose=verbose, n_steps=BATCH)
    recorder = _CurveRecorder()
    model.learn(total_timesteps=steps, callback=recorder)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return recorder.curve


def compare(steps: int, seed: int, out_dir: Path, verbose: int = 0) -> dict[str, list[float]]:
    """Train both on the same environment and return their learning curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _, ours = train_ours(steps=steps, seed=seed, out=out_dir / "ppo.pt")
    return {"ours": ours, "sb3": sb3_curve(steps, seed, out_dir / "sb3_compare", verbose)}


def agreement(curves: dict[str, list[float]]) -> dict[str, Any]:
    """How close the two curves ended up, in terms a reader can check."""
    n = min(len(curves["ours"]), len(curves["sb3"]))
    ours, sb3 = np.array(curves["ours"][:n]), np.array(curves["sb3"][:n])
    half = max(n // 2, 1)
    return {
        "updates_compared": n,
        "ours_final": float(ours[-half:].mean()),
        "sb3_final": float(sb3[-half:].mean()),
        "mean_absolute_gap": float(np.abs(ours - sb3).mean()),
        # Both curves are noisy, so the spread of one is the yardstick for
        # whether the gap between them means anything.
        "noise_in_sb3": float(sb3.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/compare"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    curves = compare(args.steps, args.seed, args.out, verbose=0 if args.quiet else 1)
    (args.out / "curves.json").write_text(json.dumps(curves, indent=2))

    summary = agreement(curves)
    print(f"\nupdates compared:   {summary['updates_compared']}")
    print(f"ours  (last half):  {summary['ours_final']:.3f}")
    print(f"sb3   (last half):  {summary['sb3_final']:.3f}")
    print(f"mean absolute gap:  {summary['mean_absolute_gap']:.3f}")
    print(f"noise in sb3:       {summary['noise_in_sb3']:.3f}")
    verdict = "within noise" if summary["mean_absolute_gap"] < summary["noise_in_sb3"] else "APART"
    print(f"\nthe two curves are {verdict}")


if __name__ == "__main__":
    main()
