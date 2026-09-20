"""Evidence for SPIKE_THRESHOLD.

Run before the threshold is chosen. Prints what each candidate would flag on a
real curve, so the constant in difficulty.py is a decision with a reason rather
than a round number someone liked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from match3.difficulty import SPIKE_SIGMAS, curve, noise_threshold, spikes
from match3.players import Player, greedy_player, random_player

PLAYERS: dict[str, Player] = {"random": random_player, "greedy": greedy_player}
CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def report(rates: list[float], episodes: int) -> None:
    drops = [rates[i - 1] - rates[i] for i in range(1, len(rates))]
    derived = noise_threshold(rates, episodes)
    print(f"levels: {len(rates)}   largest drop: {max(drops):.3f}   median: {np.median(drops):.3f}")
    print(f"noise-derived threshold: {derived:.3f}  ({SPIKE_SIGMAS:.0f} standard errors)")
    print()
    for threshold in (*CANDIDATES, derived):
        flagged = spikes(rates, threshold)
        share = len(flagged) / len(rates)
        levels = ", ".join(str(i + 1) for i in flagged[:8]) or "none"
        mark = "  <- derived" if threshold == derived else ""
        print(f"  {threshold:.3f} -> {len(flagged):2d} flagged ({share:4.0%})   {levels}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=["random", "agent", "greedy"], default="greedy")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--from-file",
        type=Path,
        help="read curves from a saved run instead of replaying every level",
    )
    args = parser.parse_args()

    if args.from_file:
        # Replaying 40 levels costs minutes; a finished run already holds the
        # same numbers, and calibrating against the curve the report will show
        # is more honest than calibrating against a fresh one that differs.
        saved = json.loads(args.from_file.read_text())
        rates = saved["solve_rate"][args.player]
        episodes = saved["episodes_per_level"][args.player]
    else:
        if args.player not in PLAYERS:
            parser.error(f"--player {args.player} needs --from-file")
        rates = curve(PLAYERS[args.player], episodes=args.episodes, seed=args.seed)
        episodes = args.episodes

    print(f"player: {args.player}   episodes per level: {episodes}\n")
    report(rates, episodes)


if __name__ == "__main__":
    main()
