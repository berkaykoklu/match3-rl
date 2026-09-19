"""Evidence for SPIKE_THRESHOLD.

Run before the threshold is chosen. Prints what each candidate would flag on a
real curve, so the constant in difficulty.py is a decision with a reason rather
than a round number someone liked.
"""

from __future__ import annotations

import argparse

import numpy as np

from match3.difficulty import curve, spikes
from match3.players import Player, greedy_player, random_player

PLAYERS: dict[str, Player] = {"random": random_player, "greedy": greedy_player}
CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def report(rates: list[float]) -> None:
    drops = [rates[i - 1] - rates[i] for i in range(1, len(rates))]
    print(f"levels: {len(rates)}   largest drop: {max(drops):.3f}   median: {np.median(drops):.3f}")
    print()
    for threshold in CANDIDATES:
        flagged = spikes(rates, threshold)
        share = len(flagged) / len(rates)
        levels = ", ".join(str(i + 1) for i in flagged[:8]) or "none"
        print(f"  {threshold:.2f} -> {len(flagged):2d} flagged ({share:4.0%})   {levels}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=sorted(PLAYERS), default="greedy")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rates = curve(PLAYERS[args.player], episodes=args.episodes, seed=args.seed)
    print(f"player: {args.player}   episodes per level: {args.episodes}\n")
    report(rates)


if __name__ == "__main__":
    main()
