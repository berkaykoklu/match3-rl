"""Everything the site shows, written once to a file.

Nothing runs on a server: the site reads this JSON. A visitor watching a player
work through a level is watching an episode that actually happened, recorded
here move by move.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from match3.difficulty import SPIKE_SIGMAS, noise_threshold, skill_sensitivity, spikes
from match3.env import COLOURS, COLS, ROWS
from match3.levels import LEVELS, Episode, Level
from match3.players import Player, greedy_player, random_player
from match3.train_sb3 import sb3_player

OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "results.json"

# One level from each part of the curve, so the replays show the range rather
# than three versions of the same story.
REPLAY_LEVELS = (3, 20, 31)
REPLAY_SEED = 11


def replay(level: Level, player: Player, seed: int) -> dict[str, Any]:
    """Every board state of one episode, with the score as it stood.

    All three players start from the same seed, so the first board is identical
    and the run diverges only where their choices do.
    """
    episode = Episode(level, seed=seed)
    rng = np.random.default_rng(seed)
    frames = [{"board": episode.board.tolist(), "collected": 0, "moves_left": episode.moves_left}]
    while not episode.done and episode.legal().any():
        episode.step(player(episode, rng))
        frames.append(
            {
                "board": episode.board.tolist(),
                "collected": episode.collected,
                "moves_left": episode.moves_left,
            }
        )
    return {"won": episode.won, "frames": frames}


def build(curves_file: Path, compare_file: Path | None) -> dict[str, Any]:
    raw = json.loads(curves_file.read_text())
    rates = raw["solve_rate"]
    episodes = raw["episodes_per_level"]

    agent = sb3_player(MaskablePPO.load("runs/sb3_long"))
    players: dict[str, Player] = {
        "random": random_player,
        "agent": agent,
        "greedy": greedy_player,
    }

    thresholds = {n: noise_threshold(rates[n], episodes[n]) for n in rates}
    out: dict[str, Any] = {
        "board": {"rows": ROWS, "cols": COLS, "colours": COLOURS},
        "levels": raw["levels"],
        "solve_rate": rates,
        "episodes_per_level": episodes,
        # The threshold is not one number: a curve measured with fewer episodes
        # wobbles more, so it has to clear a higher bar before a drop counts.
        "spike_threshold": thresholds,
        "spike_sigmas": SPIKE_SIGMAS,
        "spikes": {
            name: [raw["levels"][i]["number"] for i in spikes(rates[name], thresholds[name])]
            for name in rates
        },
        "skill_sensitivity": skill_sensitivity(rates["random"], rates["greedy"]),
        "replays": {
            str(number): {
                name: replay(LEVELS[number - 1], player, REPLAY_SEED)
                for name, player in players.items()
            }
            for number in REPLAY_LEVELS
        },
        "replay_levels": list(REPLAY_LEVELS),
    }

    if compare_file and compare_file.exists():
        out["learning_curves"] = json.loads(compare_file.read_text())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("runs/curves_raw.json"))
    parser.add_argument("--compare", type=Path, default=Path("runs/compare/curves.json"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    result = build(args.curves, args.compare)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    for name, flagged in result["spikes"].items():
        print(f"  {name}: threshold {result['spike_threshold'][name]:.3f} -> {flagged or 'none'}")


if __name__ == "__main__":
    main()
