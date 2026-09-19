from pathlib import Path

import numpy as np

from match3.levels import LEVELS, Episode
from match3.train_sb3 import sb3_player, train


def test_training_produces_a_loadable_model(tmp_path: Path) -> None:
    out = train(steps=256, seed=0, out=tmp_path / "tiny")

    assert out.exists()


def test_a_trained_model_only_proposes_legal_moves(tmp_path: Path) -> None:
    from sb3_contrib import MaskablePPO

    path = train(steps=256, seed=0, out=tmp_path / "tiny")
    player = sb3_player(MaskablePPO.load(path))

    rng = np.random.default_rng(0)
    episode = Episode(LEVELS[0], seed=0)
    for _ in range(5):
        if episode.done or not episode.legal().any():
            break
        action = player(episode, rng)
        assert episode.legal()[action], "the mask was not respected"
        episode.step(action)
