from pathlib import Path

from match3.compare import agreement, compare


def test_both_implementations_report_a_learning_curve(tmp_path: Path) -> None:
    curves = compare(steps=4096, seed=0, out_dir=tmp_path)

    assert set(curves) == {"ours", "sb3"}
    assert len(curves["ours"]) > 0
    assert len(curves["sb3"]) > 0


def test_the_two_curves_are_sampled_at_the_same_cadence(tmp_path: Path) -> None:
    """Point-for-point comparison is only meaningful if the points line up."""
    curves = compare(steps=4096, seed=0, out_dir=tmp_path)

    assert abs(len(curves["ours"]) - len(curves["sb3"])) <= 1


def test_identical_curves_show_no_gap() -> None:
    same = [1.0, 1.2, 1.4, 1.6]
    summary = agreement({"ours": same, "sb3": same})

    assert summary["mean_absolute_gap"] == 0.0
    assert summary["ours_final"] == summary["sb3_final"]


def test_a_real_gap_is_reported_as_one() -> None:
    summary = agreement({"ours": [0.0, 0.0, 0.0], "sb3": [1.0, 1.0, 1.0]})

    assert summary["mean_absolute_gap"] == 1.0
    assert summary["noise_in_sb3"] == 0.0  # a flat curve has no spread to hide behind


def test_curves_of_different_lengths_are_compared_on_the_shorter_one() -> None:
    summary = agreement({"ours": [1.0, 2.0], "sb3": [1.0, 2.0, 3.0, 4.0]})

    assert summary["updates_compared"] == 2
