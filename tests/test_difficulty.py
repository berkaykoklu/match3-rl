from match3.difficulty import difficulty, noise_threshold, skill_sensitivity, spikes


def test_difficulty_is_the_complement_of_the_solve_rate() -> None:
    assert difficulty(1.0) == 0.0
    assert difficulty(0.0) == 1.0
    assert abs(difficulty(0.75) - 0.25) < 1e-9


def test_a_sharp_drop_is_a_spike() -> None:
    assert spikes([0.9, 0.85, 0.4, 0.38], threshold=0.2) == [2]


def test_a_gentle_slope_is_not_a_spike() -> None:
    assert spikes([0.9, 0.8, 0.7, 0.6], threshold=0.2) == []


def test_an_improvement_is_never_a_spike() -> None:
    assert spikes([0.4, 0.9], threshold=0.2) == []


def test_every_drop_is_reported_not_just_the_first() -> None:
    assert spikes([0.9, 0.5, 0.45, 0.1], threshold=0.2) == [1, 3]


def test_a_single_level_has_no_spikes() -> None:
    assert spikes([0.5], threshold=0.2) == []


def test_the_threshold_separates_drops_that_are_clearly_either_side() -> None:
    """Deliberately not testing the exact boundary.

    A drop written as 0.9 - 0.7 is 0.20000000000000007 in binary floating
    point while 0.5 - 0.3 is exactly 0.2, so behaviour at the boundary depends
    on which numbers produced it. Pinning that down would test the float
    representation rather than the rule.
    """
    assert spikes([0.9, 0.69], threshold=0.2) == [1]
    assert spikes([0.9, 0.71], threshold=0.2) == []


def test_a_level_everyone_clears_separates_nobody() -> None:
    assert skill_sensitivity([1.0], [1.0]) == [0.0]


def test_a_level_nobody_clears_separates_nobody() -> None:
    assert skill_sensitivity([0.0], [0.0]) == [0.0]


def test_the_gap_is_the_measure_not_the_difficulty() -> None:
    """Two levels of very different difficulty can separate players equally."""
    easy_ish = skill_sensitivity([0.5], [0.9])
    hard_ish = skill_sensitivity([0.1], [0.5])

    assert easy_ish == hard_ish == [0.4]


def test_mismatched_curves_are_refused_rather_than_silently_truncated() -> None:
    import pytest

    with pytest.raises(ValueError):
        skill_sensitivity([0.1, 0.2], [0.5])


def test_more_episodes_make_the_threshold_stricter() -> None:
    """A quieter measurement should be allowed to call a smaller drop real."""
    rates = [0.5] * 10

    assert noise_threshold(rates, episodes=400) < noise_threshold(rates, episodes=60)


def test_a_curve_near_the_ceiling_needs_a_smaller_drop_to_convince() -> None:
    """A proportion close to 1 barely wobbles, so less movement means more."""
    assert noise_threshold([0.97] * 10, 100) < noise_threshold([0.5] * 10, 100)


def test_the_threshold_is_three_standard_errors_of_a_difference() -> None:
    import math

    rates = [0.5] * 10
    expected = 3.0 * math.sqrt(0.25 / 100) * math.sqrt(2)

    assert abs(noise_threshold(rates, 100) - expected) < 1e-12
