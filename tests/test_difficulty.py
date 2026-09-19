from match3.difficulty import difficulty, spikes


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
