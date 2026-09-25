"""Tests for the RQ2 aggregation.

These are the numbers that go into the paper, so the arithmetic is checked
against cases worked out by hand rather than against the implementation's own
output.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from analyze_rq2 import (  # noqa: E402
    cliffs_delta,
    complete_tasks,
    holm,
    mean,
    paired,
    run_tests,
    stdev,
)


# ── effect size ─────────────────────────────────────────────────────────────

def test_no_overlap_at_all_is_the_extreme():
    delta, reading = cliffs_delta([1, 2, 3], [4, 5, 6])

    assert delta == -1.0
    assert reading == "large"


def test_identical_samples_have_no_effect():
    delta, reading = cliffs_delta([3, 3, 3], [3, 3, 3])

    assert delta == 0.0
    assert reading == "negligible"


def test_direction_follows_the_argument_order():
    better, _ = cliffs_delta([5, 5, 4], [2, 1, 2])
    worse, _ = cliffs_delta([2, 1, 2], [5, 5, 4])

    assert better == pytest.approx(1.0)
    assert worse == pytest.approx(-1.0)


def test_the_romano_thresholds_are_the_ones_reported():
    # A difference can clear significance on a five-point scale and still be
    # too small for a developer to notice, so the reading is part of the claim.
    # Each case below is one winning value against a constant baseline, so
    # delta is just the share of winning pairs.
    ones = [1] * 10

    assert cliffs_delta([2] + [1] * 9, ones) == (pytest.approx(0.10), "negligible")
    assert cliffs_delta([2, 2] + [1] * 8, ones) == (pytest.approx(0.20), "small")
    assert cliffs_delta([2] * 4 + [1] * 6, ones) == (pytest.approx(0.40), "medium")
    assert cliffs_delta([2] * 6 + [1] * 4, ones) == (pytest.approx(0.60), "large")


# ── multiplicity ────────────────────────────────────────────────────────────

def test_holm_worked_by_hand():
    # p = .01 -> 3x, then .03 -> 2x = .06, then .04 -> 1x = .04 raised to .06
    # by the monotonicity rule.
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_never_lets_a_later_test_look_better_than_an_earlier_one():
    adjusted = holm([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    ordered = sorted(adjusted)

    assert adjusted == ordered


def test_holm_caps_at_one():
    assert max(holm([0.9, 0.8, 0.95])) == 1.0


def test_a_single_test_is_not_adjusted():
    assert holm([0.042]) == pytest.approx([0.042])


# ── which rows enter the tables ─────────────────────────────────────────────

def _row(task, condition, acc=4, comp=4, category="navigation"):
    return {
        "task_id": task, "condition": condition, "category": category,
        "accuracy_score": acc, "completeness_score": comp,
    }


def test_a_task_missing_one_condition_is_left_out():
    rows = [
        _row("a", "full"), _row("a", "summary"), _row("a", "hybrid"),
        _row("b", "full"), _row("b", "summary"),
    ]

    assert complete_tasks(rows) == {"a"}


def test_pairing_lines_up_the_same_task_not_the_same_position():
    rows = [
        _row("b", "full", acc=2), _row("a", "full", acc=5),
        _row("a", "summary", acc=4), _row("b", "summary", acc=1),
    ]

    xs, ys = paired(rows, "accuracy_score", "full", "summary")

    assert xs == [5, 2]   # a, b in task order
    assert ys == [4, 1]


def test_an_unpaired_task_drops_out_of_both_sides():
    rows = [_row("a", "full"), _row("a", "summary"), _row("c", "full")]

    xs, ys = paired(rows, "accuracy_score", "full", "summary")

    assert len(xs) == len(ys) == 1


# ── the all-ties case ───────────────────────────────────────────────────────

def _three_conditions(task, scores, category="navigation"):
    return [
        _row(task, c, acc=s, comp=s, category=category)
        for c, s in zip(["full", "summary", "hybrid"], scores)
    ]


def test_every_pair_tied_reports_no_p_value_instead_of_a_fake_one():
    # Navigation tasks scoring 5 across the board is the case that produced
    # this: scipy raises on an all-zero difference vector, and a p of 1.0
    # invented here would read as a test that ran and found nothing.
    rows = []
    for i in range(8):
        rows += _three_conditions(f"t{i}", [5, 5, 5])

    result = next(
        r for r in run_tests(rows)
        if r["dimension"] == "accuracy" and r["a"] == "full" and r["b"] == "summary"
    )

    assert result["n_pairs"] == 8
    assert result["n_nonzero"] == 0
    assert result["p"] != result["p"]  # NaN
    assert result["delta"] == 0.0


def test_ties_shrink_the_effective_n_and_both_numbers_are_kept():
    rows = []
    for i in range(10):
        rows += _three_conditions(f"t{i}", [5, 5, 5])
    rows += _three_conditions("t10", [5, 3, 5])
    rows += _three_conditions("t11", [5, 2, 5])

    result = next(
        r for r in run_tests(rows)
        if r["dimension"] == "accuracy" and r["a"] == "full" and r["b"] == "summary"
    )

    assert result["n_pairs"] == 12
    assert result["n_nonzero"] == 2


def test_six_tests_are_corrected_as_one_family():
    rows = []
    for i in range(12):
        rows += _three_conditions(f"t{i}", [5, 3, 4])

    results = run_tests(rows)
    testable = [r for r in results if r["n_nonzero"] > 0]

    assert len(results) == 6
    for r in testable:
        assert r["p_holm"] >= r["p"]


# ── descriptive statistics ──────────────────────────────────────────────────

def test_stdev_is_the_sample_not_the_population_one():
    # n-1 in the denominator: sqrt(10/4) for 1..5.
    assert stdev([1, 2, 3, 4, 5]) == pytest.approx(1.5811388, rel=1e-6)


def test_a_single_observation_has_no_spread_to_report():
    assert stdev([4]) != stdev([4])  # NaN, not 0


def test_mean_of_nothing_is_not_zero():
    assert mean([]) != mean([])  # NaN
