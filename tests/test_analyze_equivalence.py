"""Tests for the equivalence tests and paired effect sizes of RQ2.

The plan (margins, claims, procedure) was fixed before any of it was run; see
RUN-LOG 42 in the paper repository. These tests pin the procedure so the
numbers it prints cannot drift from what the plan describes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from analyze_equivalence import (  # noqa: E402
    bootstrap_ci,
    category_tests,
    differences,
    rank_biserial,
    tost,
)


def row(task_id, condition, category, accuracy, completeness=None):
    return {
        "task_id": task_id, "condition": condition, "category": category,
        "accuracy_score": accuracy,
        "completeness_score": accuracy if completeness is None else completeness,
    }


# ── differences ─────────────────────────────────────────────────────────────

def test_differences_are_condition_minus_full_per_task():
    rows = [row("t1", "full", "navigation", 5), row("t1", "hybrid", "navigation", 4),
            row("t2", "full", "navigation", 3), row("t2", "hybrid", "navigation", 5)]

    assert differences(rows, "accuracy_score", "hybrid") == [-1, 2]


def test_differences_can_be_limited_to_one_category():
    rows = [row("t1", "full", "navigation", 5), row("t1", "summary", "navigation", 4),
            row("t2", "full", "modification", 5), row("t2", "summary", "modification", 1)]

    assert differences(rows, "accuracy_score", "summary", category="navigation") == [-1]


# ── TOST ────────────────────────────────────────────────────────────────────

def test_identical_scores_are_equivalent_within_half_a_point():
    result = tost([0] * 15, margin=0.5)

    assert result["p"] < 0.05
    assert result["p"] == max(result["p_lower"], result["p_upper"])


def test_a_one_point_gap_is_not_equivalent_within_half_a_point():
    result = tost([-1] * 15, margin=0.5)

    # Every difference sits below -0.5, so the lower one-sided test fails.
    assert result["p_lower"] > 0.5
    assert result["p"] >= 0.05


def test_tost_is_symmetric_in_the_sign_of_the_gap():
    down, up = tost([-1, 0, 0, 1, -1, 0] * 3, 0.5), tost([1, 0, 0, -1, 1, 0] * 3, 0.5)

    assert down["p"] == pytest.approx(up["p"])


# ── bootstrap interval ──────────────────────────────────────────────────────

def test_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    d = [0, 0, 1, -1, 0, 1, 0, 0, -1, 2]

    first, second = bootstrap_ci(d), bootstrap_ci(d)

    assert first == second
    mean, low, high = first
    assert low <= mean <= high


def test_bootstrap_interval_of_constant_differences_is_a_point():
    assert bootstrap_ci([1, 1, 1, 1]) == (1.0, 1.0, 1.0)


# ── matched-pairs rank-biserial ─────────────────────────────────────────────

def test_rank_biserial_is_one_when_every_difference_is_positive():
    assert rank_biserial([1, 2, 1, 3]) == pytest.approx(1.0)


def test_rank_biserial_is_zero_for_mirrored_differences():
    assert rank_biserial([1, -1, 2, -2]) == pytest.approx(0.0)


def test_rank_biserial_follows_the_simple_difference_formula():
    # |d| ranks with zeros dropped: 1 -> 1.5, 1 -> 1.5, 2 -> 3, 3 -> 4.
    # Positive ranks 1.5 + 4 = 5.5, negative 1.5 + 3 = 4.5, total 10.
    assert rank_biserial([1, -1, -2, 3, 0]) == pytest.approx((5.5 - 4.5) / 10)


def test_rank_biserial_of_all_ties_is_undefined():
    assert rank_biserial([0, 0, 0]) is None


# ── per-category tests ──────────────────────────────────────────────────────

def test_category_tests_holm_correct_within_each_category_and_metric():
    rows = []
    for i in range(8):
        rows += [row(f"n{i}", "full", "navigation", 5),
                 row(f"n{i}", "summary", "navigation", 4 if i % 2 else 3),
                 row(f"n{i}", "hybrid", "navigation", 5)]

    results = category_tests(rows)

    nav_acc = [r for r in results
               if r["category"] == "navigation" and r["metric"] == "accuracy"]
    assert len(nav_acc) == 3
    raw = sorted(r["p"] for r in nav_acc if r["p"] == r["p"])
    holm = sorted(r["p_holm"] for r in nav_acc if r["p_holm"] == r["p_holm"])
    assert all(h >= p for h, p in zip(holm, raw))
