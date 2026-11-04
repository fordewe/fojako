"""Tests for the judge-vs-human agreement script.

The paper reports quadratic-weighted kappa as evidence that the LLM judge can
stand in for a human reader. A weighting bug would change that number without
changing the data, so the weighting is checked against scikit-learn rather than
against a second hand-written implementation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from analyze_kappa import (  # noqa: E402
    agreement,
    by_category,
    load_judge_files,
    verdict,
    weighted_kappa,
)


def row(task_id, condition, model, category, accuracy, completeness):
    return {
        "task_id": task_id,
        "condition": condition,
        "model": model,
        "category": category,
        "accuracy_score": str(accuracy),
        "completeness_score": str(completeness),
    }


# ── weighting ───────────────────────────────────────────────────────────────

def test_perfect_agreement_is_one():
    ratings = [1, 2, 3, 4, 5]
    assert weighted_kappa(ratings, ratings) == pytest.approx(1.0)


def test_constant_offset_scores_zero():
    assert weighted_kappa([1] * 10, [2] * 10) == pytest.approx(0.0)


def test_quadratic_weights_match_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rater_a = [5, 4, 3, 2, 1, 5, 3, 4, 2, 1, 5, 4]
    rater_b = [5, 4, 3, 1, 2, 4, 3, 4, 1, 1, 5, 3]

    expected = sklearn_metrics.cohen_kappa_score(
        rater_a, rater_b, weights="quadratic", labels=[1, 2, 3, 4, 5]
    )
    assert weighted_kappa(rater_a, rater_b) == pytest.approx(expected)


def test_disagreement_pulls_below_one():
    kappa = weighted_kappa([5, 5, 5, 5, 3], [5, 5, 5, 5, 1])
    assert kappa is not None
    assert kappa < 1.0


def test_undefined_when_one_rating_in_common():
    assert weighted_kappa([3] * 8, [3] * 8) is None


def test_empty_returns_none():
    assert weighted_kappa([], []) is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        weighted_kappa([1, 2], [1])


# ── joining ─────────────────────────────────────────────────────────────────

def test_join_drops_rows_missing_from_the_other_rater():
    judge = [
        row("t1", "full", "m", "navigation", 5, 5),
        row("t2", "full", "m", "navigation", 5, 5),
        row("t3", "full", "m", "navigation", 4, 4),
    ]
    human = [
        row("t1", "full", "m", "navigation", 5, 5),
        row("t2", "full", "m", "navigation", 3, 3),
    ]

    result = agreement(judge, human)

    assert result["n_matched"] == 2
    assert result["n_judge_only"] == 1
    assert result["n_human_only"] == 0
    assert result["accuracy"]["kappa"] is not None


def test_single_matched_pair_is_undefined_not_perfect():
    judge = [row("t1", "full", "m", "navigation", 5, 5)]
    human = [row("t1", "full", "m", "navigation", 5, 5)]

    result = agreement(judge, human)

    assert result["accuracy"]["kappa"] is None


def test_join_keys_on_condition_and_model():
    judge = [row("t1", "full", "m", "navigation", 5, 5)]
    human = [row("t1", "summary", "m", "navigation", 5, 5)]

    result = agreement(judge, human)

    assert result["n_matched"] == 0
    assert result["accuracy"]["kappa"] is None


def test_matched_pairs_keep_their_own_n():
    judge = [
        row("t1", "full", "m", "navigation", 5, 5),
        row("t2", "full", "m", "navigation", 1, 1),
    ]
    human = [
        row("t1", "full", "m", "navigation", 5, 5),
        row("t2", "full", "m", "navigation", 3, 3),
    ]

    result = agreement(judge, human)

    assert result["accuracy"]["n"] == 2
    assert result["n_matched"] == 2


# ── grouping ────────────────────────────────────────────────────────────────

def test_categories_are_scored_separately():
    judge = [
        row("t1", "full", "m", "navigation", 5, 5),
        row("t2", "full", "m", "understanding", 5, 5),
    ]
    human = [
        row("t1", "full", "m", "navigation", 5, 5),
        row("t2", "full", "m", "understanding", 1, 1),
    ]

    result = by_category(judge, human)

    assert sorted(result) == ["navigation", "understanding"]
    assert result["navigation"]["n_matched"] == 1
    assert result["understanding"]["n_matched"] == 1


# ── threshold ───────────────────────────────────────────────────────────────

def test_floor_flags_low_kappa():
    assert verdict(0.60) == "below floor"
    assert verdict(0.61) == "substantial or better"
    assert verdict(None) == "undefined"


# ── several judge files ─────────────────────────────────────────────────────

FIELDS = ["task_id", "condition", "model", "category",
          "accuracy_score", "completeness_score"]


def write_scores(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_judge_files_are_read_as_one_table(tmp_path):
    first = write_scores(tmp_path / "a.csv", [row("t1", "full", "m", "navigation", 5, 5)])
    second = write_scores(tmp_path / "b.csv", [row("t2", "full", "m", "navigation", 4, 4)])

    rows = load_judge_files([first, second])

    assert [r["task_id"] for r in rows] == ["t1", "t2"]


def test_a_response_judged_in_two_files_is_rejected(tmp_path):
    first = write_scores(tmp_path / "a.csv", [row("t1", "full", "m", "navigation", 5, 5)])
    second = write_scores(tmp_path / "b.csv", [row("t1", "full", "m", "navigation", 3, 3)])

    with pytest.raises(ValueError, match="t1"):
        load_judge_files([first, second])
