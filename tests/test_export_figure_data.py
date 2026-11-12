"""Tests for the CSVs behind the paper's RQ2 and RQ3 figures.

The figure script in the paper repository reconciles these CSVs with the
manuscript text, so their layout (row order, column names, decimal places) is
a contract. Before this exporter existed the RQ2 and RQ3 tables were typed by
hand from the manuscript.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from analyze_equivalence import PRIMARY_MARGIN, equivalence  # noqa: E402
from analyze_rq2 import run_tests  # noqa: E402
from export_figure_data import (  # noqa: E402
    AGGREGATE_FIELDS,
    JUDGE_COMPARISON_FIELDS,
    aggregate_rows,
    figure_aggregates,
    judge_comparison_rows,
    per_task_rows,
    total_rows,
    write_csv,
)

CATEGORIES = ["navigation", "understanding", "modification"]
CONDITIONS = ["full", "summary", "hybrid"]


def score(task_id, condition, category, accuracy, completeness):
    return {
        "task_id": task_id,
        "condition": condition,
        "category": category,
        "accuracy_score": accuracy,
        "completeness_score": completeness,
    }


def every_cell(accuracy=(5, 4), completeness=(5, 3)):
    """Two scores per category x condition, so every cell has an SD."""
    rows = []
    for category in CATEGORIES:
        for condition in CONDITIONS:
            for i, (a, c) in enumerate(zip(accuracy, completeness)):
                rows.append(score(f"{category}_{i}", condition, category, a, c))
    return rows


# ── RQ2 aggregates ──────────────────────────────────────────────────────────

def test_aggregate_rows_follow_the_figure_order():
    rows = aggregate_rows("judge", every_cell(), with_sd=True)

    assert len(rows) == 18
    assert [(r["category"], r["metric"], r["condition"]) for r in rows[:4]] == [
        ("Navigation", "Accuracy", "Full"),
        ("Navigation", "Accuracy", "Summary"),
        ("Navigation", "Accuracy", "Hybrid"),
        ("Navigation", "Completeness", "Full"),
    ]
    assert rows[-1]["category"] == "Modification"
    assert set(rows[0]) == set(AGGREGATE_FIELDS)


def test_aggregate_rows_print_two_decimals():
    row = aggregate_rows("judge", every_cell(), with_sd=True)[0]

    assert row == {
        "rater": "judge", "category": "Navigation", "condition": "Full",
        "metric": "Accuracy", "mean": "4.50", "sd": "0.71", "n": 2,
    }


def test_human_rows_leave_sd_blank():
    rows = aggregate_rows("human", every_cell(), with_sd=False)

    assert {r["sd"] for r in rows} == {""}


def test_scores_read_from_csv_as_text_are_accepted():
    rows = [
        {**r, "accuracy_score": str(r["accuracy_score"]),
         "completeness_score": str(r["completeness_score"])}
        for r in every_cell()
    ]

    assert aggregate_rows("human", rows, with_sd=False)[0]["mean"] == "4.50"


def test_an_empty_cell_is_an_error_not_a_nan():
    rows = [r for r in every_cell() if r["category"] != "modification"]

    with pytest.raises(ValueError, match="modification"):
        aggregate_rows("judge", rows, with_sd=True)


# ── RQ3 totals ──────────────────────────────────────────────────────────────

def test_total_rows_round_the_condition_means():
    consumption = {
        "full": {"input": 30651.4, "output": 1498.7, "total": 32150.1},
        "summary": {"input": 10678.2, "output": 1311.0, "total": 11989.2},
        "hybrid": {"input": 11295.0, "output": 1444.4, "total": 12739.4},
    }

    rows = total_rows(consumption)

    assert rows[0] == {"condition": "Full", "input": 30651, "output": 1499, "total": 32150}
    assert [r["condition"] for r in rows] == ["Full", "Summary", "Hybrid"]


# ── RQ3 per task ────────────────────────────────────────────────────────────

def response(task_id, condition, **extra):
    base = {"task_id": task_id, "condition": condition,
            "project": "p", "category": "navigation"}
    return {**base, **extra}


def three_conditions(task_id, requested=(), resolved=()):
    return [
        response(task_id, "full", context_tokens_full=300),
        response(task_id, "summary", context_tokens_summary=100),
        response(task_id, "hybrid", files_requested=list(requested),
                 files_resolved=list(resolved)),
    ]


def test_per_task_rows_count_requested_and_sent_files():
    responses = three_conditions("t2", ["A.kt", "B.kt"], ["A.kt"]) + three_conditions("t1")

    rows = per_task_rows(responses, {"t1", "t2"}, {"t1": 100, "t2": 180})

    assert [r["task_id"] for r in rows] == ["t1", "t2"]
    assert rows[1] == {
        "task_id": "t2", "project": "p", "category": "navigation",
        "full": 300, "summary": 100, "hybrid": 180,
        "files_requested": 2, "files_sent": 1,
    }


def test_per_task_rows_skip_tasks_outside_the_balanced_set():
    responses = three_conditions("t1") + three_conditions("t9")

    rows = per_task_rows(responses, {"t1"}, {"t1": 100, "t9": 100})

    assert [r["task_id"] for r in rows] == ["t1"]


def test_a_task_missing_a_condition_is_an_error():
    responses = [r for r in three_conditions("t1") if r["condition"] != "hybrid"]

    with pytest.raises(ValueError, match="t1"):
        per_task_rows(responses, {"t1"}, {})


# ── writing ─────────────────────────────────────────────────────────────────

def test_csv_lines_end_in_lf(tmp_path):
    out = tmp_path / "x.csv"

    write_csv(out, ["a", "b"], [{"a": 1, "b": 2}])

    assert out.read_bytes() == b"a,b\n1,2\n"
    assert list(csv.DictReader(open(out))) == [{"a": "1", "b": "2"}]


def test_figure_aggregates_can_leave_out_the_human_rows():
    # The clean rerun's human sample is still being scored; until then the
    # figure shows the judge alone rather than first-run human means.
    judge = every_cell()

    rows = figure_aggregates(judge, human=None)

    assert {r["rater"] for r in rows} == {"judge"}
    assert len(rows) == 18


def test_figure_aggregates_append_human_rows_when_given():
    rows = figure_aggregates(every_cell(), human=every_cell())

    assert [r["rater"] for r in rows].count("human") == 18


# ── comparison of two judges ────────────────────────────────────────────────

def judged(summary_drop, hybrid_drop_every=5, per_category=15):
    """Full scores 5; Summary loses ``summary_drop``; Hybrid loses 1 on every
    ``hybrid_drop_every``-th task, so no comparison is all ties."""
    rows = []
    for category in CATEGORIES:
        for i in range(per_category):
            task = f"{category}_{i}"
            hybrid = 4 if i % hybrid_drop_every == 0 else 5
            rows += [
                score(task, "full", category, 5, 5),
                score(task, "summary", category, 5 - summary_drop, 5 - summary_drop),
                score(task, "hybrid", category, hybrid, hybrid),
            ]
    return rows


def test_judge_comparison_lists_ten_rows_per_judge_in_a_fixed_order():
    rows = judge_comparison_rows({"first": judged(1), "second": judged(1)})

    assert len(rows) == 20
    first = rows[:10]
    assert {r["judge"] for r in first} == {"first"}
    assert [r["panel"] for r in first] == ["wilcoxon"] * 6 + ["tost"] * 4
    assert [(r["metric"], r["comparison"]) for r in first[:6]] == [
        ("Accuracy", "Full vs Summary"), ("Accuracy", "Full vs Hybrid"),
        ("Accuracy", "Summary vs Hybrid"), ("Completeness", "Full vs Summary"),
        ("Completeness", "Full vs Hybrid"), ("Completeness", "Summary vs Hybrid"),
    ]
    assert [(r["metric"], r["comparison"]) for r in first[6:]] == [
        ("Accuracy", "Hybrid vs Full"), ("Completeness", "Hybrid vs Full"),
        ("Accuracy", "Summary vs Full, navigation"),
        ("Completeness", "Summary vs Full, navigation"),
    ]


def test_every_row_carries_exactly_the_declared_fields():
    rows = judge_comparison_rows({"first": judged(1)})

    assert all(set(r) == set(JUDGE_COMPARISON_FIELDS) for r in rows)


def test_wilcoxon_rows_repeat_the_analysis_deltas_and_holm_values():
    scores = judged(1)
    expected = run_tests(scores)

    rows = [r for r in judge_comparison_rows({"first": scores})
            if r["panel"] == "wilcoxon"]

    for row, test in zip(rows, expected):
        assert float(row["estimate"]) == pytest.approx(test["delta"], abs=5e-4)
        assert float(row["p_holm"]) == pytest.approx(test["p_holm"], abs=5e-5)
        assert row["low"] == "" and row["high"] == ""


def test_tost_rows_repeat_the_analysis_differences_and_intervals():
    scores = judged(1)
    expected = equivalence(scores, PRIMARY_MARGIN)

    rows = [r for r in judge_comparison_rows({"first": scores})
            if r["panel"] == "tost"]

    # equivalence() lists hybrid (accuracy, completeness) then summary-navigation.
    for row, test in zip(rows, expected):
        assert float(row["estimate"]) == pytest.approx(test["mean_diff"], abs=5e-4)
        assert float(row["low"]) == pytest.approx(test["ci90"][0], abs=5e-4)
        assert float(row["high"]) == pytest.approx(test["ci90"][1], abs=5e-4)
        assert float(row["p_holm"]) == pytest.approx(test["p_holm"], abs=5e-5)


def test_significant_means_holm_below_alpha_for_wilcoxon_and_equivalent_for_tost():
    # Summary a full point below Full on every task: a clear Wilcoxon result and
    # no equivalence. Hybrid is one point below on a fifth of tasks only.
    rows = judge_comparison_rows({"first": judged(1)})

    full_vs_summary = next(r for r in rows if r["panel"] == "wilcoxon"
                           and r["comparison"] == "Full vs Summary"
                           and r["metric"] == "Accuracy")
    summary_nav = next(r for r in rows if r["panel"] == "tost"
                       and r["comparison"].startswith("Summary"))
    hybrid = next(r for r in rows if r["panel"] == "tost"
                  and r["comparison"] == "Hybrid vs Full")
    assert full_vs_summary["significant"] == "yes"
    assert summary_nav["significant"] == "no"
    assert hybrid["significant"] == "yes"


def test_a_judge_with_no_complete_task_raises_instead_of_writing_blanks():
    partial = [r for r in judged(1) if r["condition"] != "hybrid"]

    with pytest.raises(ValueError, match="first"):
        judge_comparison_rows({"first": partial})
