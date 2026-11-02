"""Tests for the RQ3 token-consumption tables.

Hybrid's context has no single recorded number: task_runner.py stores
context_tokens_summary (the fixed first-turn cost) but the second turn sends
whatever files the model asked for, which varies per task and was never
tokenized as a total. These tests check the arithmetic that rebuilds it, by
hand, before it goes into a paper table.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from analyze_rq3 import (  # noqa: E402
    hybrid_context_tokens,
    table_budget,
    table_token_consumption,
    token_consumption,
)


# ── hybrid_context_tokens ────────────────────────────────────────────────────

def test_a_hybrid_task_that_requested_nothing_costs_exactly_the_summary():
    # Hybrid can only add to Summary's context, never subtract from it: a task
    # that never asked for a file must land at the same number Summary did.
    tokens = hybrid_context_tokens(
        context_tokens_summary=7697, resolved_relpaths=[], file_contents={},
    )

    assert tokens == 7697


def test_a_requested_file_adds_its_own_tokenized_size():
    content = "class Scrollbar { fun draw() {} }"
    tokens = hybrid_context_tokens(
        context_tokens_summary=100,
        resolved_relpaths=["Scrollbar.kt"],
        file_contents={"Scrollbar.kt": content},
    )

    from token_counter import count_tokens_tiktoken

    assert tokens == 100 + count_tokens_tiktoken(content)


def test_two_requested_files_add_both():
    contents = {"A.kt": "class A", "B.kt": "class B { fun f() = 1 }"}
    tokens = hybrid_context_tokens(
        context_tokens_summary=50,
        resolved_relpaths=["A.kt", "B.kt"],
        file_contents=contents,
    )

    from token_counter import count_tokens_tiktoken

    expected = 50 + count_tokens_tiktoken(contents["A.kt"]) + count_tokens_tiktoken(contents["B.kt"])
    assert tokens == expected


# ── token_consumption ────────────────────────────────────────────────────────

def _responses():
    return [
        {"task_id": "t1", "condition": "full", "context_tokens_full": 1000,
         "context_tokens_summary": 300, "output_tokens": 100},
        {"task_id": "t1", "condition": "summary", "context_tokens_full": 1000,
         "context_tokens_summary": 300, "output_tokens": 80},
        {"task_id": "t1", "condition": "hybrid", "context_tokens_full": 1000,
         "context_tokens_summary": 300, "output_tokens": 150},
        {"task_id": "t2", "condition": "full", "context_tokens_full": 2000,
         "context_tokens_summary": 500, "output_tokens": 120},
        {"task_id": "t2", "condition": "summary", "context_tokens_full": 2000,
         "context_tokens_summary": 500, "output_tokens": 90},
        {"task_id": "t2", "condition": "hybrid", "context_tokens_full": 2000,
         "context_tokens_summary": 500, "output_tokens": 200},
    ]


def test_full_condition_uses_context_tokens_full():
    consumption = token_consumption(
        _responses(), {"t1", "t2"}, hybrid_context={"t1": 0, "t2": 0}
    )

    assert consumption["full"]["input"] == pytest.approx((1000 + 2000) / 2)
    assert consumption["full"]["output"] == pytest.approx((100 + 120) / 2)
    assert consumption["full"]["total"] == pytest.approx(1500 + 110)


def test_summary_condition_uses_context_tokens_summary():
    consumption = token_consumption(
        _responses(), {"t1", "t2"}, hybrid_context={"t1": 0, "t2": 0}
    )

    assert consumption["summary"]["input"] == pytest.approx((300 + 500) / 2)
    assert consumption["summary"]["output"] == pytest.approx((80 + 90) / 2)


def test_hybrid_condition_uses_the_precomputed_hybrid_context_not_context_tokens_full():
    # If this read context_tokens_full instead of the injected hybrid_context,
    # Hybrid would silently report Full's input size.
    hybrid_context = {"t1": 350, "t2": 600}

    consumption = token_consumption(_responses(), {"t1", "t2"}, hybrid_context)

    assert consumption["hybrid"]["input"] == pytest.approx((350 + 600) / 2)
    assert consumption["hybrid"]["output"] == pytest.approx((150 + 200) / 2)


def test_a_task_outside_the_balanced_set_is_excluded():
    responses = _responses() + [
        {"task_id": "t3", "condition": "full", "context_tokens_full": 9_999_999,
         "context_tokens_summary": 9_999_999, "output_tokens": 9_999_999},
    ]

    consumption = token_consumption(
        responses, {"t1", "t2"}, hybrid_context={"t1": 0, "t2": 0}
    )

    assert consumption["full"]["input"] == pytest.approx(1500)


# ── table formatting ─────────────────────────────────────────────────────────

def _consumption():
    return {
        "full": {"input": 1500.0, "output": 110.0, "total": 1610.0},
        "summary": {"input": 400.0, "output": 85.0, "total": 485.0},
        "hybrid": {"input": 475.0, "output": 175.0, "total": 650.0},
    }


def test_table_token_consumption_reports_full_as_the_baseline():
    table = table_token_consumption(_consumption())

    assert "Full" in table
    assert "1,610" in table
    assert "---" in table  # Full has no "vs Full" ratio


def test_table_token_consumption_expresses_others_relative_to_full():
    table = table_token_consumption(_consumption())
    lines = table.splitlines()
    summary_line = next(l for l in lines if "Summary" in l)

    # 485 / 1610 = 30.1%
    assert "30.1" in summary_line


def test_table_budget_floors_tasks_per_budget():
    table = table_budget(_consumption())

    # 100_000 // 1610 = 62
    assert "62" in table
    # 100_000 // 485 = 206
    assert "206" in table
