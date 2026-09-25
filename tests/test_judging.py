"""Tests for the LLM judge and the validation sampler.

The judge's prompt and its verdict parsing carry two guarantees the paper makes
in Section 4.5: the judge cannot tell which condition produced an answer, and a
score that did not come from the judge never enters the results.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from sample_validation import draw  # noqa: E402


@pytest.fixture
def judge():
    """Import lazily: judge_responses pulls in task_runner, which needs anthropic."""
    import judge_responses

    return judge_responses


# ── blinding ────────────────────────────────────────────────────────────────

def test_prompt_carries_source_question_and_answer(judge):
    prompt = judge.build_prompt(
        "class PlantRepository(private val dao: PlantDao)",
        "Where is PlantRepository defined?",
        "In the data package.",
    )

    assert "class PlantRepository" in prompt
    assert "Where is PlantRepository defined?" in prompt
    assert "In the data package." in prompt


def test_prompt_is_a_pure_function_of_source_question_and_answer(judge):
    # The strongest form of the blinding guarantee: two responses that differ
    # only in which condition produced them yield byte-identical prompts, so
    # the judge has nothing to key on. A judge that can see "Summary" can mark
    # it down without reading it, and the comparison collapses.
    full_row = {"condition": "full", "model": "claude-opus-5"}
    summary_row = {"condition": "summary", "model": "claude-haiku-4-5"}

    a = judge.build_prompt("source", "question", "same answer")
    b = judge.build_prompt("source", "question", "same answer")

    assert a == b
    # Nothing identifying the run reaches the judge. The rubric says "Fully
    # correct", so a bare search for "full" would fire on the rubric; the model
    # ids are the identifiers that actually matter.
    for row in (full_row, summary_row):
        assert row["model"] not in a
    assert "condition" not in a.lower()


def test_prompt_demands_evidence_for_each_score(judge):
    prompt = judge.build_prompt("source", "question", "answer")

    assert "accuracy_evidence" in prompt
    assert "completeness_evidence" in prompt
    assert "verify it appears in the source" in prompt


# ── verdict parsing ─────────────────────────────────────────────────────────

def test_parses_a_bare_json_verdict(judge):
    verdict = judge.parse_verdict(
        '{"accuracy": 4, "completeness": 3, "accuracy_evidence": "Plant.kt",'
        ' "completeness_evidence": "missed the dao", "notes": ""}'
    )

    assert verdict["accuracy"] == 4
    assert verdict["completeness"] == 3


def test_parses_json_wrapped_in_prose_and_fences(judge):
    reply = (
        "Here is my assessment:\n```json\n"
        '{"accuracy": 5, "completeness": 5}\n```\nHope that helps.'
    )

    verdict = judge.parse_verdict(reply)

    assert verdict == {"accuracy": 5, "completeness": 5}


@pytest.mark.parametrize(
    "reply",
    [
        '{"accuracy": 0, "completeness": 3}',      # below the scale
        '{"accuracy": 6, "completeness": 3}',      # above the scale
        '{"accuracy": "4", "completeness": 3}',    # string, not an integer
        '{"accuracy": 4.5, "completeness": 3}',    # not an integer
        '{"completeness": 3}',                     # missing entirely
    ],
)
def test_rejects_a_score_that_is_not_an_integer_one_to_five(judge, reply):
    # Coercing these would put a number in the results that the judge never
    # gave, and nothing downstream could tell it apart from a real score.
    with pytest.raises(ValueError):
        judge.parse_verdict(reply)


def test_rejects_a_reply_with_no_json_at_all(judge):
    with pytest.raises(ValueError, match="tidak ada objek JSON"):
        judge.parse_verdict("I think the answer is pretty good, maybe a 4.")


# ── token accounting ────────────────────────────────────────────────────────

def _out(**overrides):
    out = {
        "billed_input_tokens": 33750, "cache_write_tokens": 30000,
        "cache_read_tokens": 3000, "output_tokens": 400,
        "thinking_tokens": 150, "reported_cost_usd": 0.42,
        "attempts": 1, "latency_seconds": 12.3,
    }
    out.update(overrides)
    return out


def test_build_row_carries_token_usage_from_the_backend_call(judge):
    resp = {"task_id": "t1", "condition": "full", "model": "claude-haiku-4-5"}
    verdict = {"accuracy": 4, "completeness": 3}

    row = judge.build_row(resp, "sunflower", "claude-sonnet-5", verdict, _out())

    assert row["billed_input_tokens"] == 33750
    assert row["cache_write_tokens"] == 30000
    assert row["cache_read_tokens"] == 3000
    assert row["output_tokens"] == 400
    assert row["thinking_tokens"] == 150
    assert row["reported_cost_usd"] == 0.42
    assert row["attempts"] == 1
    assert row["latency_seconds"] == 12.3


def test_build_row_is_a_subset_of_fieldnames(judge):
    resp = {"task_id": "t1", "condition": "full", "model": "claude-haiku-4-5"}
    verdict = {"accuracy": 4, "completeness": 3}

    row = judge.build_row(resp, "sunflower", "claude-sonnet-5", verdict, _out())

    assert set(row.keys()) == set(judge.FIELDNAMES)


def test_migrate_schema_leaves_a_file_already_on_the_current_header_alone(
    judge, tmp_path
):
    path = tmp_path / "scores.csv"
    path.write_text(",".join(judge.FIELDNAMES) + "\n")

    judge.migrate_schema(path, judge.FIELDNAMES)

    assert path.read_text() == ",".join(judge.FIELDNAMES) + "\n"


def test_migrate_schema_backfills_rows_scored_before_token_columns_existed(
    judge, tmp_path
):
    # rq2_judge_scores.csv has 95 rows written before token capture existed.
    # Widening FIELDNAMES without backfilling them would misalign every column
    # DictWriter appends afterwards.
    old_fields = [
        "task_id", "condition", "model", "category", "project",
        "judge_model", "accuracy_score", "completeness_score",
        "accuracy_evidence", "completeness_evidence", "notes",
    ]
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="") as f:
        writer = __import__("csv").DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({
            "task_id": "t1", "condition": "full", "model": "claude-haiku-4-5",
            "category": "navigation", "project": "sunflower",
            "judge_model": "claude-sonnet-5", "accuracy_score": 4,
            "completeness_score": 3, "accuracy_evidence": "Plant.kt",
            "completeness_evidence": "ok", "notes": "",
        })

    judge.migrate_schema(path, judge.FIELDNAMES)

    rows = list(__import__("csv").DictReader(open(path)))
    assert len(rows) == 1
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["billed_input_tokens"] == ""
    assert set(rows[0].keys()) == set(judge.FIELDNAMES)


# ── the validation sample ───────────────────────────────────────────────────

def _responses(models=("haiku", "sonnet", "opus"),
               conditions=("full", "summary", "hybrid"),
               categories=("navigation", "understanding", "modification"),
               per_cell=8):
    out = []
    for m in models:
        for c in conditions:
            for k in categories:
                for i in range(per_cell):
                    out.append({
                        "task_id": f"{m}_{c}_{k}_{i}",
                        "model": m, "condition": c, "category": k,
                        "question": "q", "response": "a", "project": "p",
                    })
    return out


def test_draws_four_per_cell_for_a_balanced_hundred_and_eight():
    sample, short = draw(_responses(), per_cell=4, seed=42)

    assert len(sample) == 108
    assert short == []


def test_every_level_of_every_factor_carries_thirty_six():
    sample, _ = draw(_responses(), per_cell=4, seed=42)

    for field in ("model", "condition", "category"):
        counts = {}
        for r in sample:
            counts[r[field]] = counts.get(r[field], 0) + 1
        assert set(counts.values()) == {36}, f"{field} tidak seimbang: {counts}"


def test_the_same_seed_draws_the_same_sample():
    first, _ = draw(_responses(), per_cell=4, seed=7)
    second, _ = draw(_responses(), per_cell=4, seed=7)

    assert [r["task_id"] for r in first] == [r["task_id"] for r in second]


def test_draw_does_not_depend_on_the_order_of_the_input_file():
    ordered = _responses()
    shuffled = list(reversed(ordered))

    a, _ = draw(ordered, per_cell=4, seed=7)
    b, _ = draw(shuffled, per_cell=4, seed=7)

    assert sorted(r["task_id"] for r in a) == sorted(r["task_id"] for r in b)


def test_a_short_cell_is_reported_rather_than_topped_up_from_elsewhere():
    # Only one tier has been run so far, which is the state during a staged
    # sweep. The caller must learn the sample is unbalanced before scoring,
    # because a per-tier comparison cannot be made from it.
    responses = _responses(models=("haiku",), per_cell=2)

    sample, short = draw(responses, per_cell=4, seed=42)

    assert len(sample) == 18  # 9 cells x 2 available
    assert len(short) == 9
    assert all(have == 2 for _, have in short)
