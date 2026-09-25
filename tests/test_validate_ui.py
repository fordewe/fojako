"""Tests for the blind human-scoring UI.

The UI exists to produce a kappa against the LLM judge. Two things have to hold
or that number is worthless: the reader must not be able to tell which
condition produced an answer, and a score written down must survive to the CSV
with the condition attached so it can be joined back.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from validate_ui import (  # noqa: E402
    FIELDNAMES,
    PUBLIC_FIELDS,
    RUBRIC,
    append_score,
    load_done,
    public_item,
)


def _response(**over) -> dict:
    base = {
        "task_id": "sunflower_und_03",
        "condition": "summary",
        "model": "claude-haiku-4-5",
        "category": "understanding",
        "project": "sunflower",
        "question": "Where is the grow-zone filter applied?",
        "response": "In PlantListViewModel, via setGrowZoneNumber.",
        "billed_input_tokens": 41233,
        "files_resolved": [],
    }
    base.update(over)
    return base


# ── blinding ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("condition", ["full", "summary", "hybrid"])
def test_the_condition_never_reaches_the_browser(condition):
    item = public_item(_response(condition=condition), "source", 0)

    assert condition not in json.dumps(item)


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"])
def test_the_subject_model_never_reaches_the_browser(model):
    item = public_item(_response(model=model), "source", 0)

    assert model not in json.dumps(item)


def test_a_field_added_to_the_responses_file_later_does_not_leak():
    # The whitelist is the point: a blacklist would have to be updated every
    # time task_runner records something new, and forgetting once is silent.
    item = public_item(_response(secret_condition_marker="hybrid"), "source", 0)

    assert "secret_condition_marker" not in item
    assert set(item) == set(PUBLIC_FIELDS) | {"index", "source"}


def test_the_reader_still_gets_what_they_need_to_grade():
    item = public_item(_response(), "class PlantListViewModel", 7)

    assert item["question"] == "Where is the grow-zone filter applied?"
    assert item["response"].startswith("In PlantListViewModel")
    assert item["source"] == "class PlantListViewModel"
    assert item["index"] == 7


# ── the rubric matches the judge's ──────────────────────────────────────────

def test_the_rubric_is_word_for_word_the_judge_rubric():
    # Agreement between a human and the judge only means something if both
    # answered the same question.
    judge = (Path(__file__).resolve().parents[1] / "evaluation"
             / "judge_responses.py").read_text()

    for _, _, levels in RUBRIC:
        for score, text in levels:
            assert f"{score} = {text}" in judge


def test_both_dimensions_offer_five_levels():
    assert [name for name, _, _ in RUBRIC] == ["Accuracy", "Completeness"]
    for _, _, levels in RUBRIC:
        assert [s for s, _ in levels] == [1, 2, 3, 4, 5]


# ── saving ──────────────────────────────────────────────────────────────────

def _row(**over) -> dict:
    base = {
        "task_id": "sunflower_und_03", "condition": "summary",
        "model": "claude-haiku-4-5", "category": "understanding",
        "project": "sunflower", "accuracy_score": 4,
        "completeness_score": 3, "evidence": "PlantListViewModel.kt", "notes": "",
    }
    base.update(over)
    return base


def test_a_score_carries_the_condition_into_the_csv(tmp_path):
    out = tmp_path / "human.csv"

    append_score(out, _row())

    rows = list(csv.DictReader(open(out)))
    assert rows[0]["condition"] == "summary"
    assert rows[0]["accuracy_score"] == "4"
    assert list(rows[0]) == FIELDNAMES


def test_each_score_is_on_disk_before_the_next_one_is_asked_for(tmp_path):
    out = tmp_path / "human.csv"

    append_score(out, _row(task_id="a"))
    first = len(list(csv.DictReader(open(out))))
    append_score(out, _row(task_id="b"))

    assert first == 1
    assert len(list(csv.DictReader(open(out)))) == 2


def test_a_resumed_session_skips_what_was_already_scored(tmp_path):
    out = tmp_path / "human.csv"
    append_score(out, _row(task_id="a", condition="full"))

    done = load_done(out)

    assert ("a", "full", "claude-haiku-4-5") in done
    assert ("a", "summary", "claude-haiku-4-5") not in done


def test_the_same_task_under_two_conditions_is_two_separate_judgements(tmp_path):
    out = tmp_path / "human.csv"
    append_score(out, _row(condition="full"))
    append_score(out, _row(condition="summary"))

    assert len(load_done(out)) == 2


def test_no_csv_yet_means_nothing_is_done(tmp_path):
    assert load_done(tmp_path / "absent.csv") == set()
