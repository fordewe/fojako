"""Tests for the blinded package sent to external raters.

The package must carry nothing that tells a rater which condition or model
produced an answer, and the key must still join every score back.
"""

import sys
from pathlib import Path

import pytest

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

from build_rater_package import opaque_items  # noqa: E402
from rater_ui import FORBIDDEN, make_rater_ui  # noqa: E402

CONDITIONS = ("full", "summary", "hybrid")


def grouped_sample():
    """36 responses grouped by condition, as rq2_validation_sample.json is."""
    return [
        {"task_id": f"t{i:02d}", "condition": c, "model": "haiku",
         "category": "navigation", "project": "p", "question": "q",
         "response": "r", "context_tokens_full": 1}
        for c in CONDITIONS for i in range(12)
    ]


def test_shipped_items_carry_no_condition_or_model():
    shipped, _ = opaque_items(grouped_sample(), seed=1)
    assert all(item["condition"] == "" and item["model"] == "" for item in shipped)
    assert {k for item in shipped for k in item} == {
        "task_id", "condition", "model", "category", "project",
        "question", "response",
    }


def test_ids_are_sequential_and_opaque():
    shipped, _ = opaque_items(grouped_sample(), seed=1)
    assert [item["task_id"] for item in shipped] == [f"X{n:02d}" for n in range(1, 37)]


def test_key_joins_every_id_back_to_its_response():
    sample = grouped_sample()
    shipped, key = opaque_items(sample, seed=1)
    by_id = {k["id"]: (k["task_id"], k["condition"], k["model"]) for k in key}
    originals = {(r["task_id"], r["condition"], r["model"]) for r in sample}
    assert set(by_id.values()) == originals
    assert set(by_id) == {item["task_id"] for item in shipped}


def test_ids_do_not_follow_the_condition_grouping():
    _, key = opaque_items(grouped_sample(), seed=1)
    first_twelve = {k["condition"] for k in key[:12]}
    assert len(first_twelve) > 1


def test_same_seed_gives_same_ids():
    assert opaque_items(grouped_sample(), seed=7) == opaque_items(grouped_sample(), seed=7)


@pytest.fixture(scope="module")
def rater_ui_text():
    return make_rater_ui((EVALUATION / "validate_ui.py").read_text(encoding="utf-8"))


def test_rater_ui_compiles(rater_ui_text):
    compile(rater_ui_text, "validate_ui.py", "exec")


def test_rater_ui_runs_on_python_38(rater_ui_text):
    assert "from __future__ import annotations" in rater_ui_text
    assert "task_runner" not in rater_ui_text


def test_rater_ui_requires_evidence_in_page_and_server(rater_ui_text):
    assert "Isi bukti dulu" in rater_ui_text
    assert '"bukti wajib diisi"' in rater_ui_text


def test_rater_ui_names_output_by_rater(rater_ui_text):
    assert '"--nama"' in rater_ui_text
    assert 'f"hasil_{name}.csv"' in rater_ui_text


def test_rater_ui_hides_study_design_outside_csv_columns(rater_ui_text):
    prose = [
        line for line in rater_ui_text.splitlines()
        if any(w in line.lower() for w in FORBIDDEN)
    ]
    # Only the lines that name the CSV columns may mention condition/model.
    assert all('"condition"' in line or '["condition"]' in line for line in prose)


def test_rater_ui_fails_loudly_when_upstream_changes():
    original = (EVALUATION / "validate_ui.py").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="tidak cocok"):
        make_rater_ui(original.replace("import webbrowser\n", "import webbrowser  \n"))
