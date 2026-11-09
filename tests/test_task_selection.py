"""Tests for choosing which tasks a run sends, and which it may skip.

The clean re-run sends only the 45 judged tasks, and a run cut short by the
subscription limit has to pick up where it stopped without treating the
failed tasks as done.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from task_runner import completed_keys, load_task_ids  # noqa: E402


def test_task_ids_read_from_a_plain_list(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["a", "b"]))

    assert load_task_ids(path) == {"a", "b"}


def test_task_ids_read_from_response_rows(tmp_path):
    # rq2_judge_subset.json holds 135 rows, three per task.
    path = tmp_path / "subset.json"
    path.write_text(json.dumps([
        {"task_id": "a", "condition": "full"},
        {"task_id": "a", "condition": "summary"},
        {"task_id": "b", "condition": "full"},
    ]))

    assert load_task_ids(path) == {"a", "b"}


def test_an_empty_task_id_file_is_refused(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="kosong"):
        load_task_ids(path)


def test_rows_that_failed_are_not_counted_as_done():
    rows = [
        {"task_id": "a", "condition": "full", "model": "m", "response": "x"},
        {"task_id": "b", "condition": "full", "model": "m", "error": "limit"},
    ]

    assert completed_keys(rows) == {("a", "full", "m")}
