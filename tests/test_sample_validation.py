"""Tests for drawing the human validation sample from judged responses only.

The paper's sample was drawn from all 213 responses, so eight of the 36 had
no judge score and had to be judged after the human pass. Restricting the
pool to judged responses keeps that from happening on a new draw.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from sample_validation import draw, judged_keys, restrict  # noqa: E402


def response(task_id, condition, category="navigation", model="m"):
    return {
        "task_id": task_id,
        "condition": condition,
        "model": model,
        "category": category,
    }


def write_judged(path, keys):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "condition", "model"])
        writer.writeheader()
        for task_id, condition, model in keys:
            writer.writerow(
                {"task_id": task_id, "condition": condition, "model": model}
            )
    return path


def test_judged_keys_reads_several_files(tmp_path):
    first = write_judged(tmp_path / "a.csv", [("t1", "full", "m")])
    second = write_judged(tmp_path / "b.csv", [("t2", "summary", "m")])

    assert judged_keys([first, second]) == {
        ("t1", "full", "m"),
        ("t2", "summary", "m"),
    }


def test_restrict_keeps_only_judged_responses():
    responses = [response("t1", "full"), response("t1", "summary")]

    kept = restrict(responses, {("t1", "full", "m")})

    assert kept == [response("t1", "full")]


def test_restrict_matches_on_model_too():
    responses = [response("t1", "full", model="other")]

    assert restrict(responses, {("t1", "full", "m")}) == []


def test_a_restricted_draw_never_picks_an_unjudged_response():
    responses = [response(f"t{i}", "full") for i in range(10)]
    judged = {(f"t{i}", "full", "m") for i in range(0, 10, 2)}

    sample, short = draw(restrict(responses, judged), per_cell=4, seed=42)

    assert len(sample) == 4
    assert short == []
    assert all((r["task_id"], r["condition"], r["model"]) in judged for r in sample)
