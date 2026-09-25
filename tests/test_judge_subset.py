"""Tests for subsetting the judging queue.

Cutting the queue is a cost decision, but it becomes a design decision the
moment the cut correlates with anything. These check that it does not.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from select_judge_subset import select_tasks  # noqa: E402

PROJECTS = ["arch", "compose", "nia", "sunflower", "tivi"]
CATEGORIES = ["modification", "navigation", "understanding"]
PREFIX = {"modification": "mod", "navigation": "nav", "understanding": "und"}


def _tasks(per_cell: int = 5) -> list[dict]:
    out = []
    for project in PROJECTS:
        for category in CATEGORIES:
            for n in range(1, per_cell + 1):
                template = f"{PREFIX[category]}_{n:02d}"
                out.append({
                    "id": f"{project}_{template}", "project": project,
                    "category": category, "template_id": template,
                })
    return out


def test_every_cell_contributes_the_same_number_of_tasks():
    chosen = set(select_tasks(_tasks(), per_cell=3))
    picked = [t for t in _tasks() if t["id"] in chosen]

    cells = Counter((t["project"], t["category"]) for t in picked)

    assert len(chosen) == 45
    assert set(cells.values()) == {3}


def test_no_template_is_always_chosen_and_none_always_dropped():
    # The failure this guards against: taking the first three of each cell,
    # which silently turns "templates 1-3" into the definition of a category.
    chosen = set(select_tasks(_tasks(), per_cell=3))
    picked = [t for t in _tasks() if t["id"] in chosen]

    for category in CATEGORIES:
        counts = Counter(
            t["template_id"] for t in picked if t["category"] == category
        )
        assert len(counts) == 5, f"{category} dropped a template entirely"
        assert max(counts.values()) - min(counts.values()) <= 1


def test_selection_is_deterministic():
    assert select_tasks(_tasks(), 3) == select_tasks(_tasks(), 3)


def test_a_short_cell_gives_everything_it_has_and_borrows_nothing():
    # nowinandroid has three navigation tasks, not five.
    tasks = [t for t in _tasks() if not (
        t["project"] == "nia" and t["category"] == "navigation"
        and t["template_id"] in {"nav_02", "nav_05"}
    )]

    chosen = set(select_tasks(tasks, per_cell=3))
    picked = [t for t in tasks if t["id"] in chosen]
    cell = [t for t in picked if t["project"] == "nia" and t["category"] == "navigation"]

    assert len(cell) == 3
    assert {t["template_id"] for t in cell} == {"nav_01", "nav_03", "nav_04"}


def test_asking_for_more_than_a_cell_holds_never_duplicates():
    chosen = select_tasks(_tasks(per_cell=2), per_cell=3)

    assert len(chosen) == len(set(chosen))
    assert len(chosen) == len(PROJECTS) * len(CATEGORIES) * 2


def test_selection_never_looks_at_a_response_or_a_score():
    # select_tasks takes the task list alone. If it ever grew a parameter for
    # responses, choosing what to grade could depend on what the answer said.
    from inspect import signature

    assert list(signature(select_tasks).parameters) == ["tasks", "per_cell"]
