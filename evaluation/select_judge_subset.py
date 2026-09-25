"""Choose which tasks the judge grades, when grading all of them costs too much.

Grading every response means sending a module's complete source once per
response. At 71 tasks times three conditions that is 213 calls carrying only
five distinct sources, and on a subscription the bill is the reason this file
exists.

Taking the first three tasks of each cell would be the obvious shortcut and the
wrong one: task ids run ``nav_01`` to ``nav_05`` in template order, so "first
three" means "always templates 1 to 3", and any difficulty those templates
share would land on the results as a property of the category.

So the templates rotate. Project *i* in a cell takes the templates starting at
offset *i*, wrapping around. Across the five projects each template is then
picked about the same number of times, and no template is systematically in or
out. Selection never reads a response or a score.

    uv run python evaluation/select_judge_subset.py --per-cell 3

Writes a responses file the judge reads with no changes:

    uv run python evaluation/judge_responses.py \\
        --responses evaluation/results/rq2_judge_subset.json
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def select_tasks(tasks: list[dict], per_cell: int) -> list[str]:
    """Pick ``per_cell`` task ids from every project-category cell.

    A cell holding fewer tasks than asked for contributes all of them. The
    shortfall is the caller's to report; quietly borrowing from another cell
    would unbalance the design without saying so.
    """
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for task in tasks:
        cells[(task["project"], task["category"])].append(task)

    projects = sorted({t["project"] for t in tasks})
    chosen: list[str] = []

    for (project, category), pool in sorted(cells.items()):
        pool = sorted(pool, key=lambda t: t["template_id"])
        offset = projects.index(project)
        for k in range(min(per_cell, len(pool))):
            chosen.append(pool[(offset + k) % len(pool)]["id"])

    return chosen


def main():
    parser = argparse.ArgumentParser(description="Subset the judging queue")
    parser.add_argument("--tasks", default="evaluation/results/rq2_tasks.json")
    parser.add_argument("--responses", default="evaluation/results/rq2_responses.json")
    parser.add_argument("--output", default="evaluation/results/rq2_judge_subset.json")
    parser.add_argument("--per-cell", type=int, default=3)
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)["tasks"]
    with open(args.responses) as f:
        responses = json.load(f)

    keep = set(select_tasks(tasks, args.per_cell))
    subset = [r for r in responses if r["task_id"] in keep]

    by_template = Counter(
        t["template_id"] for t in tasks if t["id"] in keep
    )
    logger.info(
        "%d tugas dari %d terpilih, %d respons dari %d akan dinilai",
        len(keep), len(tasks), len(subset), len(responses),
    )
    for factor in ("project", "category", "condition"):
        counts = Counter(r[factor] for r in subset)
        logger.info("  per %s: %s", factor, dict(sorted(counts.items())))
    logger.info("  per templat: %s", dict(sorted(by_template.items())))

    short = [
        cell for cell, n in Counter(
            (t["project"], t["category"]) for t in tasks if t["id"] in keep
        ).items() if n < args.per_cell
    ]
    for cell in sorted(short):
        logger.warning("  sel %s tidak penuh", cell)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2))
    print(f"\nDitulis {len(subset)} respons ke {out}")


if __name__ == "__main__":
    main()
