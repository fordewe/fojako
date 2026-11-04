"""Draw the human validation sample for the LLM judge.

Section 4.5 fixes the design: 108 responses, four from each cell of subject
tier x condition x task category. Every level of every factor therefore carries
36 responses, which is what lets agreement be reported per category and per
tier rather than as one number.

Per tier matters here. Judge and subject share a model family, so
self-preference would favour whichever tier matches the judge. Balancing the
sample across tiers turns that from an argument into a measurement.

The output is a plain subset of the responses file, so ``validate_ui.py``
reads it with no changes:

    uv run python evaluation/sample_validation.py \\
        --responses evaluation/results/rq2_responses.json \\
        --output evaluation/results/rq2_validation_sample.json

    uv run python evaluation/validate_ui.py \\
        --sample evaluation/results/rq2_validation_sample.json \\
        --output evaluation/results/rq2_human_scores.csv

``score_responses.py`` is the older terminal scorer. Its rows carry no
``category``, so agreement per category cannot be computed from them and the
web UI is the correct path for this design.

Pass ``--judged`` with the judge score files for any new draw, so the sample
only holds responses that already have a judge score:

    uv run python evaluation/sample_validation.py \\
        --judged evaluation/results/rq2_judge_scores.csv \\
        --output evaluation/results/rq2_validation_sample_new.json

The committed ``rq2_validation_sample.json`` was drawn without it, from all
213 responses with seed 42, and running the script with no ``--judged``
reproduces it exactly. Eight of its 36 responses fell outside the judged
subset and were judged afterwards into ``rq2_judge_validation_extra.csv``.
The default stays unfiltered so that sample can still be redrawn.
"""

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PER_CELL = 4
KEY = ("task_id", "condition", "model")


def response_key(r: dict) -> tuple:
    return tuple(r.get(k, "") for k in KEY)


def judged_keys(paths: list[Path]) -> set[tuple]:
    """``(task_id, condition, model)`` of every response in the judge files."""
    keys: set[tuple] = set()
    for path in paths:
        with open(path) as f:
            keys.update(response_key(r) for r in csv.DictReader(f))
    return keys


def restrict(responses: list[dict], keys: set[tuple]) -> list[dict]:
    """The responses that have a judge score, in their original order."""
    return [r for r in responses if response_key(r) in keys]


def cells(responses: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in responses:
        grouped[(r.get("model", ""), r["condition"], r.get("category", ""))].append(r)
    return grouped


def draw(responses: list[dict], per_cell: int, seed: int) -> tuple[list[dict], list[tuple]]:
    """Take ``per_cell`` responses from each cell, with a fixed seed.

    Returns the sample and the cells that could not fill their quota. A short
    cell is reported rather than quietly topped up from elsewhere, because a
    sample that is no longer balanced cannot support a per-tier comparison and
    the caller needs to know before scoring starts.
    """
    rng = random.Random(seed)
    sample: list[dict] = []
    short: list[tuple] = []

    for cell in sorted(cells(responses)):
        pool = cells(responses)[cell]
        if len(pool) < per_cell:
            short.append((cell, len(pool)))
        # Sorted first so the draw depends on the seed, not on file order.
        pool = sorted(pool, key=lambda r: r["task_id"])
        sample.extend(rng.sample(pool, min(per_cell, len(pool))))

    return sample, short


def main():
    parser = argparse.ArgumentParser(description="Draw the validation sample")
    parser.add_argument("--responses", default="evaluation/results/rq2_responses.json")
    parser.add_argument(
        "--output", default="evaluation/results/rq2_validation_sample.json"
    )
    parser.add_argument("--per-cell", type=int, default=PER_CELL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--judged", nargs="+", default=None,
        help="judge score files; draw only from responses they score",
    )
    args = parser.parse_args()

    with open(args.responses) as f:
        responses = [r for r in json.load(f) if not r.get("error")]

    if args.judged:
        judged_paths = [Path(p) for p in args.judged]
        missing = [str(p) for p in judged_paths if not p.exists()]
        if missing:
            raise SystemExit(f"Berkas juri tidak ditemukan: {', '.join(missing)}")
        before = len(responses)
        responses = restrict(responses, judged_keys(judged_paths))
        logger.info(
            "%d dari %d respons sudah dinilai juri; sampel diambil dari situ",
            len(responses), before,
        )

    sample, short = draw(responses, args.per_cell, args.seed)

    grouped = cells(responses)
    logger.info(
        "%d respons dalam %d sel, %d diambil",
        len(responses), len(grouped), len(sample),
    )
    for factor, index in (("tier", 0), ("kondisi", 1), ("kategori", 2)):
        counts: dict[str, int] = defaultdict(int)
        for r in sample:
            counts[(r.get("model", ""), r["condition"], r.get("category", ""))[index]] += 1
        logger.info("  per %s: %s", factor, dict(sorted(counts.items())))

    if short:
        logger.warning(
            "%d sel tidak penuh; sampel tidak seimbang dan perbandingan per "
            "tier kehilangan dasarnya:", len(short)
        )
        for cell, have in short:
            logger.warning("  %s: %d dari %d", cell, have, args.per_cell)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=2))
    print(f"\nDitulis {len(sample)} respons ke {out}")


if __name__ == "__main__":
    main()
