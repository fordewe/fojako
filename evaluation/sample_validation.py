"""Draw the human validation sample for the LLM judge.

Section 4.5 fixes the design: 108 responses, four from each cell of subject
tier x condition x task category. Every level of every factor therefore carries
36 responses, which is what lets agreement be reported per category and per
tier rather than as one number.

Per tier matters here. Judge and subject share a model family, so
self-preference would favour whichever tier matches the judge. Balancing the
sample across tiers turns that from an argument into a measurement.

The output is a plain subset of the responses file, so ``score_responses.py``
reads it with no changes:

    uv run python evaluation/sample_validation.py \\
        --responses evaluation/results/rq2_responses.json \\
        --output evaluation/results/rq2_validation_sample.json

    uv run python evaluation/score_responses.py \\
        --input evaluation/results/rq2_validation_sample.json \\
        --output evaluation/results/rq2_human_scores.csv
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PER_CELL = 4


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
    args = parser.parse_args()

    with open(args.responses) as f:
        responses = [r for r in json.load(f) if not r.get("error")]

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
