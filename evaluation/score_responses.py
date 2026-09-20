"""Score RQ2 task responses using a structured rubric.

Reads responses from rq2_responses.json, presents them one at a time
for blind evaluation, and saves scores to rq2_scores.csv.

Usage:
    uv run python evaluation/score_responses.py \
        --input evaluation/results/rq2_responses.json \
        --output evaluation/results/rq2_scores.csv \
        [--reeval]  # for temporal re-evaluation of 20% subset
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path


RUBRIC = """
Scoring Rubric (1-5 Likert scale):

ACCURACY:
  1 = Completely wrong or irrelevant
  2 = Partially relevant but mostly incorrect
  3 = Correct direction but contains notable inaccuracies
  4 = Mostly correct, minor inaccuracies
  5 = Fully correct

COMPLETENESS:
  1 = Missing almost all relevant information
  2 = Covers less than half of expected information
  3 = Covers about half, missing notable elements
  4 = Mostly complete, minor gaps
  5 = Fully complete, addresses all aspects
"""


def load_existing_scores(path: Path) -> set[tuple[str, str, str]]:
    """Load already-scored (task_id, condition, model) triples.

    The model is part of the key because the same task under the same condition
    is a separate observation on each model in the capability sweep.
    """
    scored = set()
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                scored.add((row["task_id"], row["condition"], row.get("model", "")))
    return scored


def main():
    parser = argparse.ArgumentParser(description="Score RQ2 responses")
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation/results/rq2_responses.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/rq2_scores.csv",
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Re-evaluate 20%% subset for intra-rater agreement",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        responses = json.load(f)

    if args.reeval:
        random.seed(args.seed)
        n = max(1, len(responses) // 5)
        responses = random.sample(responses, n)
        args.output = args.output.replace("rq2_scores", "rq2_reeval_scores")
        print(f"Re-evaluation mode: scoring {n} responses")

    # Shuffle for blind evaluation (evaluator doesn't see condition order)
    random.seed(args.seed + 1)
    random.shuffle(responses)

    # Filter out already-scored items
    output_path = Path(args.output)
    scored = load_existing_scores(output_path)
    remaining = [
        r
        for r in responses
        if (r["task_id"], r["condition"], r.get("model", "")) not in scored
    ]

    if not remaining:
        print("All responses already scored!")
        return

    print(RUBRIC)
    print(f"\nResponses to score: {len(remaining)}")
    print("Type 'q' to quit and save progress.\n")

    new_scores = []

    for i, resp in enumerate(remaining):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(remaining)}]")
        print(f"Question: {resp['question']}")
        print(f"Category: {resp['category']}")
        # Note: condition is NOT shown (blind evaluation)
        print(f"\n--- Response ---")
        print(resp.get("response", "(no response)"))
        print(f"--- End Response ---\n")

        try:
            acc = input("Accuracy score (1-5, q=quit): ").strip()
            if acc.lower() == "q":
                break
            acc = int(acc)
            if acc < 1 or acc > 5:
                print("Invalid score, skipping")
                continue

            comp = input("Completeness score (1-5, q=quit): ").strip()
            if comp.lower() == "q":
                break
            comp = int(comp)
            if comp < 1 or comp > 5:
                print("Invalid score, skipping")
                continue

            notes = input("Notes (optional, Enter to skip): ").strip()

            new_scores.append(
                {
                    "task_id": resp["task_id"],
                    "condition": resp["condition"],
                    "model": resp.get("model", ""),
                    "accuracy_score": acc,
                    "completeness_score": comp,
                    "notes": notes,
                }
            )
        except (ValueError, EOFError):
            break

    # Save scores
    if new_scores:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_path.exists()

        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task_id",
                    "condition",
                    "model",
                    "accuracy_score",
                    "completeness_score",
                    "notes",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_scores)

        print(f"\nSaved {len(new_scores)} scores to {output_path}")
    else:
        print("\nNo scores to save.")


if __name__ == "__main__":
    main()
