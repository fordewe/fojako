"""Score RQ2 responses with a language model, against the source code.

The protocol is the one Section 4.5 of the paper describes:

  * The judge receives the question, one candidate answer, and the complete
    source of the module the question is about. It grades against the codebase,
    not against the context the subject model saw. A Summary-condition answer
    that misses something visible only in a method body loses points for it.
    Grading each condition against what it could see would build the conclusion
    into the measurement.
  * The judge never learns which condition or which subject model produced an
    answer. Neither appears in its prompt.
  * The judge reports the file and declaration behind each score, so a reader
    can audit a judgement instead of trusting it.

Judge and subject come from the same model family. That is a known limitation,
and the answer to it is the human validation sample (``sample_validation.py``),
not an argument.

Usage:
    uv run python evaluation/judge_responses.py \\
        --responses evaluation/results/rq2_responses.json \\
        --datasets evaluation/datasets.json \\
        --output evaluation/results/rq2_judge_scores.csv \\
        --judge-model claude-sonnet-5
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_backend import CliBackend  # noqa: E402
from task_runner import build_context  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You grade answers about a Kotlin/Java codebase. You are given the complete "
    "source of one module, a question about it, and one candidate answer. Grade "
    "the candidate against the source code, not against how confident or "
    "well-written it sounds. Reply with JSON only."
)

RUBRIC = """ACCURACY (1-5) — is what the answer says true of the source above?
  1 = Completely wrong or irrelevant
  2 = Partially relevant but mostly incorrect
  3 = Correct direction but contains notable inaccuracies
  4 = Mostly correct, minor inaccuracies
  5 = Fully correct

COMPLETENESS (1-5) — does it cover what the question asks for?
  1 = Missing almost all relevant information
  2 = Covers less than half of the expected information
  3 = Covers about half, missing notable elements
  4 = Mostly complete, minor gaps
  5 = Fully complete, addresses all aspects"""

FIELDNAMES = [
    "task_id", "condition", "model", "category", "project",
    "judge_model", "accuracy_score", "completeness_score",
    "accuracy_evidence", "completeness_evidence", "notes",
]


def build_prompt(full_source: str, question: str, answer: str) -> str:
    """Assemble the grading prompt.

    Condition and subject model are deliberately absent: the judge must not be
    able to tell a Full answer from a Summary one except by reading it.
    """
    return f"""Here is the complete source of the module:

{full_source}

---

QUESTION: {question}

CANDIDATE ANSWER:
{answer}

---

Grade the candidate answer on two dimensions.

{RUBRIC}

For each dimension, name the evidence you checked in the source above: file
names and declarations. Where the answer names a file, package, class or
method, verify it appears in the source.

Reply with this JSON object and nothing else:
{{"accuracy": <1-5>, "accuracy_evidence": "<files and declarations checked>",
 "completeness": <1-5>, "completeness_evidence": "<what was covered or missing>",
 "notes": "<one sentence, or empty>"}}"""


def parse_verdict(text: str) -> dict:
    """Pull the verdict out of the judge's reply.

    Models wrap JSON in prose or fences often enough that locating the outermost
    braces is worth the few lines. A verdict whose scores are not 1-5 integers is
    rejected rather than coerced, because a silently clamped score would enter
    the results as if the judge had given it.
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"tidak ada objek JSON di jawaban penilai: {text[:200]!r}")
    verdict = json.loads(match.group(0))

    for field in ("accuracy", "completeness"):
        value = verdict.get(field)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{field} bukan bilangan bulat 1-5: {value!r}")
    return verdict


def load_scored(path: Path) -> set[tuple[str, str, str]]:
    """Keys already graded, so a resumed run does not pay for them twice."""
    if not path.exists():
        return set()
    with open(path) as f:
        return {
            (row["task_id"], row["condition"], row.get("model", ""))
            for row in csv.DictReader(f)
        }


def main():
    parser = argparse.ArgumentParser(description="Grade RQ2 responses with a model")
    parser.add_argument("--responses", default="evaluation/results/rq2_responses.json")
    parser.add_argument("--datasets", default="evaluation/datasets.json")
    parser.add_argument("--output", default="evaluation/results/rq2_judge_scores.csv")
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-5",
        help="Fixed for the whole run. Changing it mid-way makes scores incomparable",
    )
    parser.add_argument("--limit", type=int, help="Grade at most N responses")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build every prompt and report its size without calling the judge",
    )
    args = parser.parse_args()

    with open(args.responses) as f:
        responses = [r for r in json.load(f) if not r.get("error")]
    with open(args.datasets) as f:
        project_paths = {
            p["name"]: Path(p["path"]).expanduser()
            for p in json.load(f)["projects"]
        }

    output_path = Path(args.output)
    scored = load_scored(output_path)
    pending = [
        r for r in responses
        if (r["task_id"], r["condition"], r.get("model", "")) not in scored
    ]
    if args.limit:
        pending = pending[: args.limit]

    logger.info(
        "%d respons, %d sudah dinilai, %d akan dinilai",
        len(responses), len(scored), len(pending),
    )
    if not pending:
        return

    backend = None if args.dry_run else CliBackend(JUDGE_SYSTEM_PROMPT)
    contexts: dict[str, dict] = {}
    prompt_tokens: list[int] = []
    written = 0

    for i, resp in enumerate(pending, 1):
        project = resp["project"]
        if project not in contexts:
            logger.info("Membangun konteks rujukan untuk %s...", project)
            contexts[project] = build_context(project_paths[project])
        ctx = contexts[project]
        if not ctx["ok"]:
            logger.warning("  %s dilewati: %s", resp["task_id"], ctx["reason"])
            continue

        prompt = build_prompt(ctx["full"], resp["question"], resp["response"])

        if args.dry_run:
            prompt_tokens.append(len(prompt) // 4)
            continue

        logger.info("[%d/%d] %s", i, len(pending), resp["task_id"])
        try:
            out = backend.call(args.judge_model, [{"role": "user", "content": prompt}])
            verdict = parse_verdict(out["response"])
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            # Recorded as a gap rather than a score. A failed judgement that
            # silently became a 3 would be indistinguishable from a real one.
            logger.error("  gagal menilai %s: %s", resp["task_id"], e)
            continue

        row = {
            "task_id": resp["task_id"],
            "condition": resp["condition"],
            "model": resp.get("model", ""),
            "category": resp.get("category", ""),
            "project": project,
            "judge_model": args.judge_model,
            "accuracy_score": verdict["accuracy"],
            "completeness_score": verdict["completeness"],
            "accuracy_evidence": verdict.get("accuracy_evidence", ""),
            "completeness_evidence": verdict.get("completeness_evidence", ""),
            "notes": verdict.get("notes", ""),
        }
        _append(output_path, row)
        written += 1

    if args.dry_run:
        total = sum(prompt_tokens)
        print(f"\n=== dry run ===\nPrompt dibangun: {len(prompt_tokens)}")
        if prompt_tokens:
            print(f"Perkiraan token per prompt: min {min(prompt_tokens):,}  "
                  f"maks {max(prompt_tokens):,}")
            print(f"Perkiraan total: {total:,} (kasar, 4 karakter per token)")
        return

    print(f"\nDinilai dan disimpan: {written} baris ke {output_path}")


def _append(path: Path, row: dict) -> None:
    """Append one verdict. Written per row so an interrupted run keeps its work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
