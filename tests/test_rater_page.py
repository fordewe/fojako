"""Tests for the self-contained HTML scoring page.

The page replaces the Python server for raters: one HTML file holds the
sample, the module sources and the rubric, keeps scores in the browser and
exports the same CSV the server wrote. It must ship nothing that tells a rater
which condition or model produced an answer, and its CSV must join back
through the key exactly as the server's did.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

from build_rater_package import opaque_items  # noqa: E402
from build_rater_page import (  # noqa: E402
    SHIPPED_FIELDS,
    TEMPLATE_PATH,
    build_page,
    page_data,
)
from judge_responses import RUBRIC as JUDGE_RUBRIC  # noqa: E402
from rater_ui import FORBIDDEN, is_csv_column_line  # noqa: E402
from validate_ui import RUBRIC  # noqa: E402


def shipped_items():
    rows = [
        {"task_id": f"t{i}", "condition": c, "model": "m", "category": "navigation",
         "project": "p", "question": "q", "response": "r"}
        for i, c in enumerate(("full", "summary", "hybrid"))
    ]
    shipped, _ = opaque_items(rows, seed=1, prefix="Y")
    return shipped


def test_opaque_ids_take_the_requested_prefix():
    assert [item["task_id"] for item in shipped_items()] == ["Y01", "Y02", "Y03"]


def test_page_data_ships_only_the_whitelisted_fields():
    data = page_data(shipped_items(), {"p": "src"}, package_id="pkg")

    assert all(set(item) == set(SHIPPED_FIELDS) for item in data["items"])
    assert "condition" not in SHIPPED_FIELDS and "model" not in SHIPPED_FIELDS


def test_page_rubric_is_the_judge_rubric():
    data = page_data(shipped_items(), {"p": "src"}, package_id="pkg")

    assert data["rubric"] == [list(r[:2]) + [[list(o) for o in r[2]]] for r in RUBRIC]
    for _, _, options in RUBRIC:
        for _, text in options:
            assert text in JUDGE_RUBRIC


def test_embedded_data_cannot_close_the_script_tag():
    items = shipped_items()
    items[0]["response"] = "</script><script>alert(1)</script> "
    data = page_data(items, {"p": "src"}, package_id="pkg")

    page = build_page(TEMPLATE_PATH.read_text(encoding="utf-8"), data)

    assert "</script><script>alert" not in page
    block = re.search(
        r'<script type="application/json" id="data">(.*?)</script>', page, re.S
    ).group(1)
    assert json.loads(block)["items"][0]["response"] == items[0]["response"]


def test_template_says_nothing_about_the_study_design():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    leaks = [
        line for line in template.splitlines()
        if any(w in line.lower() for w in FORBIDDEN) and not is_csv_column_line(line)
    ]
    assert leaks == []


# ── the page's own logic, run under Node ────────────────────────────────────

def logic_script() -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return re.search(r'<script id="logic">(.*?)</script>', template, re.S).group(1)


def run_node(test_code: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node tidak tersedia")
    proc = subprocess.run(
        [node, "-e", logic_script() + "\n" + test_code],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_rater_names_are_normalised_like_the_server_did():
    out = run_node('console.log(normalizeName("  Budi Santoso! "))')
    assert out == "budi_santoso"


def test_order_is_a_fixed_permutation_per_rater():
    out = run_node("""
      const ids = Array.from({length: 36}, (_, i) => "Y" + String(i + 1).padStart(2, "0"));
      const a = shuffled(ids, "budi"), b = shuffled(ids, "budi"), c = shuffled(ids, "sari");
      console.log(JSON.stringify([
        a.join() === b.join(), a.join() !== c.join(),
        [...a].sort().join() === ids.join(), a.join() !== ids.join()]));
    """)
    assert json.loads(out) == [True, True, True, True]


def test_csv_round_trips_commas_quotes_newlines_and_unicode():
    out = run_node(r"""
      const rows = [{task_id: "Y01", category: "navigation", project: "tivi",
        accuracy_score: 4, completeness_score: 5,
        evidence: 'A.kt, fun "x"\nB.kt — cek', notes: ""}];
      const back = parseCsv(toCsv(rows));
      console.log(JSON.stringify(back));
    """)
    back = json.loads(out)
    assert back == [{
        "task_id": "Y01", "condition": "", "model": "", "category": "navigation",
        "project": "tivi", "accuracy_score": "4", "completeness_score": "5",
        "evidence": 'A.kt, fun "x"\nB.kt — cek', "notes": "",
    }]


def test_csv_header_matches_the_server_output():
    out = run_node('console.log(toCsv([]).trim())')
    assert out == ("task_id,condition,model,category,project,"
                   "accuracy_score,completeness_score,evidence,notes")
