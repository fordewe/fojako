"""Blind scoring UI for the human validation sample.

Section 4.5 validates the LLM judge against a human reader. That reader has to
grade the same material the judge graded, under the same rubric, without
knowing which condition produced an answer. ``score_responses.py`` does this in
the terminal but never shows the module source, so the human ends up grading
plausibility while the judge grades against code. Two different constructs
produce a kappa that means nothing.

This serves the same sample as a local page: the question, the answer, and the
complete module source side by side, with the rubric on screen. Nothing is sent
anywhere. The server binds to localhost and the pages are built here.

    uv run python evaluation/sample_validation.py
    uv run python evaluation/validate_ui.py

Then open http://127.0.0.1:8765.

Blinding is enforced in one place, ``public_item``, which whitelists the fields
that reach the browser. Condition and subject model stay on the server and are
written to the CSV on save, so the scores still join back to the responses.
One leak survives and is not patched here: six Hybrid answers quote the string
REQUEST_FILES, which no other condition produces. Scrubbing them would edit the
recorded data, so they are left alone and counted instead.
"""

import argparse
import csv
import json
import logging
import random
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_runner import build_context  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIELDNAMES = [
    "task_id", "condition", "model", "category", "project",
    "accuracy_score", "completeness_score", "evidence", "notes",
]

# Word for word the rubric judge_responses.py sends. The human and the judge
# have to be answering the same question or their agreement measures nothing.
RUBRIC = [
    ("Accuracy", "is what the answer says true of the source?", [
        (1, "Completely wrong or irrelevant"),
        (2, "Partially relevant but mostly incorrect"),
        (3, "Correct direction but contains notable inaccuracies"),
        (4, "Mostly correct, minor inaccuracies"),
        (5, "Fully correct"),
    ]),
    ("Completeness", "does it cover what the question asks for?", [
        (1, "Missing almost all relevant information"),
        (2, "Covers less than half of the expected information"),
        (3, "Covers about half, missing notable elements"),
        (4, "Mostly complete, minor gaps"),
        (5, "Fully complete, addresses all aspects"),
    ]),
]

PUBLIC_FIELDS = ("task_id", "category", "project", "question", "response")


def public_item(resp: dict, source: str, index: int) -> dict:
    """Everything the browser is allowed to see, and nothing else.

    A whitelist rather than a blacklist: a field added to the responses file
    later cannot leak by being forgotten here. ``condition`` and ``model`` are
    the two that would destroy the validation if they reached the page.
    """
    item = {k: resp.get(k, "") for k in PUBLIC_FIELDS}
    item["index"] = index
    item["source"] = source
    return item


def load_done(path: Path) -> set[tuple[str, str, str]]:
    """Keys already scored, so a reader can close the tab and come back."""
    if not path.exists():
        return set()
    with open(path) as f:
        return {
            (r["task_id"], r["condition"], r.get("model", ""))
            for r in csv.DictReader(f)
        }


def append_score(path: Path, row: dict) -> None:
    """Write one score immediately.

    Scoring 108 answers takes more than one sitting, and a crash that loses an
    afternoon of judgements would be paid for in the reader's time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_sources(sample: list[dict], datasets: Path) -> dict[str, str]:
    """Load each project's module source once and keep it in memory."""
    with open(datasets) as f:
        paths = {
            p["name"]: Path(p["path"]).expanduser() for p in json.load(f)["projects"]
        }
    sources: dict[str, str] = {}
    for project in sorted({r["project"] for r in sample}):
        logger.info("Memuat sumber modul %s...", project)
        ctx = build_context(paths[project])
        sources[project] = ctx["full"] if ctx["ok"] else ""
    return sources


PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validasi manusia</title>
<style>
  :root {
    --bg: #fbfaf7; --panel: #ffffff; --ink: #1c1a17; --muted: #6b6560;
    --line: #e2ddd5; --accent: #7a5c2e; --ok: #2f6b45;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #15140f; --panel: #1e1c17; --ink: #ece7de; --muted: #9a938a;
      --line: #322e27; --accent: #c9a227; --ok: #7fc39a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.6 ui-serif, Georgia, "Times New Roman", serif;
  }
  header {
    position: sticky; top: 0; z-index: 5; background: var(--panel);
    border-bottom: 1px solid var(--line); padding: 10px 20px;
    display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap;
  }
  header b { font-size: 15px; }
  header span { color: var(--muted); font-size: 13px; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  section { padding: 20px; min-width: 0; }
  section + section { border-left: 1px solid var(--line); }
  @media (max-width: 900px) { section + section { border-left: 0; border-top: 1px solid var(--line); } }
  h2 { font-size: 13px; letter-spacing: .08em; text-transform: uppercase;
       color: var(--muted); margin: 0 0 10px; font-weight: 600; }
  .q { font-size: 17px; margin-bottom: 20px; }
  .answer, pre.src {
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 14px; white-space: pre-wrap; word-wrap: break-word;
    font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  pre.src { max-height: 62vh; overflow: auto; }
  fieldset { border: 1px solid var(--line); border-radius: 6px; margin: 18px 0 0;
             padding: 12px 14px; background: var(--panel); }
  legend { font-size: 13px; font-weight: 600; padding: 0 6px; }
  legend em { color: var(--muted); font-weight: 400; font-style: normal; }
  label.opt { display: block; padding: 3px 0; cursor: pointer; font-size: 14px; }
  label.opt input { margin-right: 8px; }
  textarea, input[type=text] {
    width: 100%; background: var(--bg); color: var(--ink); margin-top: 6px;
    border: 1px solid var(--line); border-radius: 5px; padding: 8px;
    font: 13px/1.5 ui-monospace, Menlo, monospace; resize: vertical;
  }
  .bar { margin-top: 18px; display: flex; gap: 10px; align-items: center; }
  button {
    font: 600 14px/1 ui-serif, Georgia, serif; padding: 10px 18px;
    border: 1px solid var(--accent); border-radius: 5px; cursor: pointer;
    background: var(--accent); color: var(--bg);
  }
  button.ghost { background: transparent; color: var(--ink); border-color: var(--line); }
  button:disabled { opacity: .45; cursor: default; }
  .note { color: var(--muted); font-size: 13px; }
  .done { color: var(--ok); }
  #filter { max-width: 260px; margin-bottom: 8px; }
</style>
</head>
<body>
<header>
  <b>Validasi manusia</b>
  <span id="progress"></span>
  <span id="task"></span>
  <span class="note">akurasi 1-5 lalu kelengkapan 1-5, Enter untuk simpan</span>
</header>
<main>
  <section>
    <h2>Pertanyaan</h2>
    <div class="q" id="question"></div>
    <h2>Jawaban yang dinilai</h2>
    <div class="answer" id="answer"></div>
    <form id="form">
      <div id="rubric"></div>
      <fieldset>
        <legend>Bukti <em>berkas dan deklarasi yang kamu periksa</em></legend>
        <input type="text" id="evidence" placeholder="mis. PlantListViewModel.kt, fun setGrowZoneNumber">
      </fieldset>
      <fieldset>
        <legend>Catatan <em>opsional</em></legend>
        <textarea id="notes" rows="2"></textarea>
      </fieldset>
      <div class="bar">
        <button type="submit" id="save">Simpan dan lanjut</button>
        <button type="button" class="ghost" id="skip">Lewati</button>
        <span class="note" id="status"></span>
      </div>
    </form>
  </section>
  <section>
    <h2>Sumber modul <span id="proj" class="note"></span></h2>
    <input type="text" id="filter" placeholder="saring baris...">
    <pre class="src" id="source"></pre>
  </section>
</main>
<script>
let item = null, fullSource = "";

function esc(s) { return s; }

function renderRubric(rubric) {
  document.getElementById('rubric').innerHTML = rubric.map(function (r) {
    var key = r[0].toLowerCase();
    return '<fieldset><legend>' + r[0] + ' <em>' + r[1] + '</em></legend>' +
      r[2].map(function (o) {
        return '<label class="opt"><input type="radio" name="' + key +
               '" value="' + o[0] + '">' + o[0] + ' &mdash; ' + o[1] + '</label>';
      }).join('') + '</fieldset>';
  }).join('');
}

async function load() {
  const res = await fetch('/api/next');
  const data = await res.json();
  document.getElementById('progress').textContent =
    data.done + ' / ' + data.total + ' dinilai';
  if (!data.item) {
    document.getElementById('question').textContent = 'Selesai. Semua sudah dinilai.';
    document.getElementById('answer').textContent = '';
    document.getElementById('form').style.display = 'none';
    document.getElementById('source').textContent = '';
    return;
  }
  item = data.item;
  fullSource = item.source;
  document.getElementById('task').textContent = item.category;
  document.getElementById('question').textContent = item.question;
  document.getElementById('answer').textContent = item.response || '(kosong)';
  document.getElementById('proj').textContent = item.project;
  document.getElementById('source').textContent = fullSource;
  document.getElementById('filter').value = '';
  document.getElementById('evidence').value = '';
  document.getElementById('notes').value = '';
  document.querySelectorAll('input[type=radio]').forEach(function (r) { r.checked = false; });
  document.getElementById('status').textContent = '';
  window.scrollTo(0, 0);
}

document.getElementById('filter').addEventListener('input', function (e) {
  const q = e.target.value.toLowerCase();
  document.getElementById('source').textContent = q
    ? fullSource.split('\\n').filter(function (l) { return l.toLowerCase().includes(q); }).join('\\n')
    : fullSource;
});

document.getElementById('form').addEventListener('submit', async function (e) {
  e.preventDefault();
  const acc = document.querySelector('input[name=accuracy]:checked');
  const comp = document.querySelector('input[name=completeness]:checked');
  if (!acc || !comp) {
    document.getElementById('status').textContent = 'Isi kedua skor dulu.';
    return;
  }
  await fetch('/api/score', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      index: item.index,
      accuracy: Number(acc.value),
      completeness: Number(comp.value),
      evidence: document.getElementById('evidence').value,
      notes: document.getElementById('notes').value,
    }),
  });
  load();
});

document.getElementById('skip').addEventListener('click', function () {
  fetch('/api/skip', {method: 'POST'}).then(load);
});

document.addEventListener('keydown', function (e) {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key >= '1' && e.key <= '5') {
    const acc = document.querySelector('input[name=accuracy]:checked');
    const name = acc ? 'completeness' : 'accuracy';
    const box = document.querySelector('input[name=' + name + '][value="' + e.key + '"]');
    if (box) box.checked = true;
  }
});

renderRubric(RUBRIC_JSON);
load();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    sample: list[dict] = []
    sources: dict[str, str] = {}
    output: Path = Path()
    cursor: int = 0

    def log_message(self, *args):  # quieter than the default access log
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _pending(self) -> list[int]:
        done = load_done(self.output)
        return [
            i for i, r in enumerate(self.sample)
            if (r["task_id"], r["condition"], r.get("model", "")) not in done
        ]

    def do_GET(self):
        if self.path == "/":
            page = PAGE.replace("RUBRIC_JSON", json.dumps(RUBRIC))
            self._send(page.encode(), "text/html; charset=utf-8")
            return
        if self.path == "/api/next":
            pending = self._pending()
            total = len(self.sample)
            item = None
            if pending:
                idx = pending[Handler.cursor % len(pending)]
                resp = self.sample[idx]
                item = public_item(resp, self.sources.get(resp["project"], ""), idx)
            payload = {"item": item, "done": total - len(pending), "total": total}
            self._send(json.dumps(payload).encode(), "application/json")
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/skip":
            Handler.cursor += 1
            self._send(b"{}", "application/json")
            return
        if self.path != "/api/score":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        resp = self.sample[body["index"]]

        for field in ("accuracy", "completeness"):
            value = body.get(field)
            if not isinstance(value, int) or not 1 <= value <= 5:
                self.send_error(400, f"{field} bukan bilangan bulat 1-5")
                return

        append_score(self.output, {
            "task_id": resp["task_id"],
            "condition": resp["condition"],
            "model": resp.get("model", ""),
            "category": resp.get("category", ""),
            "project": resp["project"],
            "accuracy_score": body["accuracy"],
            "completeness_score": body["completeness"],
            "evidence": body.get("evidence", ""),
            "notes": body.get("notes", ""),
        })
        Handler.cursor = 0
        logger.info("Tersimpan: %s", resp["task_id"])
        self._send(b"{}", "application/json")


def main():
    parser = argparse.ArgumentParser(description="Blind human scoring UI")
    parser.add_argument(
        "--sample", default="evaluation/results/rq2_validation_sample.json"
    )
    parser.add_argument("--datasets", default="evaluation/datasets.json")
    parser.add_argument(
        "--output", default="evaluation/results/rq2_human_scores.csv"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    with open(args.sample) as f:
        sample = json.load(f)

    # Shuffled so the three conditions of one task do not arrive together. Seen
    # back to back, a reader scores the second and third against the first
    # instead of against the source.
    random.Random(args.seed).shuffle(sample)

    Handler.sample = sample
    Handler.sources = build_sources(sample, Path(args.datasets))
    Handler.output = Path(args.output)

    done = len(load_done(Handler.output))
    url = f"http://127.0.0.1:{args.port}"
    logger.info("%d respons, %d sudah dinilai", len(sample), done)
    logger.info("Buka %s  (Ctrl-C untuk berhenti, kemajuan tersimpan)", url)

    if not args.no_open:
        webbrowser.open(url)
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        logger.info("Berhenti. %d skor di %s", len(load_done(Handler.output)), args.output)


if __name__ == "__main__":
    main()
