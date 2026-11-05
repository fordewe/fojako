"""Turn validate_ui.py into the copy shipped to external raters.

The author's scoring page stays as it is. Raters get a copy that:

  * says nothing about the study design: the original docstring and comments
    name the conditions, the judge, and the REQUEST_FILES string that marks
    Hybrid answers, and a rater who opens the script would read them;
  * runs on plain Python 3.8+ with no fojako imports, reading each module's
    text from the package's sumber/ folder instead of rebuilding it;
  * reads and writes UTF-8, so evidence typed on Windows survives;
  * refuses a score with no evidence, in the page and on the server;
  * takes --nama, which names the output file and fixes that rater's order.

Every edit must match exactly once. If validate_ui.py changes, building the
package fails instead of shipping a half-edited page.
"""

import re

NEW_DOC = '''"""Halaman penilaian lokal / Local scoring page.

    python3 validate_ui.py --nama <namamu>      (macOS / Linux)
    py validate_ui.py --nama <namamu>           (Windows)

Lalu buka / then open http://127.0.0.1:8765. Setiap skor langsung disimpan ke
hasil_<nama>.csv di folder ini; berhenti dengan Ctrl-C dan lanjutkan kapan saja
dengan perintah yang sama. Tidak ada data yang dikirim ke mana pun.
"""

from __future__ import annotations
'''

EDITS = [
    (re.compile(r'\A""".*?"""\n', re.S), NEW_DOC),
    ("import random\nimport sys\nimport webbrowser\n",
     "import random\nimport re\nimport sys\nimport webbrowser\nimport zlib\n"),
    ("sys.path.insert(0, str(Path(__file__).resolve().parent))\n\n"
     "from task_runner import build_context  # noqa: E402\n",
     "HERE = Path(__file__).resolve().parent\n"),
    ("# Word for word the rubric judge_responses.py sends. The human and the judge\n"
     "# have to be answering the same question or their agreement measures nothing.\n",
     "# The scoring rubric.\n"),
    ('    """Everything the browser is allowed to see, and nothing else.\n\n'
     "    A whitelist rather than a blacklist: a field added to the responses file\n"
     "    later cannot leak by being forgotten here. ``condition`` and ``model`` are\n"
     '    the two that would destroy the validation if they reached the page.\n    """\n',
     '    """The fields the page shows."""\n'),
    ('    """Keys already scored, so a reader can close the tab and come back."""\n'
     "    if not path.exists():\n        return set()\n    with open(path) as f:\n",
     '    """Items already scored, so a rater can close the tab and come back."""\n'
     '    if not path.exists():\n        return set()\n'
     '    with open(path, encoding="utf-8") as f:\n'),
    ('    """Write one score immediately.\n\n'
     "    Scoring 108 answers takes more than one sitting, and a crash that loses an\n"
     "    afternoon of judgements would be paid for in the reader's time.\n"
     '    """\n',
     '    """Write one score immediately, so a crash loses nothing."""\n'),
    ('    with open(path, "a", newline="") as f:\n',
     '    with open(path, "a", newline="", encoding="utf-8") as f:\n'),
    (re.compile(r"def build_sources\(.*?\n    return sources\n", re.S),
     'def build_sources(sample: list[dict], folder: Path) -> dict[str, str]:\n'
     '    """Each project\'s module text, as shown in the right-hand column."""\n'
     '    return {\n'
     '        project: (folder / f"{project}.txt").read_text(encoding="utf-8")\n'
     '        for project in sorted({r["project"] for r in sample})\n'
     '    }\n'),
    ("<legend>Bukti <em>berkas dan deklarasi yang kamu periksa</em></legend>",
     "<legend>Bukti <em>wajib: berkas dan deklarasi yang kamu periksa</em></legend>"),
    ("    document.getElementById('status').textContent = 'Isi kedua skor dulu.';\n"
     "    return;\n  }\n",
     "    document.getElementById('status').textContent = 'Isi kedua skor dulu.';\n"
     "    return;\n  }\n"
     "  if (!document.getElementById('evidence').value.trim()) {\n"
     "    document.getElementById('status').textContent =\n"
     "      'Isi bukti dulu: berkas dan deklarasi yang kamu periksa.';\n"
     "    return;\n  }\n"),
    ('                self.send_error(400, f"{field} bukan bilangan bulat 1-5")\n'
     "                return\n",
     '                self.send_error(400, f"{field} bukan bilangan bulat 1-5")\n'
     "                return\n"
     '        if not str(body.get("evidence", "")).strip():\n'
     '            self.send_error(400, "bukti wajib diisi")\n'
     "            return\n"),
    (re.compile(r'    parser\.add_argument\(\n        "--sample".*?'
                r"    Handler\.output = Path\(args\.output\)\n", re.S),
     '    parser.add_argument("--nama", required=True,\n'
     '                        help="nama penilai / rater name")\n'
     '    parser.add_argument("--port", type=int, default=8765)\n'
     '    parser.add_argument("--no-open", action="store_true")\n'
     "    args = parser.parse_args()\n\n"
     '    name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.nama.strip()).strip("_").lower()\n'
     "    if not name:\n"
     '        raise SystemExit("Nama penilai tidak boleh kosong.")\n\n'
     '    with open(HERE / "sampel.json", encoding="utf-8") as f:\n'
     "        sample = json.load(f)\n\n"
     "    # Each rater sees their own order, fixed by their name so a restart keeps it.\n"
     "    random.Random(zlib.crc32(name.encode())).shuffle(sample)\n\n"
     "    Handler.sample = sample\n"
     '    Handler.sources = build_sources(sample, HERE / "sumber")\n'
     '    Handler.output = HERE / f"hasil_{name}.csv"\n'),
    ("len(load_done(Handler.output)), args.output)",
     "len(load_done(Handler.output)), Handler.output)"),
    ('    logger.info("%d respons, %d sudah dinilai", len(sample), done)\n',
     '    logger.info("%d respons, %d sudah dinilai; hasil ke %s",\n'
     "                len(sample), done, Handler.output.name)\n"),
]

# Words that would tell a rater about the study. The CSV keeps its "condition"
# and "model" columns (left empty) so returned scores still join back by key.
FORBIDDEN = ("condition", "judge", "hybrid", "summary", "request_files",
             "subject", "fojako", "kotlin-mcp")


def is_csv_column_line(line: str) -> bool:
    return '"condition"' in line or '["condition"]' in line


def make_rater_ui(source: str) -> str:
    """The rater copy of validate_ui.py; ValueError if any edit misses."""
    text = source
    for old, new in EDITS:
        if isinstance(old, re.Pattern):
            text, count = old.subn(lambda _: new, text)
        else:
            count = text.count(old)
            text = text.replace(old, new)
        if count != 1:
            raise ValueError(
                f"Edit tidak cocok tepat sekali ({count}x): {str(old)[:70]!r}")

    for line in text.splitlines():
        if any(w in line.lower() for w in FORBIDDEN) and not is_csv_column_line(line):
            raise ValueError(f"Kata terlarang tersisa: {line.strip()!r}")
    return text
