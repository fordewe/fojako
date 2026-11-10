"""Build the self-contained HTML scoring page for the clean-run validation sample.

Raters open one file, penilaian.html, in a browser: no Python and no server.
The page carries the sample under opaque ids, each module's source exactly as
the judge received it, and the judge's rubric. It keeps scores in the browser
and exports the same CSV columns validate_ui.py wrote, so the analysis joins
them back through the key unchanged.

    uv run python evaluation/build_rater_page.py

Draws 4 responses per condition x category cell from the clean rerun with
seed 42, the seed the first sample used, and writes:

  * dist/penilaian-baru/: penilaian.html, BACA-SAYA.md, and modul/<project>/
    with the module files for browsing in an IDE;
  * dist/penilaian-baru.zip;
  * dist/penilaian-baru-kunci.json and dist/penilaian-baru-sampel.json, which
    map the ids back to tasks and conditions. Neither is shared or committed
    until every rater has finished (dist/ is gitignored).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_rater_package import export_module, opaque_items, write_zip  # noqa: E402
from sample_validation import draw  # noqa: E402
from validate_ui import RUBRIC  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "rater_page" / "penilaian.html"
README_PATH = HERE / "rater_page" / "BACA-SAYA.md"
PACKAGE_NAME = "penilaian-baru"
ID_PREFIX = "Y"  # the first package used X01-X36
ID_SEED = 20261110
SAMPLE_SEED = 42
PER_CELL = 4
SHIPPED_FIELDS = ("task_id", "category", "project", "question", "response")
DATA_MARK = "/*DATA*/"


def page_data(shipped: list[dict], sources: dict[str, str], package_id: str) -> dict:
    """What the page embeds: whitelisted item fields, sources, and the rubric.

    A whitelist, so a field added to the items later cannot reach the page by
    being forgotten here.
    """
    return {
        "package": package_id,
        "items": [{f: item[f] for f in SHIPPED_FIELDS} for item in shipped],
        "sources": {p: sources[p] for p in sorted({i["project"] for i in shipped})},
        "rubric": [[title, question, [list(o) for o in options]]
                   for title, question, options in RUBRIC],
    }


def build_page(template: str, data: dict) -> str:
    """The template with the data embedded as JSON inside a script tag.

    ``<`` is escaped so no answer or source text can close the tag, and the two
    line separators JavaScript treats as line ends are escaped as well.
    """
    if template.count(DATA_MARK) != 1:
        raise ValueError(f"template harus memuat {DATA_MARK} tepat sekali")
    payload = (json.dumps(data, ensure_ascii=False)
               .replace("<", "\\u003c")
               .replace(" ", "\\u2028")
               .replace(" ", "\\u2029"))
    return template.replace(DATA_MARK, payload)


def main():
    parser = argparse.ArgumentParser(description="Bangun halaman penilaian HTML")
    parser.add_argument(
        "--responses", default="evaluation/results/rq2_clean_responses.json")
    parser.add_argument("--datasets", default="evaluation/datasets.json")
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--id-seed", type=int, default=ID_SEED)
    args = parser.parse_args()

    for path in (args.responses, args.datasets):
        if not Path(path).exists():
            raise SystemExit(f"Tidak ditemukan: {path}")
    responses = [r for r in json.loads(Path(args.responses).read_text())
                 if not r.get("error")]
    sample, short = draw(responses, PER_CELL, args.seed)
    if short:
        raise SystemExit(f"Sel kurang dari {PER_CELL} respons: {short}")

    out = Path(args.output_dir)
    package = out / PACKAGE_NAME
    if package.exists():
        shutil.rmtree(package)
    (package / "sumber").mkdir(parents=True)

    shipped, key = opaque_items(sample, args.id_seed, prefix=ID_PREFIX)
    key_path = out / f"{PACKAGE_NAME}-kunci.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    sample_path = out / f"{PACKAGE_NAME}-sampel.json"
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    datasets = json.loads(Path(args.datasets).read_text())
    paths = {p["name"]: Path(p["path"]).expanduser() for p in datasets["projects"]}
    sources = {}
    for project in sorted({r["project"] for r in sample}):
        folder, n = export_module(project, paths[project], package)
        sources[project] = (package / "sumber" / f"{project}.txt").read_text(
            encoding="utf-8")
        print(f"{project}: {n} berkas -> {folder}")
    shutil.rmtree(package / "sumber")  # the page carries the same text

    page = build_page(TEMPLATE_PATH.read_text(encoding="utf-8"),
                      page_data(shipped, sources, PACKAGE_NAME))
    (package / "penilaian.html").write_text(page, encoding="utf-8")
    shutil.copy2(README_PATH, package / "BACA-SAYA.md")

    zip_path = out / f"{PACKAGE_NAME}.zip"
    write_zip(package, zip_path)
    print(f"{len(shipped)} respons -> {package}\nzip: {zip_path} "
          f"({zip_path.stat().st_size // 1024} KB)\n"
          f"kunci dan sampel (jangan dibagikan): {key_path}, {sample_path}")


if __name__ == "__main__":
    main()
