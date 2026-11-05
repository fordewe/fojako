"""Build the folder sent to external raters for the RQ2 validation sample.

External raters score the same 36 responses the first author scored, blind,
against the module the subject model saw. They get one folder (and a zip of
it) holding:

  * validate_ui.py, the author's scoring page made safe to share
    (``rater_ui.make_rater_ui``);
  * sampel.json, the 36 responses under opaque ids X01-X36, with condition and
    model removed;
  * sumber/<project>.txt, each module's text exactly as the judge received it;
  * modul/<project>/<module>/, the same files as a folder tree for an IDE;
  * BACA-SAYA.md, the instructions.

The key from opaque id to task, condition and model is written beside the
folder, never inside it, and must not be shared or committed until every rater
has finished. The build checks that the module files reproduce the judge's
text byte for byte, both at the source and in the copied folder.

    uv run python evaluation/build_rater_package.py

Writes dist/validasi-penilai/, dist/validasi-penilai.zip and
dist/validasi-penilai-kunci.json (dist/ is gitignored).
"""

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kotlin_mcp.summarizer import _collect_files  # noqa: E402
from modules import find_modules, module_own_files  # noqa: E402
from rater_ui import make_rater_ui  # noqa: E402
from task_runner import build_context, load_full_source  # noqa: E402

HERE = Path(__file__).resolve().parent
PACKAGE_NAME = "validasi-penilai"
ID_SEED = 20261105  # fixes which opaque id each response gets
SHIPPED_FIELDS = ("category", "project", "question", "response")


def opaque_items(sample: list[dict], seed: int = ID_SEED) -> tuple[list[dict], list[dict]]:
    """Shipped items with no condition or model, and the key that maps back.

    Ids are assigned after a shuffle. The sample file is grouped by condition,
    so ids in file order would tell a rater which answers share a condition.
    """
    order = list(sample)
    random.Random(seed).shuffle(order)
    shipped, key = [], []
    for n, r in enumerate(order, 1):
        item_id = f"X{n:02d}"
        shipped.append({"task_id": item_id, "condition": "", "model": "",
                        **{f: r[f] for f in SHIPPED_FIELDS}})
        key.append({"id": item_id, "task_id": r["task_id"],
                    "condition": r["condition"], "model": r.get("model", ""),
                    "category": r["category"], "project": r["project"]})
    return shipped, key


def export_module(project: str, project_path: Path, package: Path) -> tuple[str, int]:
    """Write the module's text and files; stop if they differ from the judge's."""
    ctx = build_context(project_path)
    if not ctx["ok"]:
        raise SystemExit(f"Modul {project} tidak bisa dibangun: {ctx}")
    module_path = Path(ctx["module_path"])
    files = module_own_files(module_path, find_modules(project_path), _collect_files)
    if load_full_source(module_path, files) != ctx["full"]:
        raise SystemExit(f"{project}: daftar berkas tidak cocok dengan konteks juri")

    (package / "sumber" / f"{project}.txt").write_text(ctx["full"], encoding="utf-8")
    dest_root = package / "modul" / project / ctx["module"]
    copied = []
    for f in files:
        dest = dest_root / f.relative_to(module_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied.append(dest)
    if load_full_source(dest_root, copied) != ctx["full"]:
        raise SystemExit(f"{project}: folder modul tidak sama dengan teks kolom kanan")
    return f"modul/{project}/{ctx['module']}", len(files)


def write_zip(package: Path, zip_path: Path) -> None:
    """Zip the folder, leaving out caches and any rater output left by a test."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(package.rglob("*")):
            if (f.is_file() and "__pycache__" not in f.parts
                    and not f.name.startswith("hasil_")):
                z.write(f, f.relative_to(package.parent))


def main():
    parser = argparse.ArgumentParser(description="Bangun paket untuk penilai eksternal")
    parser.add_argument(
        "--sample", default="evaluation/results/rq2_validation_sample.json")
    parser.add_argument("--datasets", default="evaluation/datasets.json")
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--seed", type=int, default=ID_SEED)
    args = parser.parse_args()

    for path in (args.sample, args.datasets):
        if not Path(path).exists():
            raise SystemExit(f"Tidak ditemukan: {path}")
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    datasets = json.loads(Path(args.datasets).read_text(encoding="utf-8"))
    paths = {p["name"]: Path(p["path"]).expanduser() for p in datasets["projects"]}

    out = Path(args.output_dir)
    package = out / PACKAGE_NAME
    if package.exists():
        shutil.rmtree(package)
    (package / "sumber").mkdir(parents=True)

    shipped, key = opaque_items(sample, args.seed)
    (package / "sampel.json").write_text(
        json.dumps(shipped, indent=2, ensure_ascii=False), encoding="utf-8")
    key_path = out / f"{PACKAGE_NAME}-kunci.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")

    for project in sorted({r["project"] for r in sample}):
        folder, n = export_module(project, paths[project], package)
        print(f"{project}: {n} berkas -> {folder}")

    ui = make_rater_ui((HERE / "validate_ui.py").read_text(encoding="utf-8"))
    (package / "validate_ui.py").write_text(ui, encoding="utf-8")
    shutil.copy2(HERE / "rater_package" / "BACA-SAYA.md", package / "BACA-SAYA.md")

    zip_path = out / f"{PACKAGE_NAME}.zip"
    write_zip(package, zip_path)
    print(f"{len(shipped)} respons -> {package}\nzip: {zip_path} "
          f"({zip_path.stat().st_size // 1024} KB)\n"
          f"kunci (jangan dibagikan): {key_path}")


if __name__ == "__main__":
    main()
