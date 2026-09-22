"""Tests for the behaviour the paper describes as fact.

Three claims are printed in the manuscript and each is checked here against a
real directory tree rather than a mock:

  * Section 3.3 and Figure 2: directories are pruned during the walk, and
    ``build/`` and ``out/`` are pruned only outside a ``src/`` tree because
    both are plausible package names inside one.
  * Section 4.2: a module is measured as the files it owns, excluding those
    belonging to a nested module.
  * Section 4.3: CR, TRR and ID are computed the way the definitions say.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

from kotlin_mcp.summarizer import _collect_files, _is_test_source_set  # noqa: E402
from modules import module_own_files  # noqa: E402
from token_counter import compute_metrics  # noqa: E402


def write(root: Path, rel: str, body: str = "class A\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def names(paths) -> set[str]:
    return {p.name for p in paths}


# ── Section 3.3: what the collector skips ───────────────────────────────────

@pytest.mark.parametrize(
    "parent,dirname,pruned",
    [
        ("src", "test", True),
        ("src", "androidTest", True),
        ("src", "commonTest", True),
        ("src", "TEST", True),          # matched case-insensitively
        ("src", "main", False),
        ("java", "test", False),        # a package named test, not a source set
        ("com", "latest", False),       # substring, not a source set
    ],
)
def test_test_source_sets_are_matched_at_the_source_set_level(parent, dirname, pruned):
    assert _is_test_source_set(parent, dirname) is pruned


def test_build_and_out_survive_inside_a_src_tree(tmp_path):
    # Both name build output at a repository root and both are plausible
    # package names inside src/, which is why the prune is conditional.
    write(tmp_path, "build/generated/Junk.kt")
    write(tmp_path, "out/Stale.kt")
    write(tmp_path, "app/src/main/java/com/example/build/Builder.kt")
    write(tmp_path, "app/src/main/java/com/example/out/Output.kt")

    collected = names(_collect_files(tmp_path, None))

    assert collected == {"Builder.kt", "Output.kt"}


def test_tooling_directories_are_pruned_and_source_sets_with_them(tmp_path):
    write(tmp_path, ".gradle/cache/A.kt")
    write(tmp_path, ".idea/B.kt")
    write(tmp_path, "node_modules/pkg/C.kt")
    write(tmp_path, "app/src/test/java/FooTest.kt")
    write(tmp_path, "app/src/androidTest/java/BarSpec.kt")
    write(tmp_path, "app/src/main/java/Keep.kt")

    assert names(_collect_files(tmp_path, None)) == {"Keep.kt"}


def test_generated_and_test_named_files_are_dropped_by_name(tmp_path):
    for rel in [
        "app/src/main/java/BuildConfig.kt",
        "app/src/main/java/R.java",
        "app/src/main/java/BR.java",
        "app/src/main/java/ActivityMainBinding.kt",
        "app/src/main/java/PlantTest.kt",
        "app/src/main/java/PlantSpec.java",
        "app/src/main/java/Plant.kt",
    ]:
        write(tmp_path, rel)

    assert names(_collect_files(tmp_path, None)) == {"Plant.kt"}


def test_only_kotlin_and_java_sources_are_collected(tmp_path):
    write(tmp_path, "app/src/main/Keep.kt")
    write(tmp_path, "app/src/main/Keep.java")
    write(tmp_path, "app/src/main/notes.md")
    write(tmp_path, "app/src/main/build.gradle")

    assert names(_collect_files(tmp_path, None)) == {"Keep.kt", "Keep.java"}


def test_the_tools_own_output_directory_is_never_re_read(tmp_path):
    # summarize_module writes .kotlin-summary/ into the target directory. A
    # second run must not fold the first run's output back in.
    write(tmp_path, "app/src/main/Plant.kt")
    write(tmp_path, ".kotlin-summary/Leftover.kt")

    assert names(_collect_files(tmp_path, None)) == {"Plant.kt"}


# ── Section 4.2: what counts as a module ────────────────────────────────────

def test_module_own_files_excludes_files_owned_by_a_nested_module(tmp_path):
    parent = tmp_path / "app"
    nested = parent / "feature"
    write(tmp_path, "app/src/main/Own.kt")
    write(tmp_path, "app/feature/src/main/Nested.kt")

    own = module_own_files(parent, [parent, nested], _collect_files)

    assert names(own) == {"Own.kt"}


def test_module_own_files_keeps_everything_when_nothing_is_nested(tmp_path):
    parent = tmp_path / "app"
    write(tmp_path, "app/src/main/One.kt")
    write(tmp_path, "app/src/main/deep/Two.kt")

    own = module_own_files(parent, [parent], _collect_files)

    assert names(own) == {"One.kt", "Two.kt"}


# ── Section 4.3: the compression definitions ────────────────────────────────

def test_trr_is_one_minus_cr_and_id_counts_declarations_per_summary_token():
    original = "fun a() {\n    println(1)\n    println(2)\n}\n" * 20
    summary = "fun a()\n"

    m = compute_metrics(original, summary, structural_elements=1)

    assert m["compression_ratio"] == pytest.approx(
        m["summary_tokens"] / m["original_tokens"], abs=1e-4
    )
    assert m["token_reduction_rate"] == pytest.approx(
        1 - m["compression_ratio"], abs=1e-4
    )
    assert m["information_density"] == pytest.approx(
        1 / m["summary_tokens"], abs=1e-4
    )


def test_a_summary_longer_than_its_source_yields_negative_reduction():
    # One file of the 1,478 measured behaves this way, and Finding 1 reports
    # it. The metric must not clamp the value away.
    m = compute_metrics("class A\n", "File: A.kt\nPackage: x.y\n\nClasses:\n  class A\n", 1)

    assert m["token_reduction_rate"] < 0


def test_empty_source_does_not_divide_by_zero():
    m = compute_metrics("", "", 0)

    assert m["compression_ratio"] == 0.0
    assert m["information_density"] == 0.0
