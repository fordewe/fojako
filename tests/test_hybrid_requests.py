"""Tests for the Hybrid condition's file-request round trip.

Hybrid is defined as "summary first, full files on demand". If a request the
model makes does not resolve, the second turn arrives without the file and the
condition quietly degrades into Summary plus a wasted round trip. The first run
resolved only a handful of requests because the parser took one line and split
it on commas, so every case below is one that actually occurred.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from task_runner import _load_requested_files, _parse_requested_files  # noqa: E402


# ── parsing what the model actually writes ──────────────────────────────────

@pytest.mark.parametrize(
    "reply,expected",
    [
        # Seen in the Haiku run, nowinandroid_und_03.
        ("**REQUEST_FILES:** `Scrollbar.kt` (full implementation)", ["Scrollbar.kt"]),
        # nowinandroid_mod_05: a stray bold marker before the name.
        ("REQUEST_FILES: ** `DynamicAsyncImage.kt`", ["DynamicAsyncImage.kt"]),
        # nowinandroid_mod_04 captured "**" and nothing else: the list was on
        # the lines below the marker.
        ("**REQUEST_FILES:**\n- Tag.kt\n- Background.kt", ["Tag.kt", "Background.kt"]),
        # The plain form the prompt asks for.
        ("REQUEST_FILES: A.kt, B.java", ["A.kt", "B.java"]),
        # tivi_und_04: a path rooted outside this module.
        ("REQUEST_FILES: compose/src/commonMain/app/ColorExtractor.kt",
         ["compose/src/commonMain/app/ColorExtractor.kt"]),
    ],
)
def test_parses_the_shapes_models_actually_produce(reply, expected):
    assert _parse_requested_files(reply) == expected


def test_prose_that_names_no_file_requests_nothing():
    # tivi_mod_01 asked for "Full codebase search results for usages of
    # UiMessageManager". There is no file to serve, and inventing one would be
    # worse than recording the miss.
    assert _parse_requested_files(
        "REQUEST_FILES: Full codebase search results for usages of UiMessageManager."
    ) == []


def test_a_reply_without_the_marker_requests_nothing():
    assert _parse_requested_files("I can answer directly from the summary.") == []


def test_the_same_file_named_twice_is_served_once():
    reply = "REQUEST_FILES: `Tag.kt`, Tag.kt (the full one)"

    assert _parse_requested_files(reply) == ["Tag.kt"]


# ── resolving a request against the module ──────────────────────────────────

def _module(tmp_path: Path, *rels: str) -> tuple[Path, list[Path]]:
    files = []
    for rel in rels:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"// {p.name}\n")
        files.append(p)
    return tmp_path, files


def test_resolves_an_exact_relative_path(tmp_path):
    root, files = _module(tmp_path, "src/main/Tag.kt")

    body, res = _load_requested_files(root, files, ["src/main/Tag.kt"])

    assert res["files_resolved"] == ["src/main/Tag.kt"]
    assert res["files_missing"] == []
    assert "// Tag.kt" in body


def test_falls_back_to_the_bare_name_when_the_path_is_rooted_elsewhere(tmp_path):
    # The model answers with a path from the repository root while the module
    # sits several directories down. Before the fallback this missed.
    root, files = _module(tmp_path, "src/commonMain/app/ColorExtractor.kt")

    _, res = _load_requested_files(
        root, files, ["compose/src/commonMain/app/ColorExtractor.kt"]
    )

    assert res["files_resolved"] == ["src/commonMain/app/ColorExtractor.kt"]


def test_a_duplicated_name_sends_every_candidate_and_records_the_ambiguity(tmp_path):
    # Android modules are full of same-named files in different packages.
    root, files = _module(tmp_path, "a/Screen.kt", "b/Screen.kt")

    body, res = _load_requested_files(root, files, ["Screen.kt"])

    assert res["files_ambiguous"] == ["Screen.kt"]
    assert sorted(res["files_resolved"]) == ["a/Screen.kt", "b/Screen.kt"]
    assert body.count("===") == 4  # two files, opening and closing marker each


def test_a_file_outside_the_module_is_recorded_as_missing(tmp_path):
    root, files = _module(tmp_path, "src/main/Tag.kt")

    body, res = _load_requested_files(root, files, ["Nowhere.kt"])

    assert res["files_missing"] == ["Nowhere.kt"]
    assert res["files_resolved"] == []
    assert "file not found in this module" in body
