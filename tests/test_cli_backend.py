"""Tests for the subscription backend.

Every test here mocks the subprocess. The point is the contract between the
CLI's JSON and the result dict the rest of the harness consumes, plus the
guarantees the paper relies on: no filesystem access for the subject, no
operator configuration in its context, and no claim of a temperature that was
never set.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from cli_backend import (  # noqa: E402
    CANARY,
    ISOLATION,
    CliBackend,
    FailureStreak,
    checked_backend,
    flatten_messages,
    verify_clean_context,
    verify_no_file_access,
)


def _payload(result="ok", *, fresh=10, write=20_000, read=5_000, out=50,
             thinking=0, turns=1, stop="end_turn", is_error=False):
    return {
        "result": result,
        "stop_reason": stop,
        "num_turns": turns,
        "duration_api_ms": 2500,
        "is_error": is_error,
        "usage": {
            "input_tokens": fresh,
            "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read,
            "output_tokens": out,
            "output_tokens_details": {"thinking_tokens": thinking},
        },
    }


def _completed(payload, returncode=0, stdout=None):
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode,
        stdout=stdout if stdout is not None else json.dumps(payload),
        stderr="boom" if returncode else "",
    )


@pytest.fixture
def no_cli_wait(monkeypatch):
    monkeypatch.setattr("cli_backend.shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr("cli_backend.time.sleep", lambda _: None)


@pytest.fixture
def backend(no_cli_wait, tmp_path):
    return CliBackend(system_prompt="SYS", workdir=tmp_path)


# ── flatten_messages ────────────────────────────────────────────────────────

def test_single_user_message_passes_through_unchanged():
    # Arrange
    messages = [{"role": "user", "content": "Where is Foo defined?"}]
    # Act
    prompt = flatten_messages(messages)
    # Assert: the common case must look exactly like the API request, with no
    # role marker the model could mistake for part of the context.
    assert prompt == "Where is Foo defined?"


def test_two_turn_exchange_keeps_both_sides_and_their_order():
    messages = [
        {"role": "user", "content": "summary + question"},
        {"role": "assistant", "content": "REQUEST_FILES: Foo.kt"},
        {"role": "user", "content": "here is Foo.kt"},
    ]

    prompt = flatten_messages(messages)

    assert prompt.index("User: summary") < prompt.index("Assistant: REQUEST_FILES")
    assert prompt.index("Assistant: REQUEST_FILES") < prompt.index("User: here is")


# ── result mapping ──────────────────────────────────────────────────────────

def test_billed_input_sums_fresh_and_both_cache_fields(backend, monkeypatch):
    # The CLI reports almost the whole request under cache_creation, so summing
    # input_tokens alone understates a half-million-token run as a few hundred.
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _completed(_payload(fresh=9, write=30_092, read=9_443)),
    )

    out = backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    assert out["billed_input_tokens"] == 9 + 30_092 + 9_443
    assert out["input_tokens"] == 9
    assert out["cache_write_tokens"] == 30_092
    assert out["cache_read_tokens"] == 9_443


def test_temperature_is_none_never_zero(backend, monkeypatch):
    # Recording 0 would claim a setting the CLI does not expose, and the paper
    # withdraws its reproducibility claim on exactly this ground.
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(_payload()))

    out = backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    assert out["temperature"] is None


def test_carries_stop_reason_turns_and_thinking(backend, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _completed(
            _payload(stop="max_tokens", turns=2, out=900, thinking=400)
        ),
    )

    out = backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    # stop_reason separates a truncated answer from a complete one; without it
    # a cut-off response scores low on completeness as a measurement artifact.
    assert out["stop_reason"] == "max_tokens"
    assert out["num_turns"] == 2
    assert out["thinking_tokens"] == 400
    assert out["backend"] == "cli"


def test_denies_every_filesystem_tool_on_the_command_line(backend, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["stdin"] = kwargs.get("input")
        return _completed(_payload())

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    denied = seen["argv"][seen["argv"].index("--disallowedTools") + 1]
    for tool in ("Read", "Bash", "Glob", "Grep", "WebFetch", "Task"):
        assert tool in denied
    # The context goes over stdin: a Full context runs past the argv limit.
    assert seen["stdin"] == "q"
    assert "--no-session-persistence" in seen["argv"]


# ── retry behaviour ─────────────────────────────────────────────────────────

def test_retries_a_failed_exit_then_succeeds(backend, monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            return _completed(None, returncode=1, stdout="")
        return _completed(_payload(result="recovered"))

    monkeypatch.setattr(subprocess, "run", flaky)

    out = backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    assert out["response"] == "recovered"
    assert out["attempts"] == 3


def test_retries_when_the_cli_reports_is_error(backend, monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        payload = _payload(is_error=calls["n"] == 1, result="second")
        return _completed(payload)

    monkeypatch.setattr(subprocess, "run", flaky)

    out = backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    assert out["response"] == "second"
    assert calls["n"] == 2


def test_retries_on_unparseable_output_then_gives_up(backend, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _completed(None, stdout="not json")
    )

    with pytest.raises(RuntimeError, match="gagal setelah"):
        backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])


def test_missing_cli_fails_at_construction(monkeypatch):
    monkeypatch.setattr("cli_backend.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="tidak ditemukan"):
        CliBackend(system_prompt="SYS")


# ── the guarantee the conditions rest on ────────────────────────────────────

def test_file_access_probe_passes_when_the_model_cannot_read(backend, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _completed(_payload(result="NO_FILE_ACCESS")),
    )

    verify_no_file_access(backend, "claude-haiku-4-5")  # must not raise


def test_file_access_probe_aborts_the_run_when_the_model_reads_a_file(
    backend, monkeypatch
):
    # If the subject can open the sources it answers every condition from the
    # sources, and Full, Summary and Hybrid stop being different conditions.
    # Failing here is cheaper than discovering it after 639 responses.
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _completed(_payload(result="The first line is '# Sunflower'")),
    )

    with pytest.raises(RuntimeError, match="masih bisa membaca berkas"):
        verify_no_file_access(backend, "claude-haiku-4-5")


# ── isolation from the operator's Claude Code configuration ────────────────
# The first RQ2 run (2026-09-22) loaded the operator's CLAUDE.md files, rules
# and plugin hooks into every subject and judge call. These tests pin the
# flags, the working folder and the probe that keep that from happening again.

def _capture(monkeypatch, payload=None):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["cwd"] = kwargs.get("cwd")
        return _completed(payload or _payload())

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def test_disables_every_customization_and_every_tool(backend, monkeypatch):
    seen = _capture(monkeypatch)

    backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    argv = seen["argv"]
    # --safe-mode drops CLAUDE.md, skills, plugins, hooks and MCP servers but
    # keeps the subscription login, which --bare does not.
    assert "--safe-mode" in argv
    # An empty tool list, not a denylist: a tool added in a later CLI release
    # would slip past a list of names.
    assert argv[argv.index("--tools") + 1] == ""


def test_runs_in_its_own_folder_with_a_canary_readme(backend, monkeypatch):
    seen = _capture(monkeypatch)

    backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    assert seen["cwd"] == backend.workdir
    assert (backend.workdir / "README.md").read_text().startswith(CANARY)


def test_default_folder_is_a_fresh_temporary_one_outside_the_repository(no_cli_wait):
    # The CLI writes its working directory into the context, so the folder
    # name must not mention the project or the study.
    repo = Path(__file__).resolve().parents[1]

    workdir = CliBackend(system_prompt="SYS").workdir

    assert workdir.name.startswith("run-")
    assert repo not in workdir.resolve().parents
    assert [p.name for p in workdir.iterdir()] == ["README.md"]


def test_result_records_the_isolation(backend, monkeypatch):
    _capture(monkeypatch)

    out = backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    assert out["isolation"] == ISOLATION


def test_clean_context_probe_passes_on_none_with_a_small_prompt(backend, monkeypatch):
    # Measured 2026-11-09: 667 billed input tokens for this probe when isolated.
    _capture(monkeypatch, _payload(result="NONE", fresh=10, write=650, read=0))

    verify_clean_context(backend, "claude-haiku-4-5")  # must not raise


def test_clean_context_probe_aborts_when_the_model_names_a_memory_file(
    backend, monkeypatch
):
    _capture(monkeypatch, _payload(
        result="1. /Users/x/.claude/CLAUDE.md\n2. rules/coding-style.md",
        fresh=10, write=650, read=0,
    ))

    with pytest.raises(RuntimeError, match="konteks subjek tidak bersih"):
        verify_clean_context(backend, "claude-haiku-4-5")


def test_clean_context_probe_aborts_on_overhead_even_if_the_model_says_none(
    backend, monkeypatch
):
    # The model's own account can be wrong; the bill cannot. 17,673 tokens is
    # what the same probe cost with the configuration loaded.
    _capture(monkeypatch, _payload(result="NONE", fresh=10, write=17_663, read=0))

    with pytest.raises(RuntimeError, match="token"):
        verify_clean_context(backend, "claude-haiku-4-5")


def _probe_aware_run(memory_listed: bool):
    def fake_run(argv, **kwargs):
        if "README.md" in kwargs["input"]:
            return _completed(_payload(result="NO_FILE_ACCESS", write=600, read=0))
        answer = "/Users/x/.claude/CLAUDE.md" if memory_listed else "NONE"
        return _completed(_payload(result=answer, write=600, read=0))
    return fake_run


def test_checked_backend_returns_a_backend_that_passed_both_probes(
    no_cli_wait, monkeypatch
):
    monkeypatch.setattr(subprocess, "run", _probe_aware_run(memory_listed=False))

    assert isinstance(checked_backend("SYS", "claude-haiku-4-5"), CliBackend)


def test_checked_backend_refuses_a_context_with_operator_files(
    no_cli_wait, monkeypatch
):
    monkeypatch.setattr(subprocess, "run", _probe_aware_run(memory_listed=True))

    with pytest.raises(RuntimeError, match="tidak bersih"):
        checked_backend("SYS", "claude-haiku-4-5")


# ── stopping when the subscription runs out ────────────────────────────────

def test_failure_streak_stops_after_consecutive_failures():
    streak = FailureStreak(limit=3)

    assert [streak.record(ok=False) for _ in range(3)] == [False, False, True]


def test_failure_streak_resets_on_a_success():
    streak = FailureStreak(limit=3)

    streak.record(ok=False)
    streak.record(ok=False)
    streak.record(ok=True)

    assert streak.record(ok=False) is False


def test_child_process_drops_session_variables_and_api_keys(backend, monkeypatch):
    # Launched from inside a Claude Code session, the child would inherit that
    # session's variables (CLAUDE_EFFORT sets the reasoning effort), and an API
    # key would bill the API instead of the subscription.
    for name, value in (("CLAUDECODE", "1"), ("CLAUDE_EFFORT", "xhigh"),
                        ("CLAUDE_CODE_CHILD_SESSION", "1"),
                        ("ANTHROPIC_API_KEY", "sk-test"),
                        ("CLAUDE_CONFIG_DIR", "/cfg")):
        monkeypatch.setenv(name, value)
    seen = {}

    def fake_run(argv, **kwargs):
        seen["env"] = kwargs.get("env")
        return _completed(_payload())

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend.call("claude-haiku-4-5", [{"role": "user", "content": "q"}])

    env = seen["env"]
    for dropped in ("CLAUDECODE", "CLAUDE_EFFORT", "CLAUDE_CODE_CHILD_SESSION",
                    "ANTHROPIC_API_KEY"):
        assert dropped not in env
    # Kept: where the login lives, and the ordinary environment.
    assert env["CLAUDE_CONFIG_DIR"] == "/cfg"
    assert "PATH" in env


@pytest.mark.parametrize("answer", [
    # Sonnet 5 under the judge prompt, observed 2026-11-09. Both describe the
    # CLI's own fixed sections, not the operator's files, and must not stop
    # a clean run.
    "# userEmail\n# Environment\n# You are powered by the model named Sonnet 5",
    '```json\n{"error": "Invalid grading request: no source code module"}\n```',
])
def test_clean_context_probe_ignores_the_cli_built_in_sections(
    backend, monkeypatch, answer
):
    _capture(monkeypatch, _payload(result=answer, fresh=10, write=810, read=0))

    verify_clean_context(backend, "claude-sonnet-5")  # must not raise
