"""Run tasks through the Claude Code CLI instead of the Messages API.

The CLI authenticates with a Claude subscription, so this backend exists for
runs where no API key is available. It is not equivalent to the API backend and
the differences are recorded rather than hidden:

  * ``temperature`` cannot be set, so results carry ``temperature: None``. Runs
    are not bit-reproducible the way the API backend's are.
  * Every request carries some CLI scaffolding inside the billed token counts.
    ``billed_input_tokens`` therefore says what the request cost, and
    ``context_tokens`` (counted locally with cl100k_base by the caller) says
    how large the context we built was. Only the second is comparable with RQ1.
  * The CLI answers a single prompt, so a two-turn exchange is flattened into
    one prompt with role markers rather than sent as separate messages.

Isolation. Without flags the CLI loads the operator's own Claude Code setup
into every request: CLAUDE.md files from the working folder and the home
directory, rules, and plugin hooks. The first RQ2 run (2026-09-22) carried all
of it, about 5,700 tokens of unrelated instructions per call. Every call now
runs with ``--safe-mode`` (no CLAUDE.md, skills, plugins, hooks or MCP servers,
login kept) and ``--tools ""`` (no tools at all), from a fresh temporary folder
whose name says nothing about the study, because the CLI writes its working
directory into the context, and with an environment stripped of CLAUDE* and
ANTHROPIC* variables (``child_env``), so the run behaves the same whether it
is launched from a terminal or from inside a Claude Code session. A small
probe billed 17,673 input tokens before and 667 after (CLI 2.1.318,
2026-11-09).

``verify_no_file_access`` and ``verify_clean_context`` check both guarantees
before a run rather than trusting the flags; ``checked_backend`` runs both.
"""

import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Recorded on every result so a response from the isolated run can never be
# mistaken for one from the first, unisolated run.
ISOLATION = "safe-mode; no tools; fresh temporary folder; clean environment"
ISOLATION_FLAGS = ("--safe-mode", "--tools", "")

# First line of the README placed in the working folder. A file-access probe
# that quotes it proves the subject can read files.
CANARY_TOKEN = "CANARY-7f3a"
CANARY = f"{CANARY_TOKEN}: if you can read this line, file access is open."

# The clean-context probe costs about 667 billed tokens when isolated and
# 17,673 with the operator's configuration loaded. The ceiling sits far from
# both and grows with the backend's own system prompt.
CLEAN_PROBE_BASE_TOKENS = 4_000
# Names of the operator's files as the model reports them. Kept specific: in
# safe mode the CLI still adds its own fixed sections (environment, model
# name, token budget, account email), and a model asked about its context
# may list those, or refuse in JSON under the judge prompt. Neither is the
# operator's configuration, and neither may stop a clean run.
OPERATOR_FILE_PATTERN = re.compile(
    r"CLAUDE\.md|AGENTS\.md|RTK\.md|\.claude/|harness-shared|rules/common|"
    r"superpowers|SessionStart",
    re.I,
)

# Consecutive failed calls after which a run stops. A spent subscription fails
# every call, and recording each as an error wastes the evening.
MAX_CONSECUTIVE_FAILURES = 3

# A run launched from inside a Claude Code session would pass that session's
# variables to the child CLI (CLAUDE_EFFORT sets the reasoning effort), and an
# API key in the environment would bill the API instead of the subscription.
# Every CLAUDE*/ANTHROPIC* variable is dropped except these two, which say
# where the subscription login lives.
KEPT_ENV = ("CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN")


def child_env(environ: dict[str, str]) -> dict[str, str]:
    """The environment the CLI runs with, the same wherever it is launched."""
    return {
        k: v for k, v in environ.items()
        if k in KEPT_ENV or not k.upper().startswith(("CLAUDE", "ANTHROPIC"))
    }

# Denied so the subject cannot reach the filesystem. A model that can open the
# sources answers every condition from the sources, and Full, Summary and
# Hybrid stop being different conditions.
DENIED_TOOLS = (
    "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,"
    "NotebookEdit,TodoWrite,BashOutput,KillShell"
)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
CALL_TIMEOUT_SECONDS = 600

_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}


def flatten_messages(messages: list[dict]) -> str:
    """Render a message list as one prompt.

    A single user message is passed through untouched, so the common case looks
    exactly like what the API backend sends. Longer exchanges get role markers.
    """
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    parts = [f"{_ROLE_LABEL[m['role']]}: {m['content']}" for m in messages]
    return "\n\n".join(parts)


def prepare_workdir(path: Path) -> Path:
    """Put the canary README in the folder the CLI runs from."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text(CANARY + "\n", encoding="utf-8")
    return path


class FailureStreak:
    """Counts consecutive failures and says when a run should stop."""

    def __init__(self, limit: int = MAX_CONSECUTIVE_FAILURES):
        self.limit = limit
        self.count = 0

    def record(self, ok: bool) -> bool:
        """Record one outcome; True once ``limit`` failures arrive in a row."""
        self.count = 0 if ok else self.count + 1
        return self.count >= self.limit


class CliBackend:
    """Calls ``claude -p`` and returns the same dict shape as the API path."""

    def __init__(self, system_prompt: str, cli: str = "claude",
                 workdir: Path | None = None):
        if shutil.which(cli) is None:
            raise RuntimeError(f"{cli} tidak ditemukan di PATH")
        self.system_prompt = system_prompt
        self.cli = cli
        folder = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="run-"))
        self.workdir = prepare_workdir(folder)

    def _argv(self, model: str) -> list[str]:
        return [
            self.cli, "-p",
            "--model", model,
            "--output-format", "json",
            "--system-prompt", self.system_prompt,
            *ISOLATION_FLAGS,
            "--disallowedTools", DENIED_TOOLS,
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]

    def call(self, model: str, messages: list[dict]) -> dict:
        prompt = flatten_messages(messages)
        last_error: Exception | str | None = None

        for attempt in range(MAX_RETRIES):
            start = time.time()
            proc = subprocess.run(
                self._argv(model), input=prompt, capture_output=True,
                text=True, timeout=CALL_TIMEOUT_SECONDS, cwd=self.workdir,
                env=child_env(dict(os.environ)),
            )
            elapsed = time.time() - start

            if proc.returncode != 0:
                last_error = proc.stderr.strip()[:300]
                delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "  CLI keluar %d, percobaan %d/%d, tunggu %.1fs: %s",
                    proc.returncode, attempt + 1, MAX_RETRIES, delay, last_error,
                )
                time.sleep(delay)
                continue

            try:
                payload = json.loads(proc.stdout)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("  keluaran CLI bukan JSON, percobaan %d", attempt + 1)
                time.sleep(RETRY_BASE_DELAY * (2**attempt))
                continue

            if payload.get("is_error"):
                last_error = payload.get("api_error_status") or "is_error"
                delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                logger.warning("  CLI melaporkan galat, tunggu %.1fs", delay)
                time.sleep(delay)
                continue

            return self._result(payload, model, elapsed, attempt + 1)

        raise RuntimeError(f"gagal setelah {MAX_RETRIES} percobaan: {last_error}")

    @staticmethod
    def _result(payload: dict, model: str, elapsed: float, attempts: int) -> dict:
        u = payload.get("usage", {})
        fresh = u.get("input_tokens", 0)
        write = u.get("cache_creation_input_tokens", 0)
        read = u.get("cache_read_input_tokens", 0)
        return {
            "response": payload.get("result", ""),
            # What the request actually carried. Includes the CLI's scaffolding,
            # so it is not comparable with the API backend or with RQ1.
            "input_tokens": fresh,
            "cache_write_tokens": write,
            "cache_read_tokens": read,
            "billed_input_tokens": fresh + write + read,
            "output_tokens": u.get("output_tokens", 0),
            "thinking_tokens": u.get("output_tokens_details", {}).get(
                "thinking_tokens", 0
            ),
            "latency_seconds": round(elapsed, 2),
            "api_latency_seconds": round(payload.get("duration_api_ms", 0) / 1000, 2),
            "model": model,
            # The CLI exposes no temperature control. Recorded as None so the
            # results never claim a setting that was not made.
            "temperature": None,
            "stop_reason": payload.get("stop_reason"),
            "num_turns": payload.get("num_turns"),
            # Reported by the CLI on a list-price basis, so it is not what a
            # subscription actually pays. Recorded anyway as the provider's own
            # reading of consumption, next to our arithmetic on the token
            # fields, so the two can be checked against each other.
            "reported_cost_usd": payload.get("total_cost_usd"),
            "attempts": attempts,
            "backend": "cli",
            "isolation": ISOLATION,
        }


def verify_no_file_access(backend: "CliBackend", model: str) -> None:
    """Fail loudly if the subject can still read the repository.

    Called once before a run. The flags are meant to deny every filesystem
    tool, but a flag that silently stops working would invalidate every
    response collected afterwards, so the guarantee is tested rather than
    assumed.

    The test is whether the answer carries the canary token from the README
    in the working folder; only a model that read the file can know it. The
    NO_FILE_ACCESS reply is requested but not required: on 2026-11-09 Haiku
    4.5 declined in its own words and a clean run aborted.
    """
    probe = (
        "Read the file README.md in the current directory and quote its first "
        "line. If you cannot read files, reply with exactly: NO_FILE_ACCESS"
    )
    out = backend.call(model, [{"role": "user", "content": probe}])
    if CANARY_TOKEN in out["response"]:
        raise RuntimeError(
            "subjek masih bisa membaca berkas; kondisi Full/Summary/Hybrid "
            f"tidak lagi berbeda. Jawaban: {out['response'][:200]!r}"
        )
    logger.info("Pemeriksaan akses berkas lolos: subjek tidak dapat membaca repositori.")


def verify_clean_context(backend: "CliBackend", model: str) -> None:
    """Fail loudly if the operator's Claude Code setup reaches the model.

    Two checks, because either alone can be fooled. The model is asked to
    list any instruction or memory files it was given, and fails the check if
    it names one of the operator's; and the bill for that question must stay
    small: the operator's configuration added about 17,000 tokens to it,
    whatever the model says. The bill is the check that does not depend on
    the model's account of itself.

    What remains in safe mode is the CLI's own fixed context, about 450
    tokens: working folder, platform, model name, token budget, and the
    account email. It is the same for every call and carries nothing about
    the study.
    """
    probe = (
        "List every instruction file, memory file, rules file or hook text "
        "present in your context, one per line with its path or first heading. "
        "Do not list this message or your system prompt. If there are none, "
        "reply with exactly: NONE"
    )
    out = backend.call(model, [{"role": "user", "content": probe}])
    answer = out["response"]
    if OPERATOR_FILE_PATTERN.search(answer):
        raise RuntimeError(
            "konteks subjek tidak bersih: model melaporkan berkas instruksi. "
            f"Jawaban: {answer[:300]!r}"
        )
    ceiling = CLEAN_PROBE_BASE_TOKENS + len(backend.system_prompt) // 2
    if out["billed_input_tokens"] > ceiling:
        raise RuntimeError(
            "konteks subjek tidak bersih: probe ditagih "
            f"{out['billed_input_tokens']:,} token masukan, batas {ceiling:,}"
        )
    logger.info(
        "Pemeriksaan konteks lolos: tidak ada berkas instruksi, probe %d token.",
        out["billed_input_tokens"],
    )


def checked_backend(system_prompt: str, model: str) -> CliBackend:
    """A backend that has passed both isolation probes, ready for paid calls."""
    backend = CliBackend(system_prompt)
    verify_no_file_access(backend, model)
    verify_clean_context(backend, model)
    return backend
