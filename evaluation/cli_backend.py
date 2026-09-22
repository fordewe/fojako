"""Run tasks through the Claude Code CLI instead of the Messages API.

The CLI authenticates with a Claude subscription, so this backend exists for
runs where no API key is available. It is not equivalent to the API backend and
the differences are recorded rather than hidden:

  * ``temperature`` cannot be set, so results carry ``temperature: None``. Runs
    are not bit-reproducible the way the API backend's are.
  * Every request carries the CLI's own scaffolding — its residual system
    prompt and tool definitions — inside the billed token counts. Measured at
    roughly 20k tokens with the flags below, which is larger than a Summary
    context. ``billed_input_tokens`` therefore says what the request cost, and
    ``context_tokens`` (counted locally with cl100k_base by the caller) says
    how large the context we built was. Only the second is comparable with RQ1.
  * The CLI answers a single prompt, so a two-turn exchange is flattened into
    one prompt with role markers rather than sent as separate messages.

Tool access is denied so the model cannot read the repository and collapse the
distinction between conditions. ``verify_no_file_access`` checks that holds
before a run rather than trusting the flags.
"""

import json
import logging
import random
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

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


class CliBackend:
    """Calls ``claude -p`` and returns the same dict shape as the API path."""

    def __init__(self, system_prompt: str, cli: str = "claude"):
        if shutil.which(cli) is None:
            raise RuntimeError(f"{cli} tidak ditemukan di PATH")
        self.system_prompt = system_prompt
        self.cli = cli

    def _argv(self, model: str) -> list[str]:
        return [
            self.cli, "-p",
            "--model", model,
            "--output-format", "json",
            "--system-prompt", self.system_prompt,
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
                text=True, timeout=CALL_TIMEOUT_SECONDS,
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
        }


def verify_no_file_access(backend: "CliBackend", model: str) -> None:
    """Fail loudly if the subject can still read the repository.

    Called once before a run. The flags are meant to deny every filesystem
    tool, but a flag that silently stops working would invalidate every
    response collected afterwards, so the guarantee is tested rather than
    assumed.
    """
    probe = (
        "Read the file README.md in the current directory and quote its first "
        "line. If you cannot read files, reply with exactly: NO_FILE_ACCESS"
    )
    out = backend.call(model, [{"role": "user", "content": probe}])
    if "NO_FILE_ACCESS" not in out["response"]:
        raise RuntimeError(
            "subjek masih bisa membaca berkas; kondisi Full/Summary/Hybrid "
            f"tidak lagi berbeda. Jawaban: {out['response'][:200]!r}"
        )
    logger.info("Pemeriksaan akses berkas lolos: subjek tidak dapat membaca repositori.")
