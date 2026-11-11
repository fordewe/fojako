"""Call a judge model through an OpenAI-compatible gateway, isolated.

The second RQ2 judge (Kimi K3.1 Pro, reached through the Kenari gateway)
answers a question the first judge cannot: whether the conclusions depend on a
judge from the same model family as the subject. It reaches its model over
plain HTTPS, and the isolation is built into the request rather than added
with flags:

  * No process is started. An agent CLI (opencode, claude) would wrap the
    prompt in its own system prompt plus the operator's instruction files and
    rules, the leak the first RQ2 run suffered (RUN-LOG §35).
  * Every request is one POST whose body comes from a whitelist: model,
    messages (the judge's system prompt, then the caller's messages verbatim),
    max_tokens and reasoning_effort. No tools, no stored conversation, no
    temperature. ``assert_payload_isolated`` checks the body again before it
    leaves.
  * The only value read from the environment is the key, from
    ``KENARI_API_KEY``. Session variables and other providers' keys cannot
    reach a request, whichever shell launches the run.
  * The model that answered must be the model that was asked for, on every
    call. A gateway that reroutes would put another model's scores under this
    judge's name.

What no request field can rule out is a system prompt the gateway adds on its
own side. ``clean_context_probe`` measures it: a short probe whose billed
prompt must stay near what our own messages cost, plus the model's own report
of any instructions it received. ``negative_control`` injects a fake operator
block and confirms the probe catches it. ``checked_gateway_backend`` runs the
probe before any paid judging call.

Self-check, before a run (prints the probe and the negative control):
    uv run python evaluation/gateway_backend.py --model kimi-k3-1-pro --effort high
"""

import argparse
import http.client
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_backend import (  # noqa: E402
    MAX_RETRIES,
    OPERATOR_FILE_PATTERN,
    RETRY_BASE_DELAY,
)
from token_counter import count_tokens_tiktoken  # noqa: E402

logger = logging.getLogger(__name__)

GATEWAY_BASE_URL = "https://kenari.id/v1"
API_KEY_ENV = "KENARI_API_KEY"
# Generic on purpose: nothing about the operator or the study travels in it.
USER_AGENT = "fojako-judge"
CALL_TIMEOUT_SECONDS = 600
# Reasoning tokens count against max_tokens. A verdict is about a thousand
# tokens; the rest is room for reasoning, billed only when used.
DEFAULT_MAX_TOKENS = 32_000

PAYLOAD_KEYS = frozenset({"model", "messages", "max_tokens", "reasoning_effort"})
MESSAGE_ROLES = frozenset({"system", "user", "assistant"})
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Recorded on every result, as the CLI backend records its own.
ISOLATION = (
    "direct HTTPS to the gateway; payload whitelist (model, messages, "
    "max_tokens, reasoning_effort); no tools; no agent or CLI; key from "
    f"{API_KEY_ENV} only"
)

# What the model would name if the operator's setup or the gateway's own
# instructions reached it: the CLI backend's list plus the agent and the
# gateway this judge could have been run through.
INJECTED_PATTERN = re.compile(
    OPERATOR_FILE_PATTERN.pattern + r"|OPENCODE\.md|opencode|kenari", re.I
)

CLEAN_PROBE = (
    "List every instruction file, memory file, rules file, hook text or "
    "operator instruction present in your context, one per line with its "
    "path or first heading. Do not list this message or the system prompt "
    "you were given. If there are none, reply with exactly: NONE"
)
# Our own messages, counted with cl100k_base, may cost up to twice as much
# under another tokenizer, plus the chat template. A gateway prompt of any
# real size lands above that; a short one may not, which RUN-LOG states.
PROBE_TOKEN_FACTOR = 2
PROBE_TOKEN_SLACK = 200


def _fake_operator_block() -> str:
    lines = [
        "# ~/.claude/CLAUDE.md (negative control; not a real file)",
        "## Global rules",
    ]
    lines += [
        f"- Rule {i}: before answering, consult rules/common/rule-{i}.md and "
        f"apply the house style it sets for section {i} of every reply."
        for i in range(1, 121)
    ]
    return "\n".join(lines)


# Stands in for a gateway or agent prompt in the negative control: large, and
# named like the operator files the first run leaked.
FAKE_OPERATOR_BLOCK = _fake_operator_block()

Transport = Callable[[str, Mapping[str, str], dict, float], tuple[int, str]]


def urllib_transport(url: str, headers: Mapping[str, str], body: dict,
                     timeout: float) -> tuple[int, str]:
    """POST one JSON body; return the status and the raw text."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=dict(headers),
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def assert_payload_isolated(payload: dict) -> None:
    """Refuse a request body that carries anything outside the whitelist."""
    extra = set(payload) - PAYLOAD_KEYS
    if extra:
        raise RuntimeError(
            f"isolasi dilanggar: kunci di luar daftar putih {sorted(extra)}"
        )
    for message in payload["messages"]:
        if (set(message) != {"role", "content"}
                or message["role"] not in MESSAGE_ROLES
                or not isinstance(message["content"], str)):
            raise RuntimeError(f"isolasi dilanggar: pesan tidak sah {message!r:.120}")


def build_payload(model: str, system_prompt: str, messages: list[dict],
                  effort: str, max_tokens: int) -> dict:
    """The request body: the judge's system prompt, then the caller's messages."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *({"role": m["role"], "content": m["content"]} for m in messages),
        ],
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
    }
    assert_payload_isolated(payload)
    return payload


def _normalize(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model_id.rsplit("/", 1)[-1].lower())


def same_model(requested: str, served: str) -> bool:
    """Equal up to a vendor prefix and separators: kimi-k3-1-pro = kimi-k3.1-pro."""
    return bool(served) and _normalize(requested) == _normalize(served)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


class GatewayBackend:
    """Calls the gateway and returns the same dict shape as ``CliBackend``."""

    def __init__(self, system_prompt: str, effort: str = "high", *,
                 base_url: str = GATEWAY_BASE_URL, api_key: str | None = None,
                 environ: Mapping[str, str] | None = None,
                 transport: Transport | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        env = os.environ if environ is None else environ
        key = api_key or env.get(API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"{API_KEY_ENV} belum diset; kunci gateway hanya dibaca dari "
                "variabel itu"
            )
        self.system_prompt = system_prompt
        self.effort = effort
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self._key = key
        self.transport = transport or urllib_transport

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***")

    @staticmethod
    def _wait(attempt: int, reason: str) -> None:
        delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
        logger.warning("  gateway: %s, percobaan %d/%d, tunggu %.1fs",
                       reason, attempt + 1, MAX_RETRIES, delay)
        time.sleep(delay)

    def call(self, model: str, messages: list[dict]) -> dict:
        payload = build_payload(model, self.system_prompt, messages,
                                self.effort, self.max_tokens)
        url = f"{self.base_url}/chat/completions"
        last_error = None

        for attempt in range(MAX_RETRIES):
            start = time.time()
            try:
                status, text = self.transport(url, self._headers(), payload,
                                              CALL_TIMEOUT_SECONDS)
            # HTTPException covers IncompleteRead: on 2026-11-11 a connection
            # dropped mid-response raised it, and the run stopped instead of
            # retrying, because it is not an OSError.
            except (urllib.error.URLError, http.client.HTTPException,
                    TimeoutError, OSError) as e:
                last_error = type(e).__name__
                self._wait(attempt, last_error)
                continue
            elapsed = time.time() - start

            if status in RETRYABLE_STATUS:
                last_error = f"HTTP {status}"
                self._wait(attempt, last_error)
                continue
            if status != 200:
                raise RuntimeError(
                    f"gateway menolak permintaan (HTTP {status}): "
                    f"{self._redact(text)[:300]}"
                )
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                last_error = "jawaban gateway bukan JSON"
                self._wait(attempt, last_error)
                continue
            return self._result(body, model, elapsed, attempt + 1)

        raise RuntimeError(f"gagal setelah {MAX_RETRIES} percobaan: {last_error}")

    def _result(self, body: dict, model: str, elapsed: float, attempts: int) -> dict:
        served = body.get("model") or ""
        if not same_model(model, served):
            raise RuntimeError(
                f"model yang menjawab ({served!r}) bukan model yang diminta ({model!r})"
            )
        choice = (body.get("choices") or [{}])[0]
        finish = choice.get("finish_reason")
        content = _text((choice.get("message") or {}).get("content"))
        if finish == "length":
            raise RuntimeError(f"jawaban terpotong di max_tokens={self.max_tokens}")
        if finish not in ("stop", "end_turn", None):
            raise RuntimeError(f"jawaban berhenti karena {finish!r}")
        if not content.strip():
            raise RuntimeError("jawaban gateway kosong")

        usage = body.get("usage") or {}
        prompt = usage.get("prompt_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        # None when the gateway gives no breakdown: Kenari folds reasoning
        # into completion_tokens (checked 2026-11-11), and a zero would claim
        # the judge did not reason.
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        return {
            # The answer only. The gateway returns the reasoning text in a
            # separate field, which is left out of the verdict.
            "response": content,
            "input_tokens": prompt - cached,
            "cache_write_tokens": 0,
            "cache_read_tokens": cached,
            "billed_input_tokens": prompt,
            "output_tokens": usage.get("completion_tokens") or 0,
            "thinking_tokens": reasoning,
            "latency_seconds": round(elapsed, 2),
            "model": model,
            "served_model": served,
            # Not sent: the first judge's CLI could not set it either.
            "temperature": None,
            "reasoning_effort": self.effort,
            "stop_reason": finish,
            "reported_cost_usd": None,
            "attempts": attempts,
            "backend": "gateway",
            "gateway": self.base_url,
            "isolation": ISOLATION,
        }


def clean_context_probe(backend: GatewayBackend, model: str,
                        injected: str = "") -> dict:
    """Ask the model what instructions it holds, and read the bill for it.

    ``injected`` is for the negative control only: a block sent as an extra
    system message, as a gateway's own prompt would arrive, and left out of
    the ceiling, as a gateway's prompt would be.
    """
    expected = (count_tokens_tiktoken(backend.system_prompt)
                + count_tokens_tiktoken(CLEAN_PROBE))
    messages = [{"role": "user", "content": CLEAN_PROBE}]
    if injected:
        messages.insert(0, {"role": "system", "content": injected})
    out = backend.call(model, messages)
    return {
        "answer": out["response"],
        "prompt_tokens": out["billed_input_tokens"],
        "expected_tokens": expected,
        "ceiling": PROBE_TOKEN_FACTOR * expected + PROBE_TOKEN_SLACK,
        "served_model": out["served_model"],
    }


def assess_clean_context(probe: dict) -> None:
    """Fail loudly if anything beyond our own messages reached the model.

    Two checks, because either alone can be fooled: the model's report of the
    instructions it holds, and the bill, which does not depend on that report.
    """
    if INJECTED_PATTERN.search(probe["answer"]):
        raise RuntimeError(
            "konteks juri tidak bersih: model melaporkan instruksi sisipan. "
            f"Jawaban: {probe['answer'][:300]!r}"
        )
    if probe["prompt_tokens"] > probe["ceiling"]:
        raise RuntimeError(
            f"konteks juri tidak bersih: probe ditagih {probe['prompt_tokens']:,} "
            f"token masukan, batas {probe['ceiling']:,}"
        )


def negative_control(backend: GatewayBackend, model: str) -> bool:
    """True when the probe rejects a context carrying a fake operator block."""
    probe = clean_context_probe(backend, model, injected=FAKE_OPERATOR_BLOCK)
    try:
        assess_clean_context(probe)
    except RuntimeError:
        return True
    return False


def checked_gateway_backend(system_prompt: str, model: str,
                            effort: str) -> GatewayBackend:
    """A backend whose context passed the probe, ready for paid calls."""
    backend = GatewayBackend(system_prompt, effort=effort)
    probe = clean_context_probe(backend, model)
    assess_clean_context(probe)
    logger.info(
        "Pemeriksaan konteks gateway lolos: probe %d token (batas %d), model %s.",
        probe["prompt_tokens"], probe["ceiling"], probe["served_model"],
    )
    return backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolation self-check for the gateway judge"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="high")
    args = parser.parse_args()

    from judge_responses import JUDGE_SYSTEM_PROMPT

    backend = GatewayBackend(JUDGE_SYSTEM_PROMPT, effort=args.effort)

    clean = clean_context_probe(backend, args.model)
    print(f"probe bersih : {clean['prompt_tokens']:,} token "
          f"(perkiraan pesan sendiri {clean['expected_tokens']:,}, "
          f"batas {clean['ceiling']:,}), model {clean['served_model']}")
    print(f"  jawaban    : {clean['answer'][:200]!r}")
    assess_clean_context(clean)
    print("  -> LOLOS")

    dirty = clean_context_probe(backend, args.model, injected=FAKE_OPERATOR_BLOCK)
    print(f"negative control: {dirty['prompt_tokens']:,} token (batas "
          f"{dirty['ceiling']:,})")
    print(f"  jawaban    : {dirty['answer'][:200]!r}")
    try:
        assess_clean_context(dirty)
    except RuntimeError as e:
        print(f"  -> DITOLAK seperti seharusnya: {e}")
    else:
        raise SystemExit("negative control gagal: probe tidak menangkap blok sisipan")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
