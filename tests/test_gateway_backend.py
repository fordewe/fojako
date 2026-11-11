"""Tests for the gateway backend that runs the second judge.

Every test here replaces the HTTP transport with a fake. What is under test is
the isolation the paper relies on: each request carries the judge prompt and
nothing else, no local configuration or environment variable can reach it, no
tool is offered, and the model that answered is the model that was asked for.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from gateway_backend import (  # noqa: E402
    API_KEY_ENV,
    CLEAN_PROBE,
    FAKE_OPERATOR_BLOCK,
    ISOLATION,
    PAYLOAD_KEYS,
    GatewayBackend,
    assert_payload_isolated,
    assess_clean_context,
    build_payload,
    checked_gateway_backend,
    clean_context_probe,
    negative_control,
    same_model,
)

SYSTEM = "You grade answers about a Kotlin/Java codebase. Reply with JSON only."


def _completion(content="ok", *, model="kimi-k3-1-pro", prompt=120, cached=0,
                completion=40, reasoning=25, finish="stop"):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": finish,
            "message": {"role": "assistant", "content": content,
                        "reasoning": "private chain of thought"},
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


class FakeTransport:
    """Returns queued (status, body) pairs and records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "body": body,
                              "timeout": timeout})
        status, payload = self.replies.pop(0)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return status, text


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr("gateway_backend.time.sleep", lambda _: None)


def _backend(transport, **kw):
    return GatewayBackend(SYSTEM, effort="high", api_key="sk-test-secret",
                          transport=transport, **kw)


# ── what a request carries ──────────────────────────────────────────────────

def test_payload_carries_only_the_whitelisted_keys():
    payload = build_payload("kimi-k3-1-pro", SYSTEM,
                            [{"role": "user", "content": "q"}], "high", 32_000)

    assert set(payload) == PAYLOAD_KEYS
    assert payload["model"] == "kimi-k3-1-pro"
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 32_000


def test_payload_is_the_system_prompt_then_the_callers_messages_verbatim():
    payload = build_payload("m", SYSTEM, [{"role": "user", "content": "q"}],
                            "high", 100)

    assert payload["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "q"},
    ]


def test_payload_offers_no_tools_and_sets_no_temperature():
    payload = build_payload("m", SYSTEM, [{"role": "user", "content": "q"}],
                            "high", 100)

    for key in ("tools", "tool_choice", "functions", "temperature", "store",
                "metadata", "web_search_options"):
        assert key not in payload


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(tools=[{"type": "function"}]),
    lambda p: p.update(temperature=0),
    lambda p: p["messages"].append({"role": "tool", "content": "x"}),
    lambda p: p["messages"].append({"role": "user", "content": [{"type": "file"}]}),
])
def test_isolation_guard_rejects_anything_outside_the_whitelist(mutate):
    payload = build_payload("m", SYSTEM, [{"role": "user", "content": "q"}],
                            "high", 100)
    mutate(payload)

    with pytest.raises(RuntimeError, match="isolasi"):
        assert_payload_isolated(payload)


def test_request_sends_the_payload_to_the_chat_endpoint_with_only_the_key(no_wait):
    t = FakeTransport((200, _completion()))
    _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    req = t.requests[0]
    assert req["url"] == "https://kenari.id/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test-secret"
    assert set(req["headers"]) == {"Authorization", "Content-Type", "User-Agent"}
    assert set(req["body"]) == PAYLOAD_KEYS


def test_nothing_from_the_environment_but_the_key_reaches_the_request(no_wait):
    # Run from inside a Claude Code session, the environment carries the
    # session's settings and possibly an Anthropic key. None of it may travel.
    environ = {
        API_KEY_ENV: "sk-kenari",
        "ANTHROPIC_API_KEY": "sk-ant-leak",
        "CLAUDE_EFFORT": "xhigh",
        "CLAUDE_CODE_SESSION": "session-leak",
        "HOME": "/Users/someone",
    }
    t = FakeTransport((200, _completion()))
    backend = GatewayBackend(SYSTEM, effort="high", environ=environ, transport=t)
    backend.call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    sent = json.dumps(t.requests[0])
    assert "sk-kenari" in sent
    for leaked in ("sk-ant-leak", "xhigh", "session-leak", "/Users/someone"):
        assert leaked not in sent


def test_missing_key_fails_at_construction_naming_the_variable():
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        GatewayBackend(SYSTEM, effort="high", environ={}, transport=FakeTransport())


def test_never_starts_a_process(no_wait, monkeypatch):
    # An agent CLI (opencode, claude) would add its own prompt and the
    # operator's rules. The backend must not reach for one.
    def boom(*a, **k):
        raise AssertionError("subprocess dipanggil")
    monkeypatch.setattr("subprocess.run", boom)
    monkeypatch.setattr("subprocess.Popen", boom)

    t = FakeTransport((200, _completion()))
    _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])


# ── what comes back ─────────────────────────────────────────────────────────

def test_result_maps_usage_onto_the_cli_backends_fields(no_wait):
    t = FakeTransport((200, _completion("verdict", prompt=34_000, cached=30_000,
                                        completion=900, reasoning=600)))
    out = _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    assert out["response"] == "verdict"
    assert out["billed_input_tokens"] == 34_000
    assert out["cache_read_tokens"] == 30_000
    assert out["input_tokens"] == 4_000
    assert out["cache_write_tokens"] == 0
    assert out["output_tokens"] == 900
    assert out["thinking_tokens"] == 600
    assert out["attempts"] == 1


def test_result_records_backend_effort_isolation_and_served_model(no_wait):
    t = FakeTransport((200, _completion(model="moonshotai/kimi-k3.1-pro")))
    out = _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    assert out["backend"] == "gateway"
    assert out["reasoning_effort"] == "high"
    assert out["isolation"] == ISOLATION
    assert out["served_model"] == "moonshotai/kimi-k3.1-pro"
    assert out["model"] == "kimi-k3-1-pro"
    assert out["temperature"] is None
    assert out["stop_reason"] == "stop"


def test_reasoning_tokens_are_unknown_when_the_gateway_reports_no_breakdown(no_wait):
    # Kenari folds reasoning into completion_tokens and returns no
    # completion_tokens_details (checked 2026-11-11). A zero would claim the
    # judge did not reason; it did, and the reasoning text came back.
    body = _completion(completion=178)
    del body["usage"]["completion_tokens_details"]
    t = FakeTransport((200, body))
    out = _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    assert out["thinking_tokens"] is None
    assert out["output_tokens"] == 178


def test_reasoning_text_is_not_mixed_into_the_response(no_wait):
    t = FakeTransport((200, _completion("{\"accuracy\": 5}")))
    out = _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    assert "private chain of thought" not in out["response"]


@pytest.mark.parametrize("requested,served,ok", [
    ("kimi-k3-1-pro", "kimi-k3-1-pro", True),
    ("kimi-k3-1-pro", "kimi-k3.1-pro", True),
    ("kimi-k3-1-pro", "moonshotai/kimi-k3.1-pro", True),
    ("kimi-k3-1-pro", "kimi-k3", False),
    ("kimi-k3-1-pro", "glm-5-3", False),
])
def test_same_model_tolerates_separators_but_not_another_model(requested, served, ok):
    assert same_model(requested, served) is ok


def test_a_different_served_model_fails_the_call(no_wait):
    # A gateway that quietly routes to a cheaper model would put another
    # judge's scores under this judge's name.
    t = FakeTransport((200, _completion(model="kimi-k3")))

    with pytest.raises(RuntimeError, match="model"):
        _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])


def test_a_truncated_reply_fails_instead_of_returning_half_a_verdict(no_wait):
    t = FakeTransport((200, _completion("{\"accuracy\": 4", finish="length")))

    with pytest.raises(RuntimeError, match="terpotong"):
        _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])


# ── retries ─────────────────────────────────────────────────────────────────

def test_retries_rate_limits_and_server_errors_then_succeeds(no_wait):
    t = FakeTransport((429, "slow down"), (503, "busy"), (200, _completion()))
    out = _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    assert out["attempts"] == 3
    assert len(t.requests) == 3


@pytest.mark.parametrize("error", [
    # Raised on 2026-11-11 when the connection dropped mid-response; it is an
    # HTTPException, not an OSError, and it stopped the run instead of retrying.
    lambda: __import__("http.client").client.IncompleteRead(b""),
    lambda: __import__("http.client").client.RemoteDisconnected("closed"),
    lambda: ConnectionResetError("reset"),
    lambda: TimeoutError("timed out"),
])
def test_retries_a_connection_lost_mid_request(no_wait, error):
    class Flaky(FakeTransport):
        def __call__(self, url, headers, body, timeout):
            if not self.requests:
                self.requests.append(body)
                raise error()
            return super().__call__(url, headers, body, timeout)

    t = Flaky((200, _completion()))
    out = _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])

    assert out["attempts"] == 2


def test_does_not_retry_a_client_error(no_wait):
    t = FakeTransport((400, '{"error": "bad reasoning_effort"}'))

    with pytest.raises(RuntimeError, match="400"):
        _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])
    assert len(t.requests) == 1


def test_error_messages_never_carry_the_key(no_wait):
    t = FakeTransport((401, '{"error": "invalid key"}'))

    with pytest.raises(RuntimeError) as err:
        _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])
    assert "sk-test-secret" not in str(err.value)


def test_gives_up_after_the_retry_budget(no_wait):
    t = FakeTransport(*[(503, "busy")] * 5)

    with pytest.raises(RuntimeError, match="gagal setelah"):
        _backend(t).call("kimi-k3-1-pro", [{"role": "user", "content": "q"}])


# ── the clean-context probe ─────────────────────────────────────────────────

def test_clean_probe_passes_on_none_with_a_small_prompt(no_wait):
    t = FakeTransport((200, _completion("NONE", prompt=140)))
    probe = clean_context_probe(_backend(t), "kimi-k3-1-pro")

    assess_clean_context(probe)  # does not raise
    assert probe["prompt_tokens"] == 140
    assert probe["prompt_tokens"] < probe["ceiling"]


@pytest.mark.parametrize("answer", [
    "~/.claude/CLAUDE.md", "OPENCODE.md", "You are Kenari, an assistant",
    "rules/common/coding-style.md", "opencode agent instructions",
])
def test_clean_probe_aborts_when_the_model_reports_injected_instructions(
    no_wait, answer
):
    t = FakeTransport((200, _completion(answer, prompt=140)))
    probe = clean_context_probe(_backend(t), "kimi-k3-1-pro")

    with pytest.raises(RuntimeError, match="tidak bersih"):
        assess_clean_context(probe)


def test_clean_probe_aborts_on_token_overhead_even_if_the_model_says_none(no_wait):
    # The bill does not depend on the model's account of itself. A gateway
    # that prepends its own system prompt shows up here.
    t = FakeTransport((200, _completion("NONE", prompt=5_000)))
    probe = clean_context_probe(_backend(t), "kimi-k3-1-pro")

    with pytest.raises(RuntimeError, match="token"):
        assess_clean_context(probe)


def test_injected_block_is_counted_against_the_ceiling_not_into_it(no_wait):
    t = FakeTransport((200, _completion("NONE", prompt=140)))
    plain = clean_context_probe(_backend(t), "kimi-k3-1-pro")
    t2 = FakeTransport((200, _completion("NONE", prompt=2_400)))
    injected = clean_context_probe(_backend(t2), "kimi-k3-1-pro",
                                   injected=FAKE_OPERATOR_BLOCK)

    assert injected["ceiling"] == plain["ceiling"]
    # Injected the way a gateway's own prompt would arrive: as an extra system
    # message ahead of the probe, which stays the last message.
    msgs = t2.requests[0]["body"]["messages"]
    assert {"role": "system", "content": FAKE_OPERATOR_BLOCK} in msgs
    assert msgs[-1] == {"role": "user", "content": CLEAN_PROBE}


def test_fake_operator_block_is_large_enough_to_trip_the_ceiling():
    from token_counter import count_tokens_tiktoken

    assert count_tokens_tiktoken(FAKE_OPERATOR_BLOCK) > 1_500
    assert "CLAUDE.md" in FAKE_OPERATOR_BLOCK


def test_negative_control_reports_whether_the_probe_caught_the_injection(no_wait):
    caught = FakeTransport((200, _completion("NONE", prompt=2_400)))
    missed = FakeTransport((200, _completion("NONE", prompt=140)))

    assert negative_control(_backend(caught), "kimi-k3-1-pro") is True
    assert negative_control(_backend(missed), "kimi-k3-1-pro") is False


def test_checked_backend_runs_the_probe_before_returning(no_wait, monkeypatch):
    t = FakeTransport((200, _completion("NONE", prompt=140)))
    monkeypatch.setenv(API_KEY_ENV, "sk-env")
    monkeypatch.setattr("gateway_backend.urllib_transport", t)

    backend = checked_gateway_backend(SYSTEM, "kimi-k3-1-pro", "high")

    assert isinstance(backend, GatewayBackend)
    assert len(t.requests) == 1


def test_checked_backend_refuses_an_unclean_context(no_wait, monkeypatch):
    t = FakeTransport((200, _completion("NONE", prompt=9_000)))
    monkeypatch.setenv(API_KEY_ENV, "sk-env")
    monkeypatch.setattr("gateway_backend.urllib_transport", t)

    with pytest.raises(RuntimeError, match="tidak bersih"):
        checked_gateway_backend(SYSTEM, "kimi-k3-1-pro", "high")
