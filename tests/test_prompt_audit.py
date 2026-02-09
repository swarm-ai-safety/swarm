"""Tests for prompt audit logging."""

import json

from swarm.logging.prompt_audit import PromptAuditConfig, PromptAuditLog


def test_prompt_audit_logs_hash_by_default(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = PromptAuditLog(
        PromptAuditConfig(
            path=path,
            include_system_prompt=False,
            hash_system_prompt=True,
            max_chars=10,
        )
    )

    log.append_exchange(
        agent_id="a1",
        kind="act",
        epoch=1,
        step=2,
        provider="openai",
        model="gpt-test",
        system_prompt="SYSTEM PROMPT",
        user_prompt="hello world!",
        response_text="response text",
        input_tokens=3,
        output_tokens=4,
        error=None,
        extra={"x": 1},
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["agent_id"] == "a1"
    assert rec["kind"] == "act"
    assert "system_prompt_sha256" in rec
    assert "system_prompt" not in rec
    assert rec["user_prompt"].startswith("hello worl")  # truncated to max_chars
    assert rec["response_text"].startswith("response t")  # truncated to max_chars


def test_prompt_audit_can_include_system_prompt(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = PromptAuditLog(
        PromptAuditConfig(
            path=path, include_system_prompt=True, hash_system_prompt=True, max_chars=10
        )
    )

    log.append_exchange(
        agent_id="a1",
        kind="act",
        epoch=None,
        step=None,
        provider="anthropic",
        model="claude-test",
        system_prompt="SYSTEM PROMPT",
        user_prompt="hello",
        response_text="ok",
        input_tokens=None,
        output_tokens=None,
        error="boom",
        extra=None,
    )

    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["system_prompt"].startswith("SYSTEM PRO")  # truncated to max_chars
    assert "system_prompt_sha256" in rec
