"""Prompt/response audit logging for LLM-backed agents.

Writes newline-delimited JSON (JSONL) records intended for offline security
analysis (e.g., prompt-injection / extraction regression checks).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate(text: Optional[str], max_chars: Optional[int]) -> Optional[str]:
    if text is None:
        return None
    if max_chars is None:
        return text
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    extra = len(text) - max_chars
    return text[:max_chars] + f"...[truncated {extra} chars]"


@dataclass(frozen=True)
class PromptAuditConfig:
    path: Path
    include_system_prompt: bool = False
    hash_system_prompt: bool = True
    max_chars: int = 20_000


class PromptAuditLog:
    """Append-only JSONL logger for LLM prompt/response exchanges."""

    def __init__(self, config: PromptAuditConfig):
        self.config = config
        self._ensure_parent_exists()

    def _ensure_parent_exists(self) -> None:
        try:
            self.config.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create prompt audit log directory: {e}")

    def append(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.config.path, "a", encoding="utf-8") as f:
                try:
                    os.fchmod(f.fileno(), 0o600)
                except Exception as e:
                    logger.warning(f"Failed to set prompt audit log permissions: {e}")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write prompt audit record: {e}")

    def append_exchange(
        self,
        *,
        agent_id: str,
        kind: str,
        epoch: Optional[int],
        step: Optional[int],
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_text: Optional[str],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        system_hash = _sha256_hex(system_prompt) if system_prompt else ""
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "kind": kind,
            "epoch": epoch,
            "step": step,
            "llm": {"provider": provider, "model": model},
            "system_prompt_sha256": system_hash,
            "user_prompt": _truncate(user_prompt, self.config.max_chars),
            "response_text": _truncate(response_text, self.config.max_chars),
            "tokens": {"input": input_tokens, "output": output_tokens},
            "error": _truncate(error, 4_000),
        }

        if self.config.include_system_prompt:
            record["system_prompt"] = _truncate(system_prompt, self.config.max_chars)
        elif self.config.hash_system_prompt:
            # already included above via system_prompt_sha256
            pass
        else:
            record.pop("system_prompt_sha256", None)

        if extra:
            record["extra"] = extra

        self.append(record)
