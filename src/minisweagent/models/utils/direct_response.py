"""Recognize model responses that intentionally answer without a tool call."""

from __future__ import annotations

# Chat Completions use "stop"; the Responses API reports "completed".
_DIRECT_RESPONSE_FINISH_REASONS = frozenset({"stop", "completed", "end_turn"})


def is_direct_response(content: object, finish_reason: object) -> bool:
    """Return True for a non-empty final answer that was not cut off."""
    return isinstance(content, str) and bool(content.strip()) and finish_reason in _DIRECT_RESPONSE_FINISH_REASONS
