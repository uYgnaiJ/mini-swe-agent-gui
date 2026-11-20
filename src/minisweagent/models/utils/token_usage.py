from collections.abc import Mapping
from typing import Any


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return None


def _first_int(source: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def extract_token_usage(message: Mapping[str, Any]) -> dict[str, int]:
    """Normalize token usage from a model message or API response.

    Completion APIs use ``prompt_tokens``/``completion_tokens`` while the
    Responses API uses ``input_tokens``/``output_tokens``. Models also store
    the raw response in different places. Check the common locations and
    return one stable shape for agent events and the GUI.
    """
    extra = _as_mapping(message.get("extra")) or {}
    candidates: list[Mapping[str, Any]] = [message, extra]
    for container in (message, extra):
        for key in ("usage", "token_usage"):
            if usage := _as_mapping(container.get(key)):
                candidates.append(usage)
        if response := _as_mapping(container.get("response")):
            candidates.append(response)
            if usage := _as_mapping(response.get("usage")):
                candidates.append(usage)

    for candidate in candidates:
        prompt = _first_int(candidate, "prompt_tokens", "input_tokens", "prompt_token_count", "input_token_count")
        completion = _first_int(
            candidate, "completion_tokens", "output_tokens", "completion_token_count", "output_token_count"
        )
        total = _first_int(candidate, "total_tokens", "tokens", "token_count")
        if prompt or completion or total:
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total or prompt + completion,
            }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
