import json
import logging
import os
import threading
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent import global_config_dir
from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")

# Cache for models that LiteLLM cannot price. The lookup is deterministic for a
# given (model, provider endpoint, registry, LiteLLM version), so re-running it
# for every conversation only wastes time and repeats the same warning.
_UNPRICED_CACHE_PATH = Path(os.getenv("MSWEA_PRICING_CACHE_PATH") or global_config_dir / "unpriced_models.json")
_UNPRICED_CACHE_VERSION = 1
_unpriced_cache_lock = threading.RLock()
_unpriced_cache: set[str] | None = None

try:
    _LITELLM_VERSION = version("litellm")
except PackageNotFoundError:
    _LITELLM_VERSION = "unknown"

_UNPRICED_ERROR_MARKERS = (
    "isn't mapped",
    "is not mapped",
    "not mapped yet",
    "not registered",
    "model_prices_and_context_window",
    "no pricing",
    "missing pricing",
    "cost must be > 0.0",
)


def _pricing_cache_key(model_name: str, api_base: str, registry: Path | str | None) -> str:
    """Identify a pricing lookup so provider/model changes invalidate the cache."""
    parts = [str(model_name), str(api_base or "")]
    if registry:
        registry_path = Path(registry)
        parts.append(registry_path.as_posix())
        try:
            parts.append(str(registry_path.stat().st_mtime_ns))
        except OSError:
            pass
    parts.append(f"litellm={_LITELLM_VERSION}")
    return "|".join(parts)


def _load_unpriced_cache() -> set[str]:
    global _unpriced_cache
    with _unpriced_cache_lock:
        if _unpriced_cache is None:
            try:
                payload = json.loads(_UNPRICED_CACHE_PATH.read_text(encoding="utf-8"))
                _unpriced_cache = {str(key) for key in payload.get("keys", [])}
            except (OSError, ValueError, TypeError, AttributeError):
                _unpriced_cache = set()
        return _unpriced_cache


def _save_unpriced_cache() -> None:
    with _unpriced_cache_lock:
        if _unpriced_cache is None:
            return
        try:
            _UNPRICED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _UNPRICED_CACHE_PATH.write_text(
                json.dumps({"version": _UNPRICED_CACHE_VERSION, "keys": sorted(_unpriced_cache)}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Could not persist the unpriced model cache", exc_info=True)


def _remember_unpriced(key: str) -> None:
    with _unpriced_cache_lock:
        cache = _load_unpriced_cache()
        if key in cache:
            return
        cache.add(key)
        _save_unpriced_cache()


def _looks_like_missing_pricing(error: object) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _UNPRICED_ERROR_MARKERS)


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    supports_streaming = True
    """Whether ``stream_query`` can emit token deltas as they arrive."""

    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self._missing_cost_warned = False
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def stream_query(self, messages: list[dict[str, str]], callback: Callable[[dict[str, Any]], None]) -> dict:
        """Like ``query``, but emit incremental response events through ``callback``.

        Events have the shape ``{"type": "assistant_delta", "text": "..."}``.
        A final, fully assembled model response is still parsed and returned in
        exactly the same format as :meth:`query`.
        """
        if not self.supports_streaming:
            # Some providers (e.g. the Responses API) use a different streaming
            # event format. Fall back to a single final-message event rather
            # than failing the GUI run.
            message = self.query(messages)
            callback({"type": "assistant_start"})
            text = get_content_string(message)
            if text:
                callback({"type": "assistant_delta", "text": text})
            callback({"type": "assistant_end"})
            return message
        return self.query(messages, stream_callback=callback)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "".join(parts)
        return ""

    def _emit_stream_chunk(self, chunk: Any, callback: Callable[[dict[str, Any]], None]) -> None:
        # LiteLLM emits pydantic objects for most providers, but some custom
        # providers/tests use plain dictionaries for streaming chunks.
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices")
        choices = choices or []
        if not choices:
            return
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if delta is None:
            return

        def field(obj: Any, key: str) -> Any:
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        text = self._content_to_text(field(delta, "content"))
        if text:
            callback({"type": "assistant_delta", "text": text})
        reasoning = field(delta, "reasoning_content") or field(delta, "reasoning")
        reasoning_text = self._content_to_text(reasoning)
        if reasoning_text:
            callback({"type": "reasoning_delta", "text": reasoning_text})
        for tool_call in field(delta, "tool_calls") or []:
            function = field(tool_call, "function")
            arguments = field(function, "arguments") if function is not None else None
            name = field(function, "name") if function is not None else None
            if arguments:
                callback({"type": "tool_call_delta", "text": arguments, "name": name})

    def _query_streaming(
        self,
        messages: list[dict[str, str]],
        callback: Callable[[dict[str, Any]], None],
        **kwargs,
    ) -> Any:
        callback({"type": "assistant_start"})
        chunks = []
        try:
            # ``_query`` forwards these kwargs to the underlying provider. The
            # stream option keeps all provider-specific request construction in
            # one place (and works for the text-based subclass too).
            stream_options = {"include_usage": True, **(kwargs.pop("stream_options", None) or {})}
            stream = self._query(messages, stream=True, stream_options=stream_options, **kwargs)
            for chunk in stream:
                chunks.append(chunk)
                self._emit_stream_chunk(chunk, callback)
        finally:
            callback({"type": "assistant_end"})
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        if response is None:
            msg = "LiteLLM returned an empty response stream"
            raise RuntimeError(msg)
        return response

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        stream_callback = kwargs.pop("stream_callback", None)
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                prepared_messages = self._prepare_messages_for_api(messages)
                if stream_callback is not None and self.supports_streaming:
                    response = self._query_streaming(prepared_messages, stream_callback, **kwargs)
                else:
                    response = self._query(prepared_messages, **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        # Note: all model.query() implementations must persist the response and cost on FormatError.
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:
                # model_dump failed (e.g. unserializable object); fall back to repr
                # so the spec contract ("response MUST be persisted") holds unconditionally.
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _pricing_cache_key(self) -> str:
        return _pricing_cache_key(
            self.config.model_name,
            str(self.config.model_kwargs.get("api_base") or ""),
            self.config.litellm_model_registry,
        )

    def _calculate_cost(self, response) -> dict[str, Any]:
        cache_key = self._pricing_cache_key()
        cached = cache_key in _load_unpriced_cache()
        error: Exception | None = None
        if not cached:
            try:
                cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
                if cost > 0.0:
                    return {"cost": cost, "cost_known": True}
                error = ValueError(f"Cost must be > 0.0, got {cost}")
            except Exception as e:
                error = e
            if _looks_like_missing_pricing(error):
                _remember_unpriced(cache_key)
        else:
            error = RuntimeError(
                "The model has no pricing information (cached; the lookup is skipped until the model or "
                "provider changes)"
            )
        if self.config.cost_tracking != "ignore_errors":
            msg = (
                f"Error calculating cost for model {self.config.model_name}: {error}, perhaps it's not registered? "
                "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                "Alternatively check the 'Cost tracking' section in the documentation at "
                "https://klieret.short.gy/mini-local-models. "
                " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
            )
            logger.critical(msg)
            raise RuntimeError(msg) from error
        if not cached and not self._missing_cost_warned:
            self._missing_cost_warned = True
            logger.warning(
                "No cost information available for model %s (%s). Calls are recorded without cost and are not "
                "added to the total. This is cached in %s until the model or provider changes. Configure "
                "'litellm_model_registry' or use a provider that reports costs to enable cost tracking.",
                self.config.model_name,
                error,
                _UNPRICED_CACHE_PATH,
            )
        return {"cost": 0.0, "cost_known": False}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        message = response.choices[0].message
        return parse_toolcall_actions(
            message.tool_calls or [],
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
            content=message.content,
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
