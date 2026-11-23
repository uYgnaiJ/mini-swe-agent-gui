#!/usr/bin/env python3
"""A basic web GUI for mini-SWE-agent with live streaming responses.

Run ``mini-gui`` (or ``python -m minisweagent.run.gui``), open the printed URL,
enter a task, and watch the agent think and execute commands in real time.

The implementation intentionally uses only the Python standard library for the
web server, plus Server-Sent Events (SSE) for streaming. This keeps the GUI
lightweight and avoids adding another web-framework dependency to the package.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import typer
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.exceptions import UserInterruption
from minisweagent.models import get_model
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.run.utilities.conversation_store import ConversationStore
from minisweagent.utils.serialize import UNSET, recursive_merge

logger = logging.getLogger("minisweagent.gui")

DEFAULT_CONFIG_FILE = Path(builtin_config_dir / "mini.yaml")
DEFAULT_OUTPUT_FILE = global_config_dir / "last_gui_run.traj.json"
DEFAULT_DATABASE_FILE = Path(os.getenv("MSWEA_GUI_DB_PATH") or global_config_dir / "gui_conversations.db")
DEFAULT_SETTINGS_FILE = Path(os.getenv("MSWEA_GUI_SETTINGS_PATH") or global_config_dir / "gui_settings.json")


class SettingsConflictError(ValueError):
    """Raised when a provider or model would be created twice."""


class SettingsNotFoundError(ValueError):
    """Raised when a provider or model cannot be found."""


_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "openai": ("OpenAI", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", ""),
    "gemini": ("Google Gemini", "GEMINI_API_KEY", ""),
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "portkey": ("Portkey", "PORTKEY_API_KEY", ""),
    "requesty": ("Requesty", "REQUESTY_API_KEY", ""),
    "azure": ("Azure OpenAI", "AZURE_API_KEY", ""),
    "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY", ""),
    "groq": ("Groq", "GROQ_API_KEY", ""),
    "mistral": ("Mistral", "MISTRAL_API_KEY", ""),
    "xai": ("xAI", "XAI_API_KEY", ""),
    "ollama": ("Ollama", "OLLAMA_API_KEY", "http://localhost:11434"),
    "vllm": ("vLLM", "VLLM_API_KEY", "http://localhost:8000/v1"),
    "together": ("Together AI", "TOGETHER_API_KEY", ""),
    "cohere": ("Cohere", "COHERE_API_KEY", ""),
}

# LiteLLM expects model names to include their provider prefix. Model-list
# endpoints normally return provider-local IDs (for example ``gpt-4o``), so
# loaded models have to be qualified before they can be called.
_PROVIDER_MODEL_PREFIXES: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "openrouter": "openrouter",
    "portkey": "portkey",
    "requesty": "requesty",
    "azure": "azure",
    "deepseek": "deepseek",
    "groq": "groq",
    "mistral": "mistral",
    "xai": "xai",
    "ollama": "ollama",
    "vllm": "openai",
    "together": "together_ai",
    "cohere": "cohere",
}

# Fallbacks for providers that are configured by base URL instead of a known id.
_BASE_URL_MODEL_PREFIXES: dict[str, str] = {
    "volces.com": "volcengine",
}


def _slugify(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "provider"


_SECRET_LIKE_PATTERN = re.compile(r"(?i)^(?:sk[-_]|pk[-_]|xox[baprs]-|ghp_|github_pat_|AIza|ya29\.)")


def _validate_api_key_env_name(value: Any) -> None:
    """Reject accidental API keys in the environment-variable-name field.

    The field is meant to hold a variable name such as ``OPENAI_API_KEY``. A
    pasted secret normally has whitespace, is much longer than a variable name,
    or carries a provider-specific prefix. Catching those cases here gives a
    useful error even when the GUI's client-side validation is bypassed.
    """
    env_name = str(value or "").strip()
    if not env_name:
        return
    if any(char.isspace() for char in env_name) or len(env_name) > 80 or _SECRET_LIKE_PATTERN.match(env_name):
        raise ValueError(
            "The API key environment variable must be a variable name such as OPENAI_API_KEY, "
            "not the API key itself. Paste the secret key into the API key field instead."
        )


def _provider_from_model(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        return ""
    if "/" in value:
        return _slugify(value.split("/", 1)[0])
    return _slugify(value.split("-", 1)[0])


def _provider_model_prefix(provider_id: str, base_url: str = "") -> str:
    """Return the LiteLLM provider prefix for a provider id or its API base URL.

    Provider ids registered in ``_PROVIDER_MODEL_PREFIXES`` win. Otherwise the
    base URL is inspected, so providers configured only by URL (for example
    VolcEngine Ark at ``ark.cn-beijing.volces.com``) still get the prefix that
    LiteLLM's model registry uses for pricing.
    """
    slug = _slugify(provider_id)
    if slug in _PROVIDER_MODEL_PREFIXES:
        return _PROVIDER_MODEL_PREFIXES[slug]
    host = (urlparse(str(base_url or "")).hostname or "").lower()
    for domain, prefix in _BASE_URL_MODEL_PREFIXES.items():
        if host == domain or host.endswith(f".{domain}"):
            return prefix
    return "openai"


def _qualified_model_name(provider_id: str, model_name: str, base_url: str = "") -> str:
    """Add the provider prefix LiteLLM needs to a provider-local model name."""
    name = str(model_name or "").strip()
    if not name:
        return name
    prefix = _provider_model_prefix(provider_id, base_url)
    if _slugify(provider_id) not in _PROVIDER_MODEL_PREFIXES and name.startswith("openai/"):
        # Models discovered before the base URL was known were stored with the
        # generic openai/ prefix; replace it with the provider-specific one so
        # LiteLLM can find pricing information.
        name = name.removeprefix("openai/")
    return name if name == prefix or name.startswith(f"{prefix}/") else f"{prefix}/{name}"


def _model_is_enabled_by_provider(model: dict[str, Any], providers: list[dict[str, Any]]) -> bool:
    provider_id = str(model.get("provider") or "")
    if not provider_id:
        return True
    provider = next((item for item in providers if item.get("id") == provider_id), None)
    return True if provider is None else bool(provider.get("enabled", True))


class ProviderModelQueryError(ValueError):
    """Raised when a provider's model list cannot be retrieved."""


_PROVIDER_MODELS_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "together": "https://api.together.xyz/v1",
    "cohere": "https://api.cohere.com/v1",
    "portkey": "https://api.portkey.ai/v1",
    "requesty": "https://router.requesty.ai/v1",
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000/v1",
}


def _provider_api_key(provider: dict[str, Any]) -> str:
    """Return a stored key, falling back to the provider's configured env var."""
    api_key = str(provider.get("api_key") or "").strip()
    if api_key:
        return api_key
    env_name = str(provider.get("api_key_env") or "").strip()
    return os.getenv(env_name, "").strip() if env_name else ""


def _provider_models_endpoint(provider: dict[str, Any]) -> tuple[str, str]:
    """Build the provider's model-list endpoint and return it with the API key."""
    provider_id = _slugify(provider.get("id"))
    base_url = str(provider.get("base_url") or _PROVIDER_MODELS_BASE_URLS.get(provider_id) or "").strip()
    if not base_url:
        raise ProviderModelQueryError(
            f"Provider {provider_id!r} has no base URL. Add the provider's API base URL before loading models."
        )
    endpoint = base_url.rstrip("/")
    if re.search(r"/(?:models|api/tags)$", endpoint):
        pass
    elif provider_id == "ollama" and not endpoint.rstrip("/").endswith("/v1"):
        endpoint += "/api/tags"
    elif provider_id == "openai" and endpoint == "https://api.openai.com":
        endpoint += "/v1/models"
    else:
        endpoint += "/models"
    api_key = _provider_api_key(provider)
    if provider_id == "gemini" and api_key:
        separator = "&" if "?" in endpoint else "?"
        endpoint += separator + urlencode({"key": api_key})
    return endpoint, api_key


def _extract_provider_models(payload: Any) -> list[dict[str, str]]:
    """Normalize the common model-list response shapes returned by providers."""
    items: list[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("data", "models", "results", "items", "available_models"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
            if isinstance(value, dict):
                nested = _extract_provider_models(value)
                if nested:
                    return nested
        if not items and any(payload.get(key) for key in ("id", "model", "model_name", "name")):
            items = [payload]

    models: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            model_name = item.strip()
            display_name = model_name
        elif isinstance(item, dict):
            model_name = str(
                item.get("id") or item.get("model_name") or item.get("model") or item.get("name") or ""
            ).strip()
            display_name = str(
                item.get("display_name") or item.get("displayName") or item.get("name") or model_name
            ).strip()
        else:
            continue
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        models.append({"model_name": model_name, "name": display_name or model_name})
    return models


def _query_provider_models(provider: dict[str, Any], *, timeout: float = 15.0) -> list[dict[str, str]]:
    """Query an OpenAI-compatible (or known provider) model-list endpoint."""
    endpoint, api_key = _provider_models_endpoint(provider)
    provider_id = _slugify(provider.get("id"))
    headers = {"Accept": "application/json", "User-Agent": "mini-swe-agent-gui"}
    if api_key:
        if provider_id == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        elif provider_id == "gemini":
            headers["x-goog-api-key"] = api_key
        elif provider_id == "azure":
            headers["api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        try:
            body = response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:  # noqa: BLE001 - best-effort diagnostic only
            pass
        message = f"Could not load models from {provider.get('name') or provider_id}: HTTP {exc.code}"
        if detail:
            message += f": {detail[:500]}"
        raise ProviderModelQueryError(message) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ProviderModelQueryError(
            f"Could not load models from {provider.get('name') or provider_id}: {reason}"
        ) from exc

    try:
        payload = json.loads(body if isinstance(body, str) else body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderModelQueryError(
            f"Provider {provider.get('name') or provider_id} returned a non-JSON model list."
        ) from exc
    return _extract_provider_models(payload)


class GuiModelSettings:
    """Provider/model settings used by the GUI model switcher.

    The file is intentionally a small JSON document rather than a database
    migration: it contains only a handful of user-managed providers and model
    names and can be edited independently of the conversation history.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        default_model: str = "",
        default_model_class: str = "",
    ) -> None:
        self.path = Path(path or DEFAULT_SETTINGS_FILE)
        self.default_model = str(default_model or "").strip()
        self.default_model_class = str(default_model_class or "").strip()
        self.active_model = ""
        self.providers: list[dict[str, Any]] = []
        self.models: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._load()
        self._ensure_defaults()

    @staticmethod
    def _copy(value: Any) -> Any:
        return copy.deepcopy(value)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read GUI model settings from %s", self.path, exc_info=True)
            return
        if not isinstance(payload, dict):
            return
        self.active_model = str(payload.get("active_model") or payload.get("model_name") or "").strip()
        providers = payload.get("providers")
        if isinstance(providers, list):
            for provider in providers:
                if isinstance(provider, dict) and provider.get("id"):
                    self.providers.append(self._normalize_provider(provider))
        models = payload.get("models")
        if isinstance(models, list):
            for model in models:
                if isinstance(model, dict):
                    normalized = self._normalize_model(model)
                    if normalized["model_name"]:
                        self.models.append(normalized)
        self._deduplicate()

    def _deduplicate(self) -> None:
        seen_providers: set[str] = set()
        unique_providers: list[dict[str, Any]] = []
        for provider in self.providers:
            provider_id = str(provider.get("id") or "")
            if provider_id and provider_id not in seen_providers:
                seen_providers.add(provider_id)
                unique_providers.append(provider)
        self.providers = unique_providers
        seen_models: set[str] = set()
        unique_models: list[dict[str, Any]] = []
        for model in self.models:
            model_name = str(model.get("model_name") or "")
            if model_name and model_name not in seen_models:
                seen_models.add(model_name)
                unique_models.append(model)
        self.models = unique_models

    @staticmethod
    def _normalize_provider(provider: dict[str, Any]) -> dict[str, Any]:
        name = str(provider.get("name") or provider.get("provider_name") or provider.get("id") or "Provider").strip()
        provider_id = _slugify(provider.get("id") or provider.get("provider_id") or provider.get("provider") or name)
        defaults = _PROVIDER_DEFAULTS.get(provider_id, (name, f"{provider_id.upper().replace('-', '_')}_API_KEY", ""))
        return {
            "id": provider_id,
            "name": name or defaults[0],
            "enabled": bool(provider.get("enabled", True)),
            "api_key_env": str(provider.get("api_key_env") or defaults[1]),
            "base_url": str(provider.get("base_url") or defaults[2]),
            "api_key": str(provider.get("api_key") or ""),
            "has_api_key": bool(provider.get("api_key") or provider.get("has_api_key")),
        }

    @staticmethod
    def _normalize_model(model: dict[str, Any]) -> dict[str, Any]:
        model_name = str(
            model.get("model_name") or model.get("model") or model.get("name") or model.get("id") or ""
        ).strip()
        provider = str(model.get("provider") or model.get("provider_id") or "").strip()
        if not provider:
            provider = _provider_from_model(model_name)
        return {
            "id": model_name,
            "model_name": model_name,
            "name": str(model.get("name") or model_name).strip(),
            "provider": provider,
            "model_class": str(model.get("model_class") or "").strip(),
            "enabled": bool(model.get("enabled", True)),
        }

    def _ensure_defaults(self) -> None:
        persisted_active = bool(self.active_model)
        if self.default_model:
            self._ensure_model(
                self.default_model,
                provider=_provider_from_model(self.default_model),
                model_class=self.default_model_class,
                enabled=True,
            )
            if not persisted_active:
                self.active_model = self.default_model
        if not self.active_model:
            enabled = self._enabled_models_locked()
            if enabled:
                self.active_model = str(enabled[0]["model_name"])
        elif not any(model.get("model_name") == self.active_model for model in self.models):
            enabled = self._enabled_models_locked()
            self.active_model = str(enabled[0]["model_name"]) if enabled else self.default_model
        self._ensure_active_model_locked()

    def _ensure_provider(self, provider_id: str) -> dict[str, Any]:
        provider = next((item for item in self.providers if item.get("id") == provider_id), None)
        if provider is not None:
            return provider
        default_name = _PROVIDER_DEFAULTS.get(provider_id, (provider_id.replace("-", " ").title(),))[0]
        provider = self._normalize_provider({"id": provider_id, "name": default_name})
        self.providers.append(provider)
        return provider

    def _ensure_model(
        self, model_name: str, *, provider: str = "", model_class: str = "", enabled: bool = True
    ) -> dict[str, Any]:
        model = next((item for item in self.models if item.get("model_name") == model_name), None)
        if model is not None:
            return model
        provider_id = _slugify(provider) if provider else _provider_from_model(model_name)
        self._ensure_provider(provider_id)
        model = self._normalize_model(
            {
                "model_name": model_name,
                "name": model_name,
                "provider": provider_id,
                "model_class": model_class,
                "enabled": enabled,
            }
        )
        if not model["provider"]:
            model["provider"] = provider_id
        self.models.append(model)
        return model

    def _find_provider_locked(self, provider_id: str) -> dict[str, Any]:
        key = unquote(str(provider_id or ""))
        provider = next((item for item in self.providers if item.get("id") == key or item.get("name") == key), None)
        if provider is None:
            raise SettingsNotFoundError(f"Unknown provider: {key}")
        return provider

    def _find_model_locked(self, model_name: str) -> dict[str, Any]:
        key = unquote(str(model_name or ""))
        model = next(
            (item for item in self.models if item.get("model_name") == key or item.get("id") == key),
            None,
        )
        if model is None:
            raise SettingsNotFoundError(f"Unknown model: {key}")
        return model

    def _enabled_models_locked(self) -> list[dict[str, Any]]:
        return [
            self._copy(model)
            for model in self.models
            if model.get("enabled", True) and _model_is_enabled_by_provider(model, self.providers)
        ]

    def list_providers(self) -> list[dict[str, Any]]:
        with self._lock:
            providers = self._copy(self.providers)
            model_counts: dict[str, int] = {}
            provider_models: dict[str, list[str]] = {}
            for model in self.models:
                provider = str(model.get("provider") or "")
                model_counts[provider] = model_counts.get(provider, 0) + 1
                provider_models.setdefault(provider, []).append(str(model["model_name"]))
            for provider in providers:
                provider_id = str(provider["id"])
                provider["model_count"] = model_counts.get(provider_id, 0)
                provider["models"] = provider_models.get(provider_id, [])
            return providers

    def get_provider(self, provider_id: str) -> dict[str, Any]:
        with self._lock:
            return self._copy(self._find_provider_locked(provider_id))

    def list_models(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            return self._enabled_models_locked() if enabled_only else self._copy(self.models)

    def get_model(self, model_name: str) -> dict[str, Any]:
        with self._lock:
            return self._copy(self._find_model_locked(model_name))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            enabled_models = self._enabled_models_locked()
            return {
                "active_model": self.active_model,
                "model_name": self.active_model,
                "providers": self.list_providers(),
                "models": self._copy(self.models),
                "enabled_models": enabled_models,
                "enabled_model_names": [str(model["model_name"]) for model in enabled_models],
                "settings_path": str(self.path),
            }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_model": self.active_model,
            "providers": self._copy(self.providers),
            "models": self._copy(self.models),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def add_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            _validate_api_key_env_name(payload.get("api_key_env"))
            provider = self._normalize_provider(payload)
            if any(item.get("id") == provider["id"] for item in self.providers):
                raise SettingsConflictError(f"Provider already exists: {provider['id']}")
            self.providers.append(provider)
            self._save()
            return self._copy(provider)

    def fetch_available_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        """Fetch a provider's model list without changing saved settings."""
        with self._lock:
            provider = self._copy(self._find_provider_locked(provider_id))
        if isinstance(overrides, dict):
            if "api_key_env" in overrides:
                _validate_api_key_env_name(overrides.get("api_key_env"))
            for key in ("name", "api_key_env", "base_url", "api_key"):
                if overrides.get(key):
                    provider[key] = str(overrides[key]).strip()
        return _query_provider_models(provider, timeout=timeout)

    def query_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        """Alias for :meth:`fetch_available_models`."""
        return self.fetch_available_models(provider_id, overrides, timeout=timeout)

    def fetch_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        """Alias for :meth:`fetch_available_models`."""
        return self.fetch_available_models(provider_id, overrides, timeout=timeout)

    def load_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Fetch a provider's models and add newly discovered ones as enabled models."""
        discovered = self.fetch_available_models(provider_id, overrides, timeout=timeout)
        with self._lock:
            provider = self._find_provider_locked(provider_id)
            existing = {str(model.get("model_name") or ""): model for model in self.models}
            added: list[dict[str, Any]] = []
            for choice in discovered:
                discovered_name = str(choice.get("model_name") or "").strip()
                model_name = _qualified_model_name(provider["id"], discovered_name, provider.get("base_url", ""))
                if not model_name or model_name in existing:
                    continue
                legacy_model = existing.get(discovered_name)
                if legacy_model is None:
                    # Models stored before the provider prefix could be derived
                    # still carry a different prefix (e.g. openai/ instead of
                    # volcengine/); upgrade them instead of adding a duplicate.
                    legacy_model = next(
                        (
                            model
                            for model in existing.values()
                            if model.get("provider") == provider["id"]
                            and str(model.get("model_name") or "").rsplit("/", 1)[-1] == discovered_name
                        ),
                        None,
                    )
                if legacy_model is not None and legacy_model.get("provider") == provider["id"]:
                    # Upgrade models loaded before names were provider-qualified.
                    legacy_key = str(legacy_model.get("model_name") or "")
                    legacy_model["id"] = model_name
                    legacy_model["model_name"] = model_name
                    existing.pop(legacy_key, None)
                    existing[model_name] = legacy_model
                    continue
                model = self._normalize_model(
                    {
                        "model_name": model_name,
                        "name": choice.get("name") or discovered_name or model_name,
                        "provider": provider["id"],
                        "enabled": True,
                    }
                )
                self.models.append(model)
                existing[model_name] = model
                added.append(self._copy(model))
            self._ensure_active_model_locked()
            self._save()
            snapshot = self.snapshot()
            provider_entry = next(
                (item for item in snapshot["providers"] if item.get("id") == provider["id"]), self._copy(provider)
            )
            fetched_names = [choice["model_name"] for choice in discovered]
            snapshot.update(
                {
                    "provider": provider_entry,
                    "discovered_models": self._copy(discovered),
                    "queried_models": self._copy(discovered),
                    "fetched_models": fetched_names,
                    "available_models": fetched_names,
                    "provider_models": fetched_names,
                    "remote_models": fetched_names,
                    "model_names": fetched_names,
                    "loaded_models": added,
                    "added_models": self._copy(added),
                    "message": (
                        f"Loaded {len(added)} new model(s) from {provider.get('name') or provider['id']}."
                        if added
                        else f"No new models to add from {provider.get('name') or provider['id']}."
                    ),
                }
            )
            return snapshot

    def refresh_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Alias for :meth:`load_provider_models`."""
        return self.load_provider_models(provider_id, overrides, timeout=timeout)

    def sync_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Alias for :meth:`load_provider_models`."""
        return self.load_provider_models(provider_id, overrides, timeout=timeout)

    def fetch_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        """Alias for :meth:`fetch_available_models`."""
        return self.fetch_available_models(provider_id, overrides, timeout=timeout)

    def query_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        """Alias for :meth:`fetch_available_models`."""
        return self.fetch_available_models(provider_id, overrides, timeout=timeout)

    def discover_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        """Alias for :meth:`fetch_available_models`."""
        return self.fetch_available_models(provider_id, overrides, timeout=timeout)

    def load_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Alias for :meth:`load_provider_models`."""
        return self.load_provider_models(provider_id, overrides, timeout=timeout)

    def load_models_from_provider(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Alias for :meth:`load_provider_models`."""
        return self.load_provider_models(provider_id, overrides, timeout=timeout)

    def refresh_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Alias for :meth:`load_provider_models`."""
        return self.load_provider_models(provider_id, overrides, timeout=timeout)

    def update_provider(self, provider_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            provider = self._find_provider_locked(provider_id)
            if "api_key_env" in payload:
                _validate_api_key_env_name(payload.get("api_key_env"))
            if "name" in payload and str(payload.get("name") or "").strip():
                provider["name"] = str(payload["name"]).strip()
            if "api_key_env" in payload:
                provider["api_key_env"] = str(payload.get("api_key_env") or "").strip()
            if "base_url" in payload:
                provider["base_url"] = str(payload.get("base_url") or "").strip()
            if "api_key" in payload:
                api_key = str(payload.get("api_key") or "")
                # The edit form deliberately leaves this field blank so that
                # changing a name or base URL does not erase a stored key.
                if api_key or bool(payload.get("clear_api_key")):
                    provider["api_key"] = api_key
                    provider["has_api_key"] = bool(api_key)
            if "enabled" in payload:
                provider["enabled"] = bool(payload.get("enabled"))
                self._ensure_active_model_locked()
            self._save()
            return self._copy(provider)

    def delete_provider(self, provider_id: str) -> dict[str, Any]:
        with self._lock:
            provider = self._find_provider_locked(provider_id)
            self.providers = [item for item in self.providers if item.get("id") != provider["id"]]
            self.models = [item for item in self.models if item.get("provider") != provider["id"]]
            self._ensure_active_model_locked()
            self._save()
            return self._copy(provider)

    def add_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            model = self._normalize_model(payload)
            if not model["model_name"]:
                raise ValueError("model_name must not be empty")
            if any(item.get("model_name") == model["model_name"] for item in self.models):
                raise SettingsConflictError(f"Model already exists: {model['model_name']}")
            provider_id = str(model.get("provider") or _provider_from_model(model["model_name"]))
            if any(item.get("id") == provider_id for item in self.providers):
                model["provider"] = provider_id
            else:
                self._ensure_provider(provider_id)
                model["provider"] = provider_id
            self.models.append(model)
            if not self.active_model:
                self.active_model = model["model_name"]
            self._save()
            return self._copy(model)

    def update_model(self, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            model = self._find_model_locked(model_name)
            if "name" in payload and str(payload.get("name") or "").strip():
                model["name"] = str(payload["name"]).strip()
            if "model_class" in payload:
                model["model_class"] = str(payload.get("model_class") or "").strip()
            if "provider" in payload and str(payload.get("provider") or "").strip():
                provider_id = str(payload["provider"]).strip()
                self._ensure_provider(provider_id)
                model["provider"] = provider_id
            if "enabled" in payload:
                model["enabled"] = bool(payload.get("enabled"))
                self._ensure_active_model_locked()
            self._save()
            return self._copy(model)

    def delete_model(self, model_name: str) -> dict[str, Any]:
        with self._lock:
            model = self._find_model_locked(model_name)
            self.models = [item for item in self.models if item.get("model_name") != model["model_name"]]
            self._ensure_active_model_locked()
            self._save()
            return self._copy(model)

    def set_model_enabled(self, model_name: str, enabled: bool) -> dict[str, Any]:
        return self.update_model(model_name, {"enabled": bool(enabled)})

    def set_provider_enabled(self, provider_id: str, enabled: bool) -> dict[str, Any]:
        return self.update_provider(provider_id, {"enabled": bool(enabled)})

    def _ensure_active_model_locked(self) -> None:
        current = next((model for model in self.models if model.get("model_name") == self.active_model), None)
        if (
            current is not None
            and current.get("enabled", True)
            and _model_is_enabled_by_provider(current, self.providers)
        ):
            return
        enabled = self._enabled_models_locked()
        self.active_model = str(enabled[0]["model_name"]) if enabled else ""

    def set_active_model(self, model_name: str) -> dict[str, Any]:
        with self._lock:
            model = self._find_model_locked(model_name)
            if not model.get("enabled", True):
                raise ValueError(f"Model is disabled: {model['model_name']}")
            if not _model_is_enabled_by_provider(model, self.providers):
                raise ValueError(f"Provider is disabled for model: {model['model_name']}")
            self.active_model = str(model["model_name"])
            self._save()
            return self._copy(model)

    def apply_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply a bulk settings document sent by a settings UI."""
        with self._lock:
            if isinstance(payload.get("providers"), list):
                providers: list[dict[str, Any]] = []
                seen: set[str] = set()
                for item in payload["providers"]:
                    if not isinstance(item, dict):
                        continue
                    provider = self._normalize_provider(item)
                    if provider["id"] not in seen:
                        seen.add(provider["id"])
                        providers.append(provider)
                self.providers = providers
            if isinstance(payload.get("models"), list):
                models: list[dict[str, Any]] = []
                seen_models: set[str] = set()
                for item in payload["models"]:
                    if not isinstance(item, dict):
                        continue
                    model = self._normalize_model(item)
                    if not model["model_name"] or model["model_name"] in seen_models:
                        continue
                    provider_id = str(model.get("provider") or _provider_from_model(model["model_name"]))
                    if provider_id and not any(provider.get("id") == provider_id for provider in self.providers):
                        self._ensure_provider(provider_id)
                    model["provider"] = provider_id
                    seen_models.add(model["model_name"])
                    models.append(model)
                self.models = models
            active_model = payload.get("active_model") or payload.get("model_name")
            if isinstance(active_model, str) and active_model.strip():
                self.active_model = active_model.strip()
            self._ensure_active_model_locked()
            self._save()
            return self.snapshot()


class ConversationState(str, Enum):
    """Canonical states shown for a GUI conversation."""

    RUNNING = "running"
    SUBMITTED = "submitted"
    STOPPED = "stopped"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    # Semantic aliases for callers that use completed/error terminology.
    COMPLETED = "submitted"
    ERROR = "failed"


def conversation_state(status: str, *, done: bool) -> ConversationState:
    """Normalize an agent exit status into one of the GUI conversation states."""
    normalized = str(status or "").strip().lower()
    if not done:
        return ConversationState.RUNNING
    if normalized in {"submitted", "complete", "completed", "success", "succeeded"}:
        return ConversationState.SUBMITTED
    if normalized in {"stopped", "stop", "cancelled", "canceled"}:
        return ConversationState.STOPPED
    if normalized in {"interrupted", "interrupt"}:
        return ConversationState.INTERRUPTED
    return ConversationState.FAILED


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mini-SWE-agent</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #1d2433;
      --muted: #667085;
      --border: #d9dee8;
      --accent: #315efb;
      --accent-soft: #e8eeff;
      --ok: #147a4b;
      --warn: #9a6700;
      --err: #b42318;
      --code-bg: #111827;
      --code-text: #e5e7eb;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f141d;
        --panel: #171e29;
        --text: #e7ebf2;
        --muted: #9aa6b8;
        --border: #2b3647;
        --accent: #7c9cff;
        --accent-soft: #1d2b4d;
        --ok: #43c88a;
        --warn: #e0b44a;
        --err: #ff8b80;
        --code-bg: #080b10;
        --code-text: #e5e7eb;
      }
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .status {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 4px 10px;
      border-radius: 999px;
      background: transparent;
      color: var(--accent);
      font-weight: 650;
    }
    .status.error, .status.failed { color: var(--err); }
    .status.submitted { color: var(--ok); }
    .status.stopped { color: var(--muted); }
    .status.interrupted { color: var(--warn); }

    .conversation-bar {
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
    }
    .conversation-context {
      min-width: 0;
      display: flex;
      align-items: baseline;
      gap: 10px;
    }
    .conversation-label {
      flex: 0 1 auto;
      min-width: 0;
      overflow: hidden;
      color: var(--text);
      font-size: 13px;
      font-weight: 650;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .conversation-workspace {
      min-width: 0;
      overflow: hidden;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .conversation-id {
      min-width: 0;
      overflow: hidden;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
      user-select: all;
      cursor: pointer;
    }
    .conversation-id:hover { color: var(--accent); }
    .conversation-id-row { display: flex; min-width: 0; margin-top: 2px; }
    .conversation-id-row .conversation-id { max-width: 100%; }
    .conversation-actions { display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 8px; }
    .conversation-actions button { padding: 5px 9px; border-radius: 7px; font-size: 12px; white-space: nowrap; }
    .conversation-bar .status { flex: 0 0 auto; }
    main {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 340px;
      grid-template-rows: minmax(0, 1fr);
      gap: 14px;
      width: 100%;
      max-width: none;
      height: 100dvh;
      margin: 0;
      padding: 14px;
    }
    section { display: flex; flex-direction: column; min-height: 0; gap: 12px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
    }
    .conversations {
      padding: 12px;
      gap: 8px;
      overflow: hidden;
    }
    .conversations-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 2px 2px 8px;
    }
    .conversations-head h2 {
      margin: 0;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    #new-conversation {
      padding: 5px 9px;
      border-radius: 7px;
      font-size: 12px;
      white-space: nowrap;
    }
    .conversation-list {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 5px;
      padding-right: 2px;
    }
    .conversation-item {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid transparent;
      border-radius: 9px;
      background: transparent;
      color: var(--text);
      text-align: left;
      font-weight: 500;
    }
    .conversation-item:hover {
      background: color-mix(in srgb, var(--accent-soft) 60%, transparent);
    }
    .conversation-item.active {
      background: var(--accent-soft);
      border-color: color-mix(in srgb, var(--accent) 35%, var(--border));
    }
    .conversation-title-row {
      display: flex;
      align-items: baseline;
      gap: 8px;
      min-width: 0;
    }
    .conversation-title {
      flex: 1 1 auto;
      min-width: 0;
      display: block;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      font-size: 13px;
      font-weight: 650;
    }
    .conversation-title-row .conversation-workspace { flex: 0 1 auto; max-width: 48%; }
    .conversation-meta {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 400;
    }
    .conversation-status.running { color: var(--accent); }
    .conversation-status.submitted { color: var(--ok); }
    .conversation-status.stopped { color: var(--muted); }
    .conversation-status.failed { color: var(--err); }
    .conversation-status.interrupted { color: var(--warn); }
    .conversation-empty {
      padding: 24px 10px;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }
    .composer { flex: 0 0 auto; padding: 12px; }
    .composer-attachments { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
    .composer-attachment {
      position: relative; width: 72px; height: 72px; border-radius: 8px;
      overflow: hidden; border: 1px solid var(--border); background: var(--panel);
    }
    .composer-attachment img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .composer-attachment button {
      position: absolute; top: 3px; right: 3px; width: 20px; height: 20px;
      padding: 0; border-radius: 50%; font-size: 13px; line-height: 1;
      background: rgba(0, 0, 0, 0.68); color: white;
    }
    .composer-input-row { display: flex; align-items: flex-end; gap: 8px; }
    .composer-icon-button {
      flex: 0 0 auto;
      width: 42px;
      height: 42px;
      padding: 0;
      border-radius: 10px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      font-weight: 400;
      line-height: 1;
    }
    .composer.dragging textarea { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    /* Compact single-line task input; it grows only while the user types. */
    textarea {
      flex: 1 1 auto;
      width: 100%;
      min-width: 0;
      min-height: 42px;
      height: 42px;
      max-height: 168px;
      resize: none;
      overflow-y: auto;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: transparent;
      color: var(--text);
      font: inherit;
      line-height: 1.4;
      outline: none;
    }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .composer-action { flex: 0 0 auto; align-self: flex-end; }
    .composer-action button { min-width: 76px; height: 42px; padding: 0 16px; border-radius: 10px; }
    .composer-action button[hidden] { display: none; }
    .composer-action .danger { background: var(--err); color: white; }
    .composer-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 8px; min-height: 16px; }
    .composer-actions .conversation-workspace { flex: 1 1 auto; min-width: 0; }
    .button-group { display: flex; align-items: center; gap: 8px; }
    button {
      appearance: none; border: 0; border-radius: 9px; padding: 9px 15px; font: inherit;
      font-weight: 700; cursor: pointer; background: var(--accent); color: white;
    }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .feed-wrap { position: relative; flex: 1; min-height: 0; display: flex; }
    .feed {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 9px;
      padding: 6px 4px 28px;
      scroll-behavior: smooth;
    }
    .scroll-actions {
      position: absolute;
      right: 14px;
      bottom: 14px;
      z-index: 4;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .scroll-actions button {
      padding: 7px 11px;
      font-size: 12px;
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      box-shadow: 0 2px 8px rgb(0 0 0 / 0.12);
    }
    .scroll-actions button[hidden] { display: none; }
    .empty {
      padding: 42px 20px; text-align: center; color: var(--muted);
    }
    .run-marker {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 3px 4px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .run-marker::before {
      content: "";
      flex: 0 0 auto;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--muted);
    }
    .run-marker::after {
      content: "";
      flex: 1 1 auto;
      height: 1px;
      background: linear-gradient(90deg, var(--border), transparent);
    }
    .run-marker.start::before { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .run-marker.end::before { background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 16%, transparent); }
    .run-marker-meta {
      margin-left: auto;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0;
      text-transform: none;
    }
    .card {
      position: relative;
      flex: 0 0 auto;
      max-width: min(94%, 980px);
      padding: 11px 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--panel);
      overflow-wrap: anywhere;
    }
    .card-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 7px; }
    .role { font-size: 11px; font-weight: 800; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); }
    .card.user {
      align-self: flex-end;
      border-color: color-mix(in srgb, var(--accent) 28%, var(--border));
      border-bottom-right-radius: 4px;
      background: color-mix(in srgb, var(--accent-soft) 52%, var(--panel));
    }
    /* Assistant messages read like a transcript: no bubble or per-step chrome. */
    .card.assistant {
      align-self: stretch;
      width: 100%;
      max-width: 100%;
      padding: 2px 6px;
      border: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
    }
    .card.assistant .card-head { display: none; }
    .card.assistant .content { line-height: 1.65; }
    .card.error { align-self: stretch; border-left: 3px solid var(--err); }
    .card[hidden] { display: none !important; }
    .content { white-space: pre-wrap; overflow-wrap: anywhere; }
    .content.markdown { white-space: normal; }
    .content.markdown p { margin: 0 0 0.8em; }
    .content.markdown > :first-child { margin-top: 0; }
    .content.markdown > :last-child { margin-bottom: 0; }
    .content.markdown h1,
    .content.markdown h2,
    .content.markdown h3,
    .content.markdown h4,
    .content.markdown h5,
    .content.markdown h6 { margin: 1.1em 0 0.45em; line-height: 1.3; }
    .content.markdown h1 { font-size: 1.4em; }
    .content.markdown h2 { font-size: 1.3em; }
    .content.markdown h3 { font-size: 1.18em; }
    .content.markdown h4,
    .content.markdown h5,
    .content.markdown h6 { font-size: 1.05em; }
    .content.markdown ul,
    .content.markdown ol { margin: 0.45em 0 0.85em; padding-left: 1.7em; }
    .content.markdown li + li { margin-top: 0.25em; }
    .content.markdown blockquote { margin: 0.7em 0; padding: 0.1em 0 0.1em 0.9em; border-left: 3px solid var(--border); color: var(--muted); }
    .content.markdown code { padding: 0.12em 0.35em; border-radius: 4px; background: var(--bg); font-family: var(--mono); font-size: 0.9em; }
    .content.markdown pre { max-width: 100%; margin: 0.65em 0 0.9em; padding: 11px 13px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; background: var(--code-bg); color: var(--code-text); white-space: pre; }
    .content.markdown pre code { padding: 0; background: transparent; overflow-wrap: normal; }
    .content.markdown table { width: 100%; margin: 0.65em 0 0.9em; border-collapse: collapse; }
    .content.markdown th,
    .content.markdown td { padding: 0.45em 0.65em; border: 1px solid var(--border); text-align: left; vertical-align: top; }
    .content.markdown th { background: var(--bg); font-weight: 600; }
    .content.markdown a { color: var(--accent); text-decoration: underline; text-underline-offset: 2px; }
    .content.markdown hr { margin: 1em 0; border: 0; border-top: 1px solid var(--border); }
    .card-images { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 9px; }
    .card-images a { display: block; line-height: 0; }
    .card-images a { cursor: zoom-in; }
    body.preview-open { overflow: hidden; }
    .image-preview { position: fixed; inset: 0; z-index: 10020; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; padding: 48px 24px 24px; background: rgb(0 0 0 / 82%); }
    .image-preview[hidden] { display: none; }
    .image-preview img { max-width: 94vw; max-height: 84vh; object-fit: contain; border-radius: 8px; box-shadow: 0 12px 50px rgb(0 0 0 / 45%); }
    .image-preview-close { position: absolute; top: 12px; right: 16px; width: 36px; height: 36px; padding: 0; border: 0; border-radius: 50%; background: rgb(255 255 255 / 14%); color: white; font-size: 26px; cursor: pointer; }
    .image-preview a { color: white; text-decoration: underline; }
    .card-images img {
      max-width: 220px; max-height: 220px; border-radius: 8px;
      border: 1px solid var(--border); object-fit: contain; background: var(--bg);
    }
    .reasoning { margin-top: 4px; color: var(--muted); font-size: 11px; }
    .reasoning summary {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: var(--muted);
      cursor: pointer;
      list-style: none;
      user-select: none;
    }
    .reasoning summary::-webkit-details-marker { display: none; }
    .reasoning summary::before {
      content: "\25B8";
      font-size: 9px;
      transition: transform 120ms ease;
    }
    .reasoning[open] summary::before { transform: rotate(90deg); }
    .reasoning-body { margin-top: 6px; padding-left: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }

    /* A tool call is one quiet line in the transcript.  Its command, exit
       status, duration and output belong in the expandable Activity view. */
    .tool-call {
      display: flex;
      align-items: baseline;
      gap: 7px;
      flex: 0 0 auto;
      width: 100%;
      max-width: 100%;
      margin: -2px 0 2px;
      padding: 5px 7px;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: var(--text);
      font: 500 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-align: left;
      cursor: pointer;
    }
    .tool-call:hover {
      border-color: var(--border);
      background: color-mix(in srgb, var(--panel) 84%, var(--bg));
    }
    .tool-call:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
    .tool-call-icon { flex: 0 0 auto; color: var(--muted); font-size: 9px; }
    .tool-call-prompt { flex: 0 0 auto; color: var(--ok); font-weight: 800; }
    .tool-call-command {
      flex: 1 1 auto;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .tool-call-status {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
      white-space: nowrap;
    }
    .tool-call-status.running { color: var(--accent); }
    .tool-call-status.ok { color: var(--ok); }
    .tool-call-status.error { color: var(--err); }
    pre {
      margin: 8px 0 0;
      padding: 10px 12px;
      overflow: auto;
      border-radius: 8px;
      background: var(--code-bg);
      color: var(--code-text);
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .meta { color: var(--muted); font-size: 11px; }
    aside { min-height: 0; overflow: hidden; display: flex; flex-direction: column; gap: 12px; }
    aside.conversations, aside.details { position: sticky; top: 14px; align-self: start; height: calc(100dvh - 28px); }
    aside.details { overflow: hidden; }
    .details-settings { flex: 0 0 auto; display: flex; justify-content: flex-end; gap: 8px; }
    .details-settings .tree-toggle { flex: 1 1 auto; min-width: 0; justify-content: space-between; }
    .stats { padding: 16px; flex: 0 0 auto; }
    .tree-count { color: var(--muted); font-size: 11px; }
    .main-tree-view { position: absolute; inset: 0; z-index: 6; overflow: auto; padding: 18px; background: var(--bg); }
    .main-tree-view[hidden] { display: none; }
    .main-tree-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 18px; }
    .main-conversation-tree { position: relative; min-width: 260px; }
    .main-tree-connectors { position: absolute; inset: 0; z-index: 0; overflow: visible; pointer-events: none; }
    .main-tree-edge { fill: none; stroke: var(--border); stroke-width: 1.5; stroke-linecap: round; }
    .main-tree-family { position: relative; z-index: 1; min-width: 260px; }
    .main-tree-node-row { position: relative; display: grid; grid-template-columns: 18px minmax(0, 1fr); gap: 10px; align-items: start; }
    .main-tree-dot { position: relative; z-index: 2; box-sizing: border-box; width: 12px; height: 12px; margin: 16px 0 0 3px; border: 2px solid var(--muted); border-radius: 50%; background: var(--bg); }
    .main-tree-dot.active { border-color: var(--accent); background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .main-tree-node { display: block; width: min(100%, 760px); margin: 0; padding: 8px 10px; border: 0; border-radius: 7px; background: transparent; color: var(--text); text-align: left; cursor: pointer; }
    .main-tree-node:hover, .main-tree-node.active { background: var(--accent-soft); }
    .main-tree-node.active { color: var(--accent); }
    .main-tree-node-task { display: block; overflow-wrap: anywhere; white-space: pre-wrap; }
    .main-tree-node-meta { display: block; margin-top: 3px; color: var(--muted); font-size: 11px; }
    .main-tree-children { position: relative; }
    .main-tree-children.branched { padding-left: 32px; }
    .card.user.tree-highlight { box-shadow: 0 0 0 3px var(--accent-soft), 0 0 0 1px var(--accent); }

    .context-menu { position: fixed; z-index: 10000; min-width: 190px; padding: 6px; border: 1px solid var(--border); border-radius: 9px; background: var(--panel); box-shadow: 0 12px 36px rgb(0 0 0 / 22%); }
    .context-menu[hidden] { display: none; }
    .context-menu button { display: block; width: 100%; padding: 8px 10px; border: 0; border-radius: 6px; background: transparent; color: var(--text); text-align: left; cursor: pointer; }
    .context-menu button:hover:not(:disabled) { background: var(--accent-soft); }
    .context-menu button.danger { color: var(--err); }
    .context-menu button:disabled { color: var(--muted); cursor: not-allowed; }
    .conversation-item.pinned .conversation-title::before { content: "★ "; color: var(--accent); }
    .card.user.editing { align-self: stretch; width: 100%; max-width: 100%; padding: 8px; }
    .user-editor { display: grid; gap: 6px; width: 100%; min-width: 0; }
    .user-editor textarea { width: 100%; min-height: 64px; max-height: 320px; resize: vertical; padding: 9px; border: 1px solid var(--accent); border-radius: 8px; background: var(--bg); color: var(--text); font: inherit; }
    .user-editor-actions { display: flex; justify-content: flex-end; gap: 7px; }
    .user-editor-actions button { padding: 6px 10px; }
    .stats h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 12px; }
    .stat { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--border); }
    .stat:last-child { border-bottom: 0; }
    .stat > span:first-child { flex: 0 0 auto; }
    .stat > span:last-child { min-width: 0; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; font-weight: 600; }

    /* Activity: a compact, turn-aware ledger modelled after the DSH trajectory view. */
    .activity {
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      color: var(--text);
      font-size: 12px;
    }
    .activity-header {
      flex: 0 0 auto;
      display: flex;
      flex-direction: column;
      gap: 9px;
      padding: 13px 14px 10px;
      border-bottom: 1px solid var(--border);
    }
    .activity-heading {
      --status-color: var(--accent);
      position: relative;
      isolation: isolate;
      overflow: hidden;
      padding: 4px 8px;
      border-radius: 12px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      align-items: center;
      gap: 10px;
    }
    .activity-heading::before,
    .activity-heading::after {
      content: "";
      position: absolute;
      pointer-events: none;
      -webkit-mask-image: linear-gradient(180deg, #000 0%, transparent 100%);
      mask-image: linear-gradient(180deg, #000 0%, transparent 100%);
    }
    .activity-heading::before {
      inset: 0;
      z-index: -2;
      border-radius: inherit;
      background:
        radial-gradient(ellipse 38% 100% at 18% 30%, color-mix(in srgb, var(--status-color) 65%, transparent), transparent 72%),
        radial-gradient(ellipse 42% 100% at 72% 65%, color-mix(in srgb, var(--accent) 45%, transparent), transparent 75%),
        linear-gradient(90deg, transparent, color-mix(in srgb, var(--status-color) 30%, transparent) 25%, color-mix(in srgb, var(--status-color) 38%, transparent) 60%, transparent);
      background-size: 140% 100%, 160% 100%, 220% 100%;
    }
    .activity-heading::after {
      inset: 0;
      z-index: -1;
      border-radius: inherit;
      background: linear-gradient(105deg, transparent 25%, rgb(255 255 255 / 18%) 42%, transparent 58%);
      background-size: 220% 100%;
    }
    .activity-heading[data-status="running"]::before { animation: silk-flow 5s ease-in-out infinite alternate; }
    .activity-heading[data-status="running"]::after { animation: silk-sheen 5s ease-in-out infinite alternate; }
    .activity-heading[data-status="error"], .activity-heading[data-status="failed"] { --status-color: var(--err); }
    .activity-heading[data-status="submitted"] { --status-color: var(--ok); }
    .activity-heading[data-status="stopped"] { --status-color: var(--muted); }
    .activity-heading[data-status="interrupted"] { --status-color: var(--warn); }
    @keyframes silk-flow { from { background-position: 0% 50%, 20% 50%, 0% 50%; } to { background-position: 100% 50%, 80% 50%, 100% 50%; } }
    @keyframes silk-sheen { from { background-position: 0% 50%; } to { background-position: 100% 50%; } }
    .activity-heading > .status { justify-self: center; white-space: nowrap; }
    .activity-heading h2 {
      margin: 0;
      color: var(--text);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .activity-heading-actions { flex: 0 0 auto; display: flex; align-items: center; justify-self: end; gap: 6px; }
    #activity-expand {
      flex: 0 0 auto;
      padding: 4px 8px;
      border-radius: 7px;
      font-size: 10px;
      line-height: 1.3;
    }
    #activity-expand[aria-expanded="true"] {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .activity-summary {
      display: flex;
      align-items: center;
      gap: 6px;
      min-height: 19px;
      color: var(--muted);
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
      letter-spacing: 0.01em;
    }
    .activity-summary.running { color: var(--accent); }
    .activity-summary.done { color: var(--ok); }
    .activity-summary.done::before { background: var(--ok); }
    .activity-summary.error { color: var(--err); }
    .activity-summary.error::before { background: var(--err); }
    .activity-overview {
      display: flex;
      align-items: stretch;
      gap: 2px;
      width: 100%;
      height: 10px;
      min-height: 10px;
      overflow: hidden;
    }
    .activity-overview-track { display: flex; align-items: stretch; gap: 2px; height: 100%; flex-shrink: 0; will-change: transform; }
    .activity-segment {
      flex: 0 0 0;
      min-width: 0;
      border-radius: 2px;
      background: var(--accent);
    }
    .activity-segment.kind-thinking { background: var(--accent); }
    .activity-segment.kind-reply { background: var(--ok); }
    .activity-segment.kind-tool { background: var(--warn); }
    .activity-segment.kind-done { background: var(--ok); }
    .activity-segment.kind-error { background: var(--err); }
    .activity-segment.kind-running { background: var(--border); margin-left: auto; }
    .activity-overview-empty {
      flex: 1;
      color: var(--muted);
      font-size: 9px;
      line-height: 10px;
    }

    .activity-scroll {
      position: relative;
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      /* Keep the newest entries at the top.  When an older entry does not
         fit, the unused space belongs below the ledger rather than above it. */
      justify-content: flex-start;
      overflow: hidden;
      padding: 10px 12px 12px;
    }
    /* A continuous timeline spine keeps the guide line visible through the
       empty space below the last visible activity. */
    .activity-scroll.has-activity::after {
      content: "";
      position: absolute;
      left: 22px;
      top: 32px;
      bottom: 12px;
      width: 1px;
      background: var(--border);
      pointer-events: none;
      z-index: 0;
    }
    /* When older activities are clipped, lead into the first visible marker
       so the guide line still communicates that the timeline continues. */
    .activity-scroll.has-clipped-history::before {
      content: "";
      position: absolute;
      left: 22px;
      top: 10px;
      width: 1px;
      height: 22px;
      background: var(--border);
      pointer-events: none;
      z-index: 0;
    }
    #activity-log {
      position: relative;
      z-index: 1;
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-direction: column;
      flex: 0 0 auto;
      min-height: 0;
    }
    .activity-entry[hidden] { display: none; }
    .activity-empty {
      padding: 18px 8px 12px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
      text-align: center;
    }
    .activity-empty[hidden] { display: none; }
    .activity-entry {
      position: relative;
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr);
      gap: 8px;
      padding-bottom: 12px;
      min-width: 0;
    }
    .activity-entry:not(:last-child)::before {
      content: "";
      position: absolute;
      left: 10px;
      top: 22px;
      bottom: 0;
      width: 1px;
      background: var(--border);
    }
    .activity-marker {
      position: relative;
      z-index: 1;
      display: grid;
      place-items: center;
      align-self: start;
      width: 22px;
      height: 22px;
      border-radius: 7px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 10px;
      font-weight: 800;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }
    .activity-entry.kind-tool .activity-marker { background: color-mix(in srgb, var(--warn) 13%, transparent); color: var(--warn); }
    .activity-entry.kind-done .activity-marker { background: color-mix(in srgb, var(--ok) 13%, transparent); color: var(--ok); }
    .activity-entry.kind-error .activity-marker { background: color-mix(in srgb, var(--err) 13%, transparent); color: var(--err); }
    .activity-entry.kind-info .activity-marker { background: color-mix(in srgb, var(--muted) 13%, transparent); color: var(--muted); }
    .activity-card { min-width: 0; padding-top: 1px; }
    .activity-row-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; min-width: 0; }
    .activity-label {
      min-width: 0;
      overflow: hidden;
      color: var(--text);
      font-size: 11px;
      font-weight: 700;
      line-height: 1.35;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .activity-time {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 9px;
      font-variant-numeric: tabular-nums;
      line-height: 1.35;
    }
    .activity-duration {
      display: none;
      flex: 0 0 auto;
      padding: 1px 5px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 9px;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
      line-height: 1.35;
      white-space: nowrap;
    }
    .activity-copy {
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .activity-copy code {
      display: block;
      overflow: hidden;
      color: var(--text);
      font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .activity-command-full { display: none; }
    /* Keep tool output out of the compact rail; Activity Expand is the detail view. */
    .activity-output {
      display: none;
      margin-top: 7px;
      padding: 7px 9px;
      overflow: hidden;
      border-radius: 6px;
      background: color-mix(in srgb, var(--code-bg) 6%, var(--bg));
      color: var(--muted);
      font: 10px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .activity-output-label {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font: 800 9px/1.2 ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .activity-output-text { white-space: pre-wrap; overflow-wrap: anywhere; }
    .activity-entry.activity-highlight .activity-card {
      border-radius: 8px;
      animation: activity-edge-flash 1100ms ease-out 1 both;
    }
    @keyframes activity-edge-flash {
      0% { box-shadow: 0 0 0 0 transparent; background: transparent; }
      18% { box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 72%, transparent); background: var(--accent-soft); }
      100% { box-shadow: 0 0 0 0 transparent; background: transparent; }
    }
    .activity-meta-row { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-top: 5px; }
    .activity-status {
      display: inline-flex;
      align-items: center;
      padding: 1px 6px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--muted) 10%, transparent);
      color: var(--muted);
      font-size: 9px;
      font-weight: 750;
      line-height: 1.45;
      text-transform: lowercase;
    }
    .activity-status[hidden], .activity-meta[hidden] { display: none; }
    .activity-status.running { background: var(--accent-soft); color: var(--accent); }
    .activity-status.ok { background: color-mix(in srgb, var(--ok) 12%, transparent); color: var(--ok); }
    .activity-status.error { background: color-mix(in srgb, var(--err) 12%, transparent); color: var(--err); }
    .activity-meta { color: var(--muted); font-size: 9px; font-variant-numeric: tabular-nums; line-height: 1.45; }
    .activity-entry.kind-error .activity-label { color: var(--err); }
    .activity-entry.kind-done .activity-label { color: var(--ok); }
    /* Expanded activity page: the right rail takes over the conversation area. */
    body.activity-expanded main {
      grid-template-columns: 280px minmax(0, 1fr);
    }
    body.activity-expanded main > section { display: none; }
    body.activity-expanded aside.details {
      display: flex;
      min-width: 0;
      overflow: hidden;
    }
    body.activity-expanded aside.details .stats, body.activity-expanded .details-settings { display: none; }
    body.activity-expanded .activity {
      flex: 1 1 auto;
      min-height: 0;
      overflow: hidden;
    }
    body.activity-expanded .activity-header {
      padding: 16px 20px 12px;
    }
    body.activity-expanded .activity-scroll {
      justify-content: flex-start;
      overflow-x: hidden;
      overflow-y: auto;
      padding: 16px 20px 20px;
      scrollbar-gutter: stable;
    }
    body.activity-expanded #activity-log {
      /* Let the log grow with its content so the scroll container can reach it. */
      flex: 0 0 auto;
      justify-content: flex-start;
      gap: 8px;
      min-height: auto;
      width: 100%;
    }
    body.activity-expanded .activity-entry {
      display: grid;
      grid-template-columns: 26px minmax(0, 1fr);
      flex: 0 0 auto;
      align-items: start;
      gap: 10px;
      width: var(--activity-width, 100%);
      max-width: 100%;
      min-width: 0;
      min-height: 0;
      margin-left: var(--activity-offset, 0px);
      padding: 10px 12px;
      overflow: visible;
      border: 1px solid color-mix(in srgb, var(--accent) 24%, var(--border));
      border-radius: 10px;
      background: color-mix(in srgb, var(--accent-soft) 58%, var(--panel));
      box-shadow: 0 1px 2px rgb(0 0 0 / 0.035);
    }
    body.activity-expanded .activity-entry::before { display: none; }
    body.activity-expanded .activity-scroll::before,
    body.activity-expanded .activity-scroll::after { display: none; }
    body.activity-expanded .activity-entry.kind-tool {
      border-color: color-mix(in srgb, var(--warn) 28%, var(--border));
      background: color-mix(in srgb, var(--warn) 8%, var(--panel));
    }
    body.activity-expanded .activity-entry.kind-done {
      border-color: color-mix(in srgb, var(--ok) 30%, var(--border));
      background: color-mix(in srgb, var(--ok) 8%, var(--panel));
    }
    body.activity-expanded .activity-entry.kind-error {
      border-color: color-mix(in srgb, var(--err) 30%, var(--border));
      background: color-mix(in srgb, var(--err) 8%, var(--panel));
    }
    body.activity-expanded .activity-marker {
      width: 24px;
      height: 24px;
      border-radius: 8px;
      font-size: 10px;
    }
    body.activity-expanded .activity-card {
      min-width: 0;
      padding-top: 1px;
    }
    body.activity-expanded .activity-row-head {
      align-items: baseline;
      gap: 8px;
      flex-wrap: wrap;
    }
    body.activity-expanded .activity-label {
      flex: 1 1 auto;
      font-size: 12px;
      white-space: normal;
    }
    body.activity-expanded .activity-time { font-size: 10px; }
    body.activity-expanded .activity-duration {
      display: inline-flex;
      align-items: center;
      font-size: 10px;
    }
    body.activity-expanded .activity-copy {
      margin-top: 5px;
      font-size: 12px;
    }
    body.activity-expanded .activity-copy code {
      overflow: visible;
      white-space: pre-wrap;
    }
    body.activity-expanded .activity-command-preview { display: none; }
    body.activity-expanded .activity-command-full { display: block; }
    body.activity-expanded .activity-output {
      display: block;
      max-height: 18rem;
      overflow: auto;
      font-size: 11px;
    }
    body.activity-expanded .activity-meta-row { margin-top: 7px; }
    #activity-open { display: none; }

    @media (max-width: 1100px) {
      main { grid-template-columns: 220px minmax(0, 1fr); }
      aside.details { display: none; }
      #activity-open { display: inline-flex; }
      body.activity-expanded main { grid-template-columns: 220px minmax(0, 1fr); }
      body.activity-expanded aside.details { display: flex; }
      body.activity-expanded #activity-open { display: none; }
    }
    @media (max-width: 700px) {
      main { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
      aside.conversations { position: relative; top: 0; height: auto; max-height: 180px; }
      body.activity-expanded main { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr); }
      body.activity-expanded aside.conversations { display: none; }
    }
    .settings-button {
      display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
      border: 1px solid var(--border); background: var(--panel); color: var(--text);
      padding: 7px 11px; border-radius: 9px; cursor: pointer; font-weight: 600;
    }
    .settings-button:hover { border-color: var(--accent); color: var(--accent); }
    .settings-button .gear { font-size: 15px; line-height: 1; }
    .visually-hidden {
      position: absolute !important; width: 1px !important; height: 1px !important;
      padding: 0 !important; margin: -1px !important; overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important; white-space: nowrap !important; border: 0 !important;
    }
    .model-switcher-control {
      display: inline-flex; align-items: center; gap: 0; min-width: 0;
      color: var(--muted); font-size: 12px; font-weight: 650;
    }
    #model-switcher {
      max-width: min(420px, 42vw); min-width: 130px;
      border: 1px solid var(--border); border-radius: 8px;
      background: var(--panel); color: var(--text);
      padding: 5px 26px 5px 9px; font: inherit; font-size: 12px;
      text-overflow: ellipsis; cursor: pointer;
    }
    #model-switcher:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
    .modal-backdrop {
      position: fixed; inset: 0; z-index: 100;
      display: grid; place-items: center; padding: 20px;
      background: rgba(10, 15, 24, 0.48); backdrop-filter: blur(3px);
    }
    .modal-backdrop[hidden] { display: none; }
    .settings-modal {
      width: min(880px, 100%); max-height: min(86vh, 780px);
      display: flex; flex-direction: column; overflow: hidden;
      border: 1px solid var(--border); border-radius: 16px;
      background: var(--panel); color: var(--text);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }
    .settings-head {
      display: flex; align-items: center; justify-content: space-between; gap: 14px;
      padding: 16px 20px; border-bottom: 1px solid var(--border);
    }
    .settings-head h2 { margin: 0; font-size: 18px; letter-spacing: -0.01em; }
    .settings-close {
      border: 0; background: transparent; color: var(--muted); cursor: pointer;
      font-size: 24px; line-height: 1; padding: 2px 7px; border-radius: 8px;
    }
    .settings-close:hover { color: var(--text); background: var(--bg); }
    .settings-tabs {
      display: flex; gap: 4px; padding: 10px 16px 0; border-bottom: 1px solid var(--border);
      overflow-x: auto;
    }
    .settings-tab {
      border: 0; border-bottom: 2px solid transparent; border-radius: 7px 7px 0 0;
      background: transparent; color: var(--muted); cursor: pointer;
      padding: 9px 13px; font-size: 13px; font-weight: 650; white-space: nowrap;
    }
    .settings-tab:hover { color: var(--text); background: var(--bg); }
    .settings-tab.active { color: var(--accent); border-bottom-color: var(--accent); background: var(--accent-soft); }
    .settings-content { padding: 18px 20px 24px; overflow-y: auto; min-height: 280px; }
    .settings-panel[hidden] { display: none; }
    .settings-panel h3 { margin: 2px 0 10px; font-size: 14px; }
    .settings-panel p.hint { margin: 0 0 14px; color: var(--muted); font-size: 12px; }
    .settings-section-head {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      margin: 18px 0 9px;
    }
    .settings-section-head:first-child { margin-top: 0; }
    .settings-section-head h3 { margin: 0; }
    .settings-list { display: flex; flex-direction: column; gap: 8px; }
    .settings-list[hidden] { display: none; }
    .provider-group { display: flex; flex-direction: column; gap: 7px; }
    .provider-row { cursor: pointer; }
    .provider-row-main { min-width: 0; display: flex; flex-direction: column; gap: 3px; cursor: pointer; }
    .provider-row-main:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 7px; }
    .provider-models {
      display: flex; flex-direction: column; gap: 7px;
      margin: 0 0 4px 18px; padding: 8px 0 0 14px;
      border-left: 2px solid var(--border);
    }
    .provider-models[hidden] { display: none; }
    .provider-models-head {
      display: flex; align-items: center; justify-content: space-between; gap: 8px;
      color: var(--muted); font-size: 11px; font-weight: 650; padding-left: 2px;
    }
    .provider-models-head button { padding: 3px 7px; border-radius: 6px; font-size: 10px; }
    .provider-model-row { padding: 8px 10px; background: var(--bg); }
    .provider-models-empty {
      color: var(--muted); font-size: 11px; padding: 8px 10px;
      border: 1px dashed var(--border); border-radius: 8px;
    }
    .settings-empty {
      color: var(--muted); font-size: 12px; padding: 12px;
      border: 1px dashed var(--border); border-radius: 9px;
    }
    .settings-row {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px;
      background: color-mix(in srgb, var(--panel) 82%, var(--bg));
    }
    .settings-row-main { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
    .settings-row-title { font-weight: 650; font-size: 13px; overflow: hidden; text-overflow: ellipsis; }
    .settings-row-meta { color: var(--muted); font-size: 11px; overflow: hidden; text-overflow: ellipsis; }
    .settings-row-actions { display: flex; align-items: center; gap: 6px; flex: 0 0 auto; }
    .settings-row-actions button { padding: 4px 8px; border-radius: 7px; font-size: 11px; }
    .settings-badge {
      display: inline-flex; align-items: center; padding: 2px 7px; border-radius: 999px;
      background: var(--accent-soft); color: var(--accent); font-size: 10px; font-weight: 700;
    }
    .settings-badge.off { background: color-mix(in srgb, var(--muted) 14%, transparent); color: var(--muted); }
    .settings-form {
      margin-top: 10px; padding: 13px; border: 1px solid var(--border); border-radius: 11px;
      background: var(--bg); display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;
    }
    .settings-form[hidden] { display: none; }
    .settings-form .field { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
    .settings-form .field.full, .settings-form .field-help.full { grid-column: 1 / -1; }
    .settings-form label { color: var(--muted); font-size: 11px; font-weight: 650; }
    .settings-form .field-help {
      margin: 0; color: var(--muted); font-size: 11px; line-height: 1.45;
    }
    .settings-form .field-help code {
      font-size: 10px; padding: 1px 4px; border-radius: 4px;
      background: color-mix(in srgb, var(--muted) 13%, transparent);
    }
    .settings-form input, .settings-form select {
      width: 100%; min-width: 0; border: 1px solid var(--border); border-radius: 7px;
      background: var(--panel); color: var(--text); padding: 7px 9px; font: inherit; font-size: 12px;
    }
    .settings-form .form-actions { grid-column: 1 / -1; display: flex; justify-content: flex-end; gap: 8px; }
    .settings-toggle { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }
    .settings-note { color: var(--muted); font-size: 12px; line-height: 1.7; }
    @media (max-width: 720px) {
      .settings-form { grid-template-columns: 1fr; }
      #model-switcher { max-width: 38vw; }
      .settings-row { align-items: flex-start; }
      .settings-row-actions { flex-wrap: wrap; justify-content: flex-end; }
    }
  </style>
</head>
<body>

  <main>
    <aside class="panel conversations" aria-label="Conversations">
      <div class="conversations-head">
        <h2>Conversation List</h2>
        <button id="new-conversation" class="secondary" type="button">New</button>
      </div>
      <div class="conversation-list" id="conversation-list">
        <div class="conversation-empty" id="conversation-empty">No conversations yet.</div>
      </div>
    </aside>
    <section>
      <div class="panel conversation-bar" aria-label="Conversation toolbar">
        <div class="conversation-context">
          <span class="conversation-label" id="conversation-label">New conversation</span>
        </div>
        <div class="conversation-actions">
          <button id="activity-open" class="secondary" type="button" aria-controls="activity-panel" title="Open activity timeline">Activity</button>
          <label class="model-switcher-control" title="Switch the model for this conversation">
            <select id="model-switcher" aria-label="Switch model"></select>
          </label>
        </div>
      </div>
      <div class="feed-wrap">
        <div class="feed" id="feed">
          <div class="panel empty" id="empty">Responses stream here as the agent works.</div>
        </div>
        <div id="main-tree-view" class="main-tree-view" role="region" aria-label="Conversation branch tree" hidden>
          <div class="main-tree-header"><strong>Conversation tree</strong><button id="main-tree-close" class="secondary" type="button">Back to conversation</button></div>
          <div id="main-conversation-tree" class="main-conversation-tree"></div>
        </div>

        <div class="scroll-actions">
          <button id="scroll-top" class="secondary" type="button" title="Go to top" aria-label="Go to top">↑ Go to top</button>
          <button id="scroll-bottom" class="secondary" type="button" title="Go to end" aria-label="Go to end">Go to end ↓</button>
        </div>
      </div>
      <div class="panel composer">
        <div class="composer-attachments" id="composer-attachments" hidden></div>
        <div class="composer-input-row">
          <button id="attach-image" class="secondary composer-icon-button" type="button" title="Attach an image" aria-label="Attach an image">+</button>
          <textarea id="task" rows="1" placeholder="Describe the software engineering task. For example: find and fix the failing tests in this repository."></textarea>
          <div class="composer-action">
            <button id="send" type="button" title="Send message" aria-label="Send message">Send</button>
            <button id="stop" class="danger" type="button" title="Stop run" aria-label="Stop run" disabled hidden>Stop</button>
          </div>
          <input id="image-input" type="file" accept="image/*" multiple hidden>
        </div>
        <div class="composer-actions">
          <span class="conversation-workspace" id="conversation-workspace" title="Workspace directory"></span>
        </div>
      </div>
    </section>
    <aside class="details">
      <div class="details-settings">
        <button id="tree-toggle" class="settings-button tree-toggle" type="button" aria-expanded="false" aria-controls="main-tree-view" title="Open current conversation tree">
          <span>Tree</span><span id="tree-count" class="tree-count">0</span>
        </button>
        <button id="settings-button" class="settings-button" type="button" aria-haspopup="dialog" aria-controls="settings-modal" title="Open settings"><span class="gear" aria-hidden="true">&#9881;</span><span>Settings</span></button>
      </div>
      <div class="panel stats">
        <h2>Run</h2>
        <div class="stat"><span>Step</span><span id="stat-step">-</span></div>
        <div class="stat"><span>Tool calls</span><span id="stat-tools">-</span></div>
        <div class="stat"><span>Cost</span><span id="stat-cost">$0.00</span></div>
        <div class="stat"><span>Tokens</span><span id="stat-tokens">-</span></div>
        <div class="stat" title="Current context length after the latest model call"><span>Context</span><span id="stat-context">-</span></div>
        <div class="stat"><span>Input</span><span id="stat-input">-</span></div>
        <div class="stat"><span>Output</span><span id="stat-output">-</span></div>
      </div>

      <section class="panel activity" id="activity-panel" aria-label="Activity timeline">
        <div class="activity-header">
          <div class="activity-heading" data-status="idle">
            <div>
              <h2>Activity</h2>
            </div>
            <span id="status" class="status" role="status" aria-live="polite">idle</span>
            <div class="activity-heading-actions">
              <button id="activity-expand" class="secondary" type="button" aria-controls="activity-panel" aria-expanded="false" title="Expand activity view">Expand</button>
            </div>
          </div>
          <div class="activity-summary" id="activity-summary" role="status" aria-live="polite">No activity yet</div>
          <div class="activity-overview" id="activity-overview" aria-label="Activity overview"></div>

        </div>
        <div class="activity-scroll" id="activity-scroll">
          <ol id="activity-log">
            <li class="activity-empty" id="activity-empty">Start a run to see each model step, command, and outcome here in order.</li>
          </ol>
        </div>
      </section>
    </aside>
  </main>
  <div id="settings-modal" class="modal-backdrop" hidden>
    <div class="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div class="settings-head">
        <div>
          <h2 id="settings-title">Settings</h2>
          <div class="settings-row-meta">Configure model providers, enabled models, and GUI preferences.</div>
        </div>
        <button id="settings-close" class="settings-close" type="button" aria-label="Close settings">&times;</button>
      </div>
      <div class="settings-tabs" role="tablist" aria-label="Settings modules">
        <button id="settings-tab-general" class="settings-tab active" type="button" role="tab" aria-selected="true" aria-controls="settings-general" data-settings-tab="general">General</button>
        <button id="settings-tab-models" class="settings-tab" type="button" role="tab" aria-selected="false" aria-controls="settings-models" data-settings-tab="models">Models</button>
        <button id="settings-tab-conversation" class="settings-tab" type="button" role="tab" aria-selected="false" aria-controls="settings-conversation" data-settings-tab="conversation">Conversation</button>
        <button id="settings-tab-about" class="settings-tab" type="button" role="tab" aria-selected="false" aria-controls="settings-about" data-settings-tab="about">About</button>
      </div>
      <div class="settings-content">
        <section id="settings-general" class="settings-panel" role="tabpanel" aria-labelledby="settings-tab-general" data-settings-panel="general">
          <h3>General</h3>
          <p class="hint">These values come from the configuration used to launch <code>mini-gui</code>.</p>
          <div class="settings-list">
            <div class="settings-row">
              <div class="settings-row-main">
                <span class="settings-row-title">Active model</span>
                <span class="settings-row-meta">The model used for new conversations and follow-up turns.</span>
              </div>
              <span class="settings-badge" id="settings-active-model">Not configured</span>
            </div>
            <div class="settings-row">
              <div class="settings-row-main">
                <span class="settings-row-title">Workspace</span>
                <span class="settings-row-meta" id="settings-workspace">-</span>
              </div>
            </div>
            <div class="settings-row">
              <div class="settings-row-main">
                <span class="settings-row-title">Conversation database</span>
                <span class="settings-row-meta" id="settings-database">-</span>
              </div>
            </div>
          </div>
        </section>
        <section id="settings-models" class="settings-panel" role="tabpanel" aria-labelledby="settings-tab-models" data-settings-panel="models" hidden>
          <div class="settings-section-head">
            <div>
              <h3>Model providers</h3>
              <p class="hint">Providers describe where model requests are sent and which API key they use. Click a provider to show its models. Use <strong>Load models</strong> to query a saved provider and add the models it returns.</p>
            </div>
            <button id="add-provider" class="secondary" type="button">Add provider</button>
          </div>
          <div id="providers-list" class="settings-list" aria-label="Model providers"></div>
          <form id="provider-form" class="settings-form" hidden>
            <input id="provider-edit-id" type="hidden">
            <p class="field-help full" id="provider-api-key-location-help">
              <strong>Where should I put my API key?</strong> Paste the key itself into the
              <strong>API key (optional, stored locally)</strong> field below. The
              <strong>API key environment variable</strong> field is only the <em>name</em> of a
              shell variable (for example, <code>OPENAI_API_KEY</code>), never the key itself.
            </p>
            <div class="field">
              <label for="provider-name">Provider name</label>
              <input id="provider-name" name="name" type="text" placeholder="OpenAI" autocomplete="off" required>
            </div>
            <div class="field">
              <label for="provider-id">Provider ID</label>
              <input id="provider-id" name="id" type="text" placeholder="openai" autocomplete="off">
            </div>
            <div class="field">
              <label for="provider-api-key-env">API key environment variable (variable name)</label>
              <input id="provider-api-key-env" name="api_key_env" type="text" placeholder="OPENAI_API_KEY" autocomplete="off" aria-describedby="provider-api-key-env-help">
              <span class="field-help" id="provider-api-key-env-help">The name of the environment variable that contains your key, such as <code>OPENAI_API_KEY</code>. Do not paste the secret key here.</span>
            </div>
            <div class="field">
              <label for="provider-base-url">Base URL (optional)</label>
              <input id="provider-base-url" name="base_url" type="url" placeholder="https://api.example.com/v1" autocomplete="off">
            </div>
            <div class="field full">
              <label for="provider-api-key">API key (optional, stored locally)</label>
              <input id="provider-api-key" name="api_key" type="password" placeholder="Paste your API key here" autocomplete="new-password" aria-describedby="provider-api-key-help">
              <span class="field-help" id="provider-api-key-help">Paste your API key here to store it locally in the GUI settings. If you already exported it as the environment variable above, you can leave this blank.</span>
            </div>
            <div class="form-actions">
              <button id="provider-cancel" class="secondary" type="button">Cancel</button>
              <button id="provider-fetch-models" class="secondary" type="button"
                title="Query models from this provider and load them into the conversation model selector"
                aria-label="Query models and load them">Fetch &amp; load models</button>
              <button type="submit">Save provider</button>
            </div>
          </form>
          <div class="settings-section-head">
            <div>
              <h3>Models</h3>
              <p class="hint">Models are grouped under their provider. Expand a provider above to enable, edit, delete, or add models for it.</p>
            </div>
            <button id="add-model" class="secondary" type="button">Add model</button>
          </div>
          <div id="models-list" class="settings-list" aria-label="Models" hidden></div>
          <form id="model-form" class="settings-form" hidden>
            <input id="model-edit-id" type="hidden">
            <div class="field">
              <label for="model-name">Model name</label>
              <input id="model-name" name="model_name" type="text" placeholder="openai/gpt-5.4-mini" autocomplete="off" required>
            </div>
            <div class="field">
              <label for="model-provider">Provider</label>
              <select id="model-provider" name="provider"></select>
            </div>
            <div class="field">
              <label for="model-display-name">Display name (optional)</label>
              <input id="model-display-name" name="name" type="text" placeholder="GPT-5.4 mini" autocomplete="off">
            </div>
            <div class="field">
              <label for="model-class">Model class (optional)</label>
              <input id="model-class" name="model_class" type="text" placeholder="litellm" autocomplete="off">
            </div>
            <div class="form-actions">
              <button id="model-cancel" class="secondary" type="button">Cancel</button>
              <button type="submit">Save model</button>
            </div>
          </form>
        </section>
        <section id="settings-conversation" class="settings-panel" role="tabpanel" aria-labelledby="settings-tab-conversation" data-settings-panel="conversation" hidden>
          <h3>Conversation</h3>
          <p class="settings-note">The conversation header shows the active model selector. Use it to switch models without opening Settings; the selected model is used for the next run in the conversation.</p>
          <div class="settings-list">
            <div class="settings-row">
              <div class="settings-row-main">
                <span class="settings-row-title">Status display</span>
                <span class="settings-row-meta">Run status remains visible in the Run panel and in the conversation list.</span>
              </div>
              <span class="settings-badge">Model switcher</span>
            </div>
          </div>
        </section>
        <section id="settings-about" class="settings-panel" role="tabpanel" aria-labelledby="settings-tab-about" data-settings-panel="about" hidden>
          <h3>About mini-SWE-agent</h3>
          <p class="settings-note">A lightweight browser GUI for inspecting and steering mini-SWE-agent conversations. Settings are stored locally beside the conversation database.</p>
        </section>
      </div>
    </div>
  </div>
  <div id="image-preview" class="image-preview" role="dialog" aria-modal="true" aria-label="Image preview" hidden>
    <button id="image-preview-close" class="image-preview-close" type="button" aria-label="Close image preview">×</button>
    <img id="image-preview-image" alt="Image preview">
    <a id="image-preview-open" href="#" target="_blank" rel="noopener noreferrer">Open original</a>
  </div>

  <div id="context-menu" class="context-menu" role="menu" hidden></div>

  <script>
    const feed = document.getElementById("feed");
    const empty = document.getElementById("empty");
    const taskEl = document.getElementById("task");
    const composerEl = document.querySelector(".composer");
    const attachImageButton = document.getElementById("attach-image");
    const imageInput = document.getElementById("image-input");
    const composerAttachments = document.getElementById("composer-attachments");
    const sendButton = document.getElementById("send");
    const stopButton = document.getElementById("stop");
    const statusEl = document.getElementById("status");
    const conversationLabel = document.getElementById("conversation-label");
    const conversationWorkspace = document.getElementById("conversation-workspace");
    const statStep = document.getElementById("stat-step");
    const statCost = document.getElementById("stat-cost");
    const statTokens = document.getElementById("stat-tokens");
    const statContext = document.getElementById("stat-context");
    const statInput = document.getElementById("stat-input");
    const statOutput = document.getElementById("stat-output");
    const statTools = document.getElementById("stat-tools");
    const activityLog = document.getElementById("activity-log");
    const activityEmpty = document.getElementById("activity-empty");
    const activityScroller = document.getElementById("activity-scroll");
    const activitySummary = document.getElementById("activity-summary");
    const activityOverview = document.getElementById("activity-overview");
    const activityExpandButton = document.getElementById("activity-expand");
    const activityOpenButton = document.getElementById("activity-open");
    const conversationList = document.getElementById("conversation-list");
    const treeToggle = document.getElementById("tree-toggle");
    const treeCount = document.getElementById("tree-count");
    const mainTreeView = document.getElementById("main-tree-view");
    const mainConversationTree = document.getElementById("main-conversation-tree");
    const mainTreeClose = document.getElementById("main-tree-close");
    const contextMenu = document.getElementById("context-menu");
    const imagePreview = document.getElementById("image-preview");
    const imagePreviewImage = document.getElementById("image-preview-image");
    const imagePreviewOpen = document.getElementById("image-preview-open");
    const imagePreviewClose = document.getElementById("image-preview-close");
    const conversationEmpty = document.getElementById("conversation-empty");
    const CONVERSATION_STATES = ["running", "submitted", "stopped", "failed", "interrupted"];
    const newConversationButton = document.getElementById("new-conversation");
    const scrollTopButton = document.getElementById("scroll-top");
    const scrollBottomButton = document.getElementById("scroll-bottom");
    const modelSwitcher = document.getElementById("model-switcher");
    const settingsButton = document.getElementById("settings-button");
    const settingsModal = document.getElementById("settings-modal");
    const settingsCloseButton = document.getElementById("settings-close");
    const settingsTabs = Array.from(document.querySelectorAll(".settings-tab"));
    const settingsPanels = Array.from(document.querySelectorAll(".settings-panel"));
    const providersList = document.getElementById("providers-list");
    const modelsList = document.getElementById("models-list");
    const providerForm = document.getElementById("provider-form");
    const modelForm = document.getElementById("model-form");
    const addProviderButton = document.getElementById("add-provider");
    const addModelButton = document.getElementById("add-model");
    const providerCancelButton = document.getElementById("provider-cancel");
    const providerFetchModelsButton = document.getElementById("provider-fetch-models");
    const modelCancelButton = document.getElementById("model-cancel");
    const modelProviderSelect = document.getElementById("model-provider");
    const SCROLL_THRESHOLD = 32;
    const MAX_RECENT_ACTIVITIES = 6;
    const SESSION_STORAGE_KEY = "mini-swe-agent-session";
    let eventSource = null;
    let activeSessionId = null;
    let replayingHistory = false;
    let lastEventOffset = 0;
    let lastEventType = "";
    let selectedSessionId = null;
    let conversationSelectionVersion = 0;
    let pendingTreeNavigation = null;
    let conversationTreeEdges = [];
    let conversationTreeDots = new Map();
    let conversationTreeEdgesFrame = null;
    let conversations = [];
    let defaultWorkspace = "";
    let databasePath = "";
    let currentAssistant = null;
    let currentReasoning = null;
    let pendingImages = [];
    const MAX_IMAGES = 8;
    const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
    let followOutput = true;
    let currentTokenUsage = { prompt: 0, completion: 0, total: 0 };
    // Conversation totals: `completedTokenUsage` sums finished turns, `runTokenUsage`
    // tracks the model calls of the turn that is currently running.
    let completedTokenUsage = { prompt: 0, completion: 0, total: 0 };
    let runTokenUsage = { prompt: 0, completion: 0, total: 0 };
    let guiSettings = { providers: [], models: [], enabled_models: [], active_model: "" };
    let currentSettingsTab = "general";
    let editingProviderId = "";
    let editingModelId = "";
    const expandedProviderIds = new Set();
    // Completed assistant messages are kept here until the immediately following
    // events have been inspected.  If one of those events is a tool call for the
    // same command, the command is removed from the assistant text so that the
    // tool-call row is the single place where the command is shown.
    let pendingAssistantMessage = null;
    let activityTimer = null;
    let activityLayoutFrame = null;
    let activityOverviewFrame = null;
    let activityFollowOutput = true;
    let currentActivityStep = null;
    let currentActivityStepNumber = null;
    let currentActivityThinking = null;
    let activityLiveText = "";
    let activityReasoningText = "";
    let activityState = {
      steps: new Set(),
      tools: 0,
      errors: 0,
      segments: [],
      startedAt: 0,
      finishedAt: 0,
      finishedWithError: false,
    };
    let activityStepEntries = new Map();
    let activityToolEntries = new Map();

    function numeric(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number : 0;
    }

    function formatTokens(value) {
      return numeric(value).toLocaleString();
    }

    function setTokenStats(prompt, completion, total) {
      const promptTokens = numeric(prompt);
      const completionTokens = numeric(completion);
      const totalTokens = numeric(total) || promptTokens + completionTokens;
      statTokens.textContent = totalTokens ? formatTokens(totalTokens) : "-";
      statInput.textContent = totalTokens ? formatTokens(promptTokens) : "-";
      statOutput.textContent = totalTokens ? formatTokens(completionTokens) : "-";
    }

    function contextLengthFromAssistant(event, prompt, completion) {
      const usage = event.usage || {};
      return numeric(event.total_tokens ?? event.tokens ?? usage.total_tokens) || numeric(prompt) + numeric(completion);
    }

    function setContextStats(total, prompt = 0, completion = 0) {
      const contextTokens = numeric(total);
      statContext.textContent = contextTokens ? formatTokens(contextTokens) : '-';
      statContext.title = contextTokens
        ? formatTokens(contextTokens) + ' tokens after the latest model call (' + formatTokens(prompt) + ' in / ' + formatTokens(completion) + ' out)'
        : 'Current context length after the latest model call';
    }

    function setTokenStatsFromConversation(conversation) {
      const usage = (conversation && (conversation.token_usage || conversation.usage)) || {};
      currentTokenUsage = {
        prompt: numeric(conversation && (conversation.total_prompt_tokens ?? conversation.prompt_tokens ?? usage.prompt_tokens)),
        completion: numeric(conversation && (conversation.total_completion_tokens ?? conversation.completion_tokens ?? usage.completion_tokens)),
        total: numeric(conversation && (conversation.total_tokens_used ?? conversation.total_tokens ?? conversation.tokens ?? usage.total_tokens)),
      };
      completedTokenUsage = { ...currentTokenUsage };
      runTokenUsage = { prompt: 0, completion: 0, total: 0 };
      setTokenStats(currentTokenUsage.prompt, currentTokenUsage.completion, currentTokenUsage.total);
      setContextStats(conversation && conversation.context_tokens);
    }

    function updateTokenStatsFromAssistant(event) {
      const usage = event.usage || {};
      const callPrompt = numeric(event.prompt_tokens ?? usage.prompt_tokens);
      const callCompletion = numeric(event.completion_tokens ?? usage.completion_tokens);
      const callTotal = contextLengthFromAssistant(event, callPrompt, callCompletion);
      if (!replayingHistory) {
        const conversationUsage = event.conversation_usage;
        if (conversationUsage) {
          currentTokenUsage = {
            prompt: numeric(conversationUsage.prompt_tokens),
            completion: numeric(conversationUsage.completion_tokens),
            total: numeric(conversationUsage.total_tokens),
          };
        } else if (event.cumulative_usage) {
          const prompt = numeric(event.cumulative_usage.prompt_tokens);
          const completion = numeric(event.cumulative_usage.completion_tokens);
          const total = numeric(event.cumulative_usage.total_tokens) || prompt + completion;
          currentTokenUsage = {
            prompt: completedTokenUsage.prompt + prompt,
            completion: completedTokenUsage.completion + completion,
            total: completedTokenUsage.total + total,
          };
        } else {
          runTokenUsage.prompt += callPrompt;
          runTokenUsage.completion += callCompletion;
          runTokenUsage.total += callTotal;
          currentTokenUsage = {
            prompt: completedTokenUsage.prompt + runTokenUsage.prompt,
            completion: completedTokenUsage.completion + runTokenUsage.completion,
            total: completedTokenUsage.total + runTokenUsage.total,
          };
        }
        setTokenStats(currentTokenUsage.prompt, currentTokenUsage.completion, currentTokenUsage.total);
      }
      setContextStats(callTotal, callPrompt, callCompletion);
      return callTotal;
    }

    function updateTokenStatsFromDone(event) {
      const conversationUsage = event.conversation_usage;
      const usage = event.token_usage || event.cumulative_usage || event.usage || {};
      if (conversationUsage) {
        completedTokenUsage = {
          prompt: numeric(conversationUsage.prompt_tokens),
          completion: numeric(conversationUsage.completion_tokens),
          total: numeric(conversationUsage.total_tokens),
        };
      } else {
        const prompt = numeric(event.total_prompt_tokens ?? event.prompt_tokens ?? usage.prompt_tokens);
        const completion = numeric(event.total_completion_tokens ?? event.completion_tokens ?? usage.completion_tokens);
        const total = numeric(event.total_tokens_used ?? event.total_tokens ?? event.tokens ?? usage.total_tokens) || prompt + completion;
        completedTokenUsage = {
          prompt: completedTokenUsage.prompt + prompt,
          completion: completedTokenUsage.completion + completion,
          total: completedTokenUsage.total + total,
        };
      }
      runTokenUsage = { prompt: 0, completion: 0, total: 0 };
      currentTokenUsage = { ...completedTokenUsage };
      setTokenStats(currentTokenUsage.prompt, currentTokenUsage.completion, currentTokenUsage.total);
    }

    function setStatus(text, state = "") {
      statusEl.textContent = text;
      statusEl.className = "status " + state;
      statusEl.parentElement.dataset.status = state || "idle";
    }

    async function apiRequest(url, options = {}) {
      const response = await fetch(url, options);
      let data = {};
      try { data = await response.json(); } catch (err) { data = {}; }
      if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
      return data;
    }

    function modelNameOf(model) {
      return String((model && (model.model_name || model.id)) || "");
    }

    function modelDisplayName(model) {
      const name = String((model && model.name) || "");
      const modelName = modelNameOf(model);
      if (!name || name === modelName) return modelName;
      return name + " \u00b7 " + modelName;
    }

    function modelProviderName(model) {
      const provider = String((model && model.provider) || "");
      return provider ? providerName(provider) : "";
    }

    function modelOptionLabel(model) {
      const label = modelDisplayName(model);
      const provider = modelProviderName(model);
      return provider ? label + " \u2014 " + provider : label;
    }

    function enabledModels() {
      if (Array.isArray(guiSettings.enabled_models) && guiSettings.enabled_models.length) {
        return guiSettings.enabled_models;
      }
      return (guiSettings.models || []).filter((model) => model.enabled !== false);
    }

    function renderModelSwitcher(preferred) {
      const previous = preferred !== undefined ? String(preferred || "") : String(modelSwitcher.value || "");
      const enabled = enabledModels().slice();
      const known = new Set(enabled.map(modelNameOf));
      if (previous && !known.has(previous)) {
        const knownModel = (guiSettings.models || []).find((model) => modelNameOf(model) === previous);
        enabled.unshift(knownModel || { model_name: previous, name: previous, enabled: true });
        known.add(previous);
      }
      modelSwitcher.textContent = "";
      if (!enabled.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No model configured";
        option.disabled = true;
        modelSwitcher.appendChild(option);
        modelSwitcher.value = "";
        return;
      }
      for (const model of enabled) {
        const modelName = modelNameOf(model);
        if (!modelName) continue;
        const option = document.createElement("option");
        option.value = modelName;
        option.textContent = modelOptionLabel(model);
        modelSwitcher.appendChild(option);
      }
      const active = String(guiSettings.active_model || "");
      const selected = previous && known.has(previous) ? previous : (known.has(active) ? active : modelNameOf(enabled[0]));
      modelSwitcher.value = selected;
    }

    function renderProviderOptions(selected) {
      const current = String(selected || "");
      modelProviderSelect.textContent = "";
      const providers = guiSettings.providers || [];
      if (!providers.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No providers yet";
        modelProviderSelect.appendChild(option);
        return;
      }
      for (const provider of providers) {
        const option = document.createElement("option");
        option.value = provider.id;
        option.textContent = provider.name + (provider.enabled === false ? " (disabled)" : "");
        modelProviderSelect.appendChild(option);
      }
      if (current) modelProviderSelect.value = current;
    }

    function providerName(providerId) {
      const provider = (guiSettings.providers || []).find((item) => item.id === providerId);
      return provider ? provider.name : providerId;
    }

    function showProviderForm(provider) {
      editingProviderId = provider ? String(provider.id || "") : "";
      providerForm.hidden = false;
      document.getElementById("provider-edit-id").value = editingProviderId;
      document.getElementById("provider-id").value = editingProviderId;
      document.getElementById("provider-id").disabled = Boolean(provider);
      document.getElementById("provider-name").value = provider ? String(provider.name || "") : "";
      document.getElementById("provider-api-key-env").value = provider ? String(provider.api_key_env || "") : "";
      document.getElementById("provider-base-url").value = provider ? String(provider.base_url || "") : "";
      const apiKey = document.getElementById("provider-api-key");
      apiKey.value = "";
      apiKey.placeholder = provider && provider.has_api_key ? "Leave blank to keep the existing key" : "Paste your API key here";
      document.getElementById("provider-name").focus();
    }

    function hideProviderForm() {
      editingProviderId = "";
      providerForm.hidden = true;
      providerForm.reset();
      document.getElementById("provider-id").disabled = false;
    }

    function showModelForm(model, providerId = "") {
      if (!(guiSettings.providers || []).length) {
        window.alert("Add a provider first.");
        return;
      }
      editingModelId = model ? modelNameOf(model) : "";
      const modelNameInput = document.getElementById("model-name");
      modelForm.hidden = false;
      document.getElementById("model-edit-id").value = editingModelId;
      modelNameInput.value = editingModelId;
      modelNameInput.disabled = Boolean(model);
      document.getElementById("model-display-name").value = model ? String(model.name || "") : "";
      document.getElementById("model-class").value = model ? String(model.model_class || "") : "";
      const selectedProvider = model
        ? String(model.provider || "")
        : String(providerId || (guiSettings.providers[0] || {}).id || "");
      renderProviderOptions(selectedProvider);
      modelNameInput.focus();
    }

    function hideModelForm() {
      editingModelId = "";
      modelForm.hidden = true;
      modelForm.reset();
      document.getElementById("model-name").disabled = false;
    }

    function renderProviderModelRow(model) {
      const modelName = modelNameOf(model);
      const row = document.createElement("div");
      row.className = "settings-row model-row provider-model-row";
      row.dataset.modelName = modelName;
      const main = document.createElement("div");
      main.className = "settings-row-main";
      const title = document.createElement("span");
      title.className = "settings-row-title";
      title.textContent = String(model.name || modelName);
      const meta = document.createElement("span");
      meta.className = "settings-row-meta";
      const bits = [modelName];
      if (model.model_class) bits.push(String(model.model_class));
      meta.textContent = bits.join(" \u00b7 ");
      main.appendChild(title);
      main.appendChild(meta);
      const actions = document.createElement("div");
      actions.className = "settings-row-actions";
      const toggleLabel = document.createElement("label");
      toggleLabel.className = "settings-toggle";
      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.className = "model-enabled-toggle";
      toggle.checked = model.enabled !== false;
      toggle.addEventListener("change", () => toggleModel(modelName, toggle.checked));
      toggleLabel.appendChild(toggle);
      toggleLabel.appendChild(document.createTextNode(" Enabled"));
      const edit = document.createElement("button");
      edit.type = "button";
      edit.className = "secondary";
      edit.textContent = "Edit";
      edit.addEventListener("click", () => showModelForm(model, String(model.provider || "")));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "danger";
      remove.textContent = "Delete";
      remove.addEventListener("click", () => deleteModel(modelName));
      actions.appendChild(toggleLabel);
      actions.appendChild(edit);
      actions.appendChild(remove);
      row.appendChild(main);
      row.appendChild(actions);
      return row;
    }

    function toggleProviderModels(providerId) {
      const key = String(providerId || "");
      if (!key) return;
      if (expandedProviderIds.has(key)) expandedProviderIds.delete(key);
      else expandedProviderIds.add(key);
      renderSettings();
    }

    function renderSettings() {
      const activeModel = String(guiSettings.active_model || "");
      document.getElementById("settings-active-model").textContent = activeModel || "Not configured";
      document.getElementById("settings-workspace").textContent = defaultWorkspace || "-";
      const databaseText = databasePath || "-";
      document.getElementById("settings-database").textContent = databaseText;

      providersList.textContent = "";
      const providers = guiSettings.providers || [];
      const models = guiSettings.models || [];
      if (!providers.length) {
        const emptyState = document.createElement("div");
        emptyState.className = "settings-empty";
        emptyState.textContent = "No providers yet. Add one to manage model endpoints and API keys.";
        providersList.appendChild(emptyState);
      }
      for (const provider of providers) {
        const providerId = String(provider.id || "");
        const providerModels = models.filter((model) => String(model.provider || "") === providerId);
        const expanded = expandedProviderIds.has(providerId);
        const group = document.createElement("div");
        group.className = "provider-group";
        group.dataset.providerId = providerId;

        const row = document.createElement("div");
        row.className = "settings-row provider-row";
        row.dataset.providerId = providerId;
        const main = document.createElement("div");
        main.className = "settings-row-main provider-row-main";
        main.tabIndex = 0;
        main.setAttribute("role", "button");
        main.setAttribute("aria-expanded", expanded ? "true" : "false");
        main.title = expanded ? "Hide models" : "Show models";
        const title = document.createElement("span");
        title.className = "settings-row-title";
        title.textContent = (expanded ? "\u25be " : "\u25b8 ") + String(provider.name || provider.id || "Provider");
        const meta = document.createElement("span");
        meta.className = "settings-row-meta";
        const bits = [String(provider.id || "")];
        if (provider.api_key_env) bits.push("env var: " + provider.api_key_env);
        if (provider.has_api_key) bits.push("API key saved locally");
        if (provider.base_url) bits.push(provider.base_url);
        bits.push(providerModels.length + " model(s)");
        meta.textContent = bits.join(" \u00b7 ");
        main.appendChild(title);
        main.appendChild(meta);
        main.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            toggleProviderModels(providerId);
          }
        });
        const actions = document.createElement("div");
        actions.className = "settings-row-actions";
        const badge = document.createElement("span");
        badge.className = "settings-badge" + (provider.enabled === false ? " off" : "");
        badge.textContent = provider.enabled === false ? "Disabled" : "Enabled";
        const edit = document.createElement("button");
        edit.type = "button";
        edit.className = "secondary";
        edit.textContent = "Edit";
        edit.addEventListener("click", () => showProviderForm(provider));
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "secondary";
        toggle.textContent = provider.enabled === false ? "Enable" : "Disable";
        toggle.addEventListener("click", () => toggleProvider(provider));
        const loadModels = document.createElement("button");
        loadModels.type = "button";
        loadModels.className = "secondary provider-load-models";
        loadModels.textContent = "Load models";
        loadModels.title = "Query this provider's model list and add the results under this provider";
        loadModels.dataset.action = "fetch-models";
        loadModels.addEventListener("click", () => loadProviderModels(provider, loadModels));
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "danger";
        remove.textContent = "Delete";
        remove.addEventListener("click", () => deleteProvider(provider));
        actions.appendChild(badge);
        actions.appendChild(loadModels);
        actions.appendChild(edit);
        actions.appendChild(toggle);
        actions.appendChild(remove);
        row.appendChild(main);
        row.appendChild(actions);
        row.addEventListener("click", (event) => {
          if (event.target.closest("button, input, label")) return;
          toggleProviderModels(providerId);
        });
        group.appendChild(row);

        const nested = document.createElement("div");
        nested.className = "provider-models";
        nested.dataset.providerId = providerId;
        nested.hidden = !expanded;
        const nestedHead = document.createElement("div");
        nestedHead.className = "provider-models-head";
        const nestedLabel = document.createElement("span");
        nestedLabel.textContent = providerModels.length + " model(s)";
        const addModel = document.createElement("button");
        addModel.type = "button";
        addModel.className = "secondary provider-add-model";
        addModel.textContent = "Add model";
        addModel.addEventListener("click", () => showModelForm(null, providerId));
        nestedHead.appendChild(nestedLabel);
        nestedHead.appendChild(addModel);
        nested.appendChild(nestedHead);
        if (!providerModels.length) {
          const empty = document.createElement("div");
          empty.className = "provider-models-empty";
          empty.textContent = "No models under this provider yet.";
          nested.appendChild(empty);
        } else {
          for (const model of providerModels) nested.appendChild(renderProviderModelRow(model));
        }
        group.appendChild(nested);
        providersList.appendChild(group);
      }
      renderModelSwitcher();
    }

    async function loadGuiSettings(options = {}) {
      try {
        guiSettings = await apiRequest("/api/settings", { cache: "no-store" });
        renderSettings();
        if (options.selectActive) renderModelSwitcher(String(guiSettings.active_model || ""));
      } catch (err) {
        console.error("Could not load GUI settings", err);
      }
    }

    async function switchModel(modelName) {
      if (!modelName) return;
      try {
        const data = await apiRequest("/api/settings/model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_name: modelName }),
        });
        guiSettings.active_model = data.active_model || modelName;
        if (!Array.isArray(guiSettings.models)) guiSettings.models = [];
        renderModelSwitcher(modelName);
        renderSettings();
        if (selectedSessionId) {
          const conversation = conversations.find((item) => conversationId(item) === selectedSessionId);
          if (conversation) conversation.model_name = modelName;
        }
      } catch (err) {
        window.alert("Could not switch model: " + String(err));
        await loadGuiSettings();
      }
    }

    async function toggleProvider(provider) {
      try {
        await apiRequest("/api/providers/" + encodeURIComponent(String(provider.id || "")), {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: provider.enabled === false }),
        });
        await loadGuiSettings();
      } catch (err) {
        window.alert("Could not update provider: " + String(err));
      }
    }

    async function deleteProvider(provider) {
      if (!window.confirm("Delete provider \"" + String(provider.name || provider.id) + "\" and its models?")) return;
      const providerId = String(provider.id || "");
      try {
        await apiRequest("/api/providers/" + encodeURIComponent(providerId), { method: "DELETE" });
        expandedProviderIds.delete(providerId);
        hideProviderForm();
        hideModelForm();
        await loadGuiSettings();
      } catch (err) {
        window.alert("Could not delete provider: " + String(err));
      }
    }

    async function toggleModel(modelName, enabled) {
      try {
        await apiRequest("/api/models/" + encodeURIComponent(modelName), {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: Boolean(enabled) }),
        });
        await loadGuiSettings();
      } catch (err) {
        window.alert("Could not update model: " + String(err));
        await loadGuiSettings();
      }
    }

    async function deleteModel(modelName) {
      if (!window.confirm("Delete model \"" + modelName + "\"?")) return;
      try {
        await apiRequest("/api/models/" + encodeURIComponent(modelName), { method: "DELETE" });
        hideModelForm();
        await loadGuiSettings();
      } catch (err) {
        window.alert("Could not delete model: " + String(err));
      }
    }

    function providerFormPayload() {
      const envNameInput = document.getElementById("provider-api-key-env");
      const payload = {
        id: document.getElementById("provider-id").value.trim(),
        name: document.getElementById("provider-name").value.trim(),
        api_key_env: envNameInput.value.trim(),
        base_url: document.getElementById("provider-base-url").value.trim(),
        api_key: document.getElementById("provider-api-key").value,
      };
      if (!payload.name) return null;
      if (payload.api_key_env && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(payload.api_key_env)) {
        window.alert(
          "The \"API key environment variable\" field expects the name of a variable (for example OPENAI_API_KEY), not your API key. " +
          "Paste the actual key into the \"API key (optional, stored locally)\" field below."
        );
        envNameInput.focus();
        return null;
      }
      return payload;
    }

    async function persistProviderForm() {
      const payload = providerFormPayload();
      if (!payload) return null;
      if (editingProviderId) {
        delete payload.id;
        return await apiRequest("/api/providers/" + encodeURIComponent(editingProviderId), {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }
      return await apiRequest("/api/providers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    async function loadProviderModels(provider, button) {
      const providerId = String((provider && provider.id) || "");
      if (!providerId) {
        window.alert("Save the provider before loading its models.");
        return;
      }
      const originalLabel = button ? button.textContent : "";
      if (button) {
        button.disabled = true;
        button.textContent = "Loading...";
      }
      try {
        const data = await apiRequest(
          "/api/providers/" + encodeURIComponent(providerId) + "/load-models",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          },
        );
        guiSettings = data;
        expandedProviderIds.add(providerId);
        renderSettings();
        renderModelSwitcher(String(data.active_model || modelSwitcher.value || ""));
        const fetched = Array.isArray(data.fetched_models) ? data.fetched_models : [];
        const loaded = Array.isArray(data.loaded_models) ? data.loaded_models : [];
        if (loaded.length) {
          window.alert(
            "Loaded " + loaded.length + " new model(s). They are enabled and available in the conversation model selector."
          );
        } else if (fetched.length) {
          window.alert("The provider's models are already loaded. They are available in the conversation model selector.");
        } else {
          window.alert("The provider returned no models.");
        }
      } catch (err) {
        window.alert("Could not load models: " + String(err));
        await loadGuiSettings();
      } finally {
        if (button) {
          button.disabled = false;
          button.textContent = originalLabel;
        }
      }
    }

    async function fetchProviderModels(provider, button) {
      // The /fetch-models and /load-models API routes are equivalent aliases.
      return await loadProviderModels(provider, button);
    }

    async function saveProviderForm(event) {
      event.preventDefault();
      try {
        const data = await persistProviderForm();
        if (!data) return;
        const providerId = String((data.provider || {}).id || "");
        if (providerId) expandedProviderIds.add(providerId);
        hideProviderForm();
        await loadGuiSettings();
      } catch (err) {
        window.alert("Could not save provider: " + String(err));
      }
    }

    async function saveAndLoadProviderModels() {
      try {
        const data = await persistProviderForm();
        if (!data) return;
        const provider = data.provider || (data.providers || []).find((item) => item.id === editingProviderId);
        if (!provider || !provider.id) throw new Error("The provider could not be saved.");
        hideProviderForm();
        await loadProviderModels(provider, null);
      } catch (err) {
        window.alert("Could not save provider: " + String(err));
      }
    }

    async function saveModelForm(event) {
      event.preventDefault();
      const payload = {
        model_name: document.getElementById("model-name").value.trim(),
        name: document.getElementById("model-display-name").value.trim(),
        provider: document.getElementById("model-provider").value,
        model_class: document.getElementById("model-class").value.trim(),
      };
      if (!payload.model_name) return;
      try {
        if (editingModelId) {
          delete payload.model_name;
          await apiRequest("/api/models/" + encodeURIComponent(editingModelId), {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
        } else {
          await apiRequest("/api/models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
        }
        hideModelForm();
        await loadGuiSettings();
      } catch (err) {
        window.alert("Could not save model: " + String(err));
      }
    }

    function switchSettingsTab(tabName) {
      currentSettingsTab = tabName || "general";
      for (const tab of settingsTabs) {
        const active = tab.dataset.settingsTab === currentSettingsTab;
        tab.classList.toggle("active", active);
        tab.setAttribute("aria-selected", active ? "true" : "false");
      }
      for (const panel of settingsPanels) {
        panel.hidden = panel.dataset.settingsPanel !== currentSettingsTab;
      }
    }

    function openSettings(tabName) {
      settingsModal.hidden = false;
      switchSettingsTab(tabName || currentSettingsTab);
      loadGuiSettings();
      document.getElementById("settings-title").focus();
    }

    function closeSettings() {
      settingsModal.hidden = true;
      hideProviderForm();
      hideModelForm();
    }

    function setComposerRunning(running) {
      sendButton.hidden = running;
      stopButton.hidden = !running;
      sendButton.disabled = running;
      stopButton.disabled = !running;
    }

    function resizeTaskInput() {
      if (!taskEl.value) {
        taskEl.style.height = "42px";
        taskEl.style.overflowY = "hidden";
        return;
      }
      taskEl.style.height = "auto";
      const height = Math.max(42, Math.min(taskEl.scrollHeight, 168));
      taskEl.style.height = height + "px";
      taskEl.style.overflowY = taskEl.scrollHeight > 168 ? "auto" : "hidden";
    }

    function isFeedAtBottom() {
      return feed.scrollHeight - feed.scrollTop - feed.clientHeight <= SCROLL_THRESHOLD;
    }

    function updateScrollButtons() {
      scrollTopButton.hidden = feed.scrollTop <= 4;
      scrollBottomButton.hidden = isFeedAtBottom();
    }

    function scrollFeedToBottom(behavior = "auto") {
      feed.scrollTo({ top: feed.scrollHeight, behavior });
    }

    function reasoningIsExpanded() {
      if (!currentReasoning) return false;
      const details = currentReasoning.closest("details");
      return Boolean(details && details.open);
    }

    function eventAffectsVisibleFeed(event) {
      // A collapsed reasoning block hides its streamed text, so appending more
      // deltas must not keep forcing the transcript to the bottom.
      return event.type !== "reasoning_delta" || reasoningIsExpanded();
    }

    feed.addEventListener("scroll", () => {
      followOutput = isFeedAtBottom();
      updateScrollButtons();
    }, { passive: true });

    scrollTopButton.addEventListener("click", () => {
      followOutput = false;
      feed.scrollTo({ top: 0, behavior: "smooth" });
    });

    scrollBottomButton.addEventListener("click", () => {
      followOutput = true;
      scrollFeedToBottom();
    });

    function formatClock(timestamp = Date.now()) {
      return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function formatElapsed(milliseconds) {
      const totalSeconds = Math.max(0, Math.floor(numeric(milliseconds) / 1000));
      if (totalSeconds < 60) return totalSeconds + "s";
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      if (minutes < 60) return minutes + "m " + String(seconds).padStart(2, "0") + "s";
      const hours = Math.floor(minutes / 60);
      return hours + "h " + String(minutes % 60).padStart(2, "0") + "m";
    }

    function previewText(value, maxLength = 180) {
      const text = String(value ?? "").replace(/\r/g, "").trim();
      if (!text) return "";
      const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
      const first = (lines[0] || "").replace(/\s+/g, " ");
      const suffix = lines.length > 1 ? "  +" + (lines.length - 1) + " more line" + (lines.length === 2 ? "" : "s") : "";
      const shortened = first.length > maxLength ? first.slice(0, maxLength - 1) + "…" : first;
      return shortened + suffix;
    }

    function activityStatusTone(status) {
      const normalized = String(status || "").toLowerCase();
      if (normalized === "running" || normalized === "thinking" || normalized === "waiting") return "running";
      if (normalized === "responded" || normalized.includes("done") || normalized.includes("complete") || normalized === "submitted" || normalized === "exit 0") return "ok";
      if (normalized.includes("error") || normalized.includes("fail") || (normalized.startsWith("exit ") && normalized !== "exit 0")) return "error";
      return "";
    }

    function activityMarker(kind) {
      if (kind === "tool") return "→";
      if (kind === "done") return "✓";
      if (kind === "error") return "!";
      if (kind === "run") return "▶";
      return "•";
    }

    function activityStatusIsFinished(status) {
      const normalized = String(status || "").toLowerCase();
      return normalized === "responded"
        || normalized === "submitted"
        || normalized === "stopped"
        || normalized === "failed"
        || normalized === "finished"
        || normalized.includes("complete")
        || normalized.includes("failed")
        || normalized.includes("done")
        || normalized.startsWith("exit ");
    }

    function activityIsExpanded() {
      return document.body.classList.contains("activity-expanded");
    }

    function formatActivityDuration(milliseconds) {
      const duration = Math.max(0, numeric(milliseconds));
      if (duration < 1000) return Math.round(duration) + "ms";
      return formatElapsed(duration);
    }

    function activityDuration(entry) {
      const startedAt = Number(entry.dataset.startedAt || 0);
      if (!startedAt) return 0;
      const endedAt = Number(entry.dataset.endedAt || 0) || Date.now();
      return Math.max(0, endedAt - startedAt);
    }

    function markActivityEnded(entry, timestamp = Date.now()) {
      if (entry && !entry.dataset.endedAt) entry.dataset.endedAt = String(timestamp);
    }

    function finalizeOpenActivities(timestamp = Date.now()) {
      activityLog.querySelectorAll(".activity-entry").forEach((entry) => {
        endActivityPhase(entry, timestamp);
        if (!entry.dataset.endedAt) markActivityEnded(entry, timestamp);
      });
    }

    function scheduleActivityLayout() {
      if (activityLayoutFrame != null) return;
      activityLayoutFrame = requestAnimationFrame(() => {
        activityLayoutFrame = null;
        layoutActivityEntries();
        if (activityIsExpanded() && activityFollowOutput) scrollActivityToBottom();
      });
    }

    function activityIsNearBottom() {
      const remaining = activityScroller.scrollHeight - activityScroller.scrollTop - activityScroller.clientHeight;
      return remaining <= 24;
    }

    function scrollActivityToBottom() {
      activityScroller.scrollTop = activityScroller.scrollHeight;
      activityFollowOutput = true;
    }

    function setActivitySpineState(hasEntries, clippedHistory) {
      activityScroller.classList.toggle("has-activity", hasEntries);
      activityScroller.classList.toggle("has-clipped-history", clippedHistory);
    }

    function layoutActivityEntries() {
      const entries = Array.from(activityLog.querySelectorAll(".activity-entry"));
      if (!entries.length) {
        setActivitySpineState(false, false);
        drawActivityOverview();
        return;
      }
      if (!activityIsExpanded()) {
        const firstVisible = Math.max(0, entries.length - MAX_RECENT_ACTIVITIES);
        entries.forEach((entry, index) => { entry.hidden = index < firstVisible; });
        const candidates = entries.slice(firstVisible);
        candidates.forEach((entry) => { entry.hidden = false; });
        // clientHeight includes the scroller padding; don't count that as
        // space that can hold entries.
        const scrollerStyles = getComputedStyle(activityScroller);
        const verticalPadding = (parseFloat(scrollerStyles.paddingTop) || 0)
          + (parseFloat(scrollerStyles.paddingBottom) || 0);
        const available = Math.max(0, activityScroller.clientHeight - verticalPadding);
        let used = 0;
        let clippedHistory = firstVisible > 0;
        for (let index = candidates.length - 1; index >= 0; index -= 1) {
          const entry = candidates[index];
          const height = entry.getBoundingClientRect().height;
          if (used > 0 && used + height > available + 0.5) {
            entry.hidden = true;
            clippedHistory = true;
          } else {
            // Each entry already includes its own bottom padding; there is no
            // gap between entries to account for here.
            used += height;
          }
        }
        setActivitySpineState(candidates.some((entry) => !entry.hidden), clippedHistory);
        entries.forEach((entry) => {
          entry.style.removeProperty("--activity-offset");
          entry.style.removeProperty("--activity-width");
          const duration = entry.querySelector(".activity-duration");
          if (duration) duration.hidden = true;
        });
        drawActivityOverview();
        return;
      }

      // Expanded activity is a readable, scrollable ledger. Keep the legacy
      // custom properties as layout hooks, but give every row the full width.
      setActivitySpineState(true, false);
      entries.forEach((entry) => {
        entry.hidden = false;
        const duration = activityDuration(entry);
        entry.style.setProperty("--activity-offset", "0px");
        entry.style.setProperty("--activity-width", "100%");
        const durationEl = entry.querySelector(".activity-duration");
        if (durationEl) {
          durationEl.hidden = false;
          durationEl.textContent = formatActivityDuration(duration);
        }
        const label = entry.querySelector(".activity-label");
        entry.title = (label ? label.textContent : "Activity") + " · " + formatActivityDuration(duration);
      });
      drawActivityOverview();
    }

    function setActivityExpanded(expanded) {
      document.body.classList.toggle("activity-expanded", expanded);
      activityExpandButton.textContent = expanded ? "Collapse" : "Expand";
      activityExpandButton.setAttribute("aria-expanded", String(expanded));
      activityExpandButton.title = expanded ? "Collapse activity view" : "Expand activity view";
      if (expanded) activityFollowOutput = true;
      scheduleActivityLayout();
    }

    activityExpandButton.addEventListener("click", () => setActivityExpanded(!activityIsExpanded()));
    activityOpenButton.addEventListener("click", () => setActivityExpanded(true));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && activityIsExpanded()) setActivityExpanded(false);
    });
    activityScroller.addEventListener("scroll", () => {
      if (activityIsExpanded()) {
        activityFollowOutput = activityIsNearBottom();
        scheduleActivityOverview();
      }
    }, { passive: true });
    window.addEventListener("resize", scheduleActivityLayout, { passive: true });

    function activityEventTime(event) {
      const timestamp = Number(event.timestamp || 0);
      return timestamp > 0 ? (timestamp < 1e12 ? timestamp * 1000 : timestamp) : Date.now();
    }

    function beginActivityPhase(entry, kind, timestamp) {
      if (!entry) return;
      const startedAt = Number(timestamp) || Date.now();
      const phases = entry._activityPhases || (entry._activityPhases = []);
      const current = phases[phases.length - 1];
      if (current && !current.endedAt) {
        if (current.kind === kind) return;
        current.endedAt = startedAt;
      }
      if (!phases.length) entry.dataset.startedAt = String(startedAt);
      delete entry.dataset.endedAt;
      phases.push({ kind, startedAt, endedAt: 0 });
      drawActivityOverview();
    }

    function endActivityPhase(entry, timestamp) {
      if (!entry) return;
      const phases = entry._activityPhases || [];
      const current = phases[phases.length - 1];
      if (!current || current.endedAt) return;
      current.endedAt = Number(timestamp) || Date.now();
      entry.dataset.endedAt = String(current.endedAt);
      drawActivityOverview();
    }

    function activityPhaseSegments(entries, now = Date.now()) {
      const segments = [];
      for (const entry of entries) {
        if (entry.hidden) continue;
        const visibility = Math.max(0, Math.min(1, Number(entry._activityVisibility ?? 1)));
        if (!visibility) continue;
        const phases = entry._activityPhases || [];
        if (!phases.length) continue;
        let duration = 0;
        let ongoing = false;
        let kind = 'step';
        for (const phase of phases) {
          const endedAt = Number(phase.endedAt || 0);
          duration += Math.max(0, (endedAt || now) - Number(phase.startedAt));
          ongoing = ongoing || !endedAt;
          kind = phase.kind;
        }
        segments.push({
          kind: ongoing ? 'running' : kind,
          label: entry._activityTitle || 'Activity',
          duration,
          visibility,
          ongoing,
        });
      }
      return segments;
    }

    function activitySegmentWeights(segments) {
      const completed = [];
      const running = [];
      segments.forEach((segment, index) => {
        segment.weight = 0;
        const item = { segment, index };
        (segment.ongoing ? running : completed).push(item);
      });
      const runningShare = running.reduce((total, item) => total + item.segment.visibility * 0.1, 0);
      if (runningShare >= 1) {
        const minimumTotal = segments.reduce((total, segment) => total + segment.visibility * 0.1, 0);
        segments.forEach((segment) => { segment.weight = segment.visibility * 0.1 / minimumTotal; });
        return segments;
      }
      running.forEach((item) => { item.segment.weight = item.segment.visibility * 0.1; });
      const available = 1 - runningShare;
      if (!completed.length) return segments;
      const minimum = (item) => item.segment.visibility * 0.1;
      const duration = (item) => item.segment.duration * item.segment.visibility;
      const minimumTotal = completed.reduce((total, item) => total + minimum(item), 0);
      if (minimumTotal >= available) {
        completed.forEach((item) => { item.segment.weight = available * minimum(item) / minimumTotal; });
        return segments;
      }
      let pending = completed.slice();
      let remaining = available;
      while (pending.length) {
        const totalDuration = pending.reduce((total, item) => total + duration(item), 0);
        const floor = pending.find((item) => {
          const share = totalDuration ? duration(item) / totalDuration * remaining : remaining / pending.length;
          return share + 1e-12 < minimum(item);
        });
        if (!floor) {
          pending.forEach((item) => {
            item.segment.weight = totalDuration ? duration(item) / totalDuration * remaining : remaining / pending.length;
          });
          break;
        }
        floor.segment.weight = minimum(floor);
        remaining -= minimum(floor);
        pending = pending.filter((item) => item !== floor);
      }
      return segments;
    }

    function activityEntriesInView() {
      const entries = Array.from(activityLog.querySelectorAll('.activity-entry')).filter((entry) => !entry.hidden);
      if (!activityIsExpanded()) {
        entries.forEach((entry) => { entry._activityVisibility = 1; });
        return entries;
      }
      const viewport = activityScroller.getBoundingClientRect();
      entries.forEach((entry) => {
        const rect = entry.getBoundingClientRect();
        const height = Math.max(0, rect.bottom - rect.top);
        const visible = Math.max(0, Math.min(rect.bottom, viewport.bottom) - Math.max(rect.top, viewport.top));
        entry._activityVisibility = height ? Math.max(0, Math.min(1, visible / height)) : 0;
      });
      return entries.filter((entry) => entry._activityVisibility > 0);
    }

    function syncActivityOverviewScroll() {
      const track = activityOverview.querySelector('.activity-overview-track');
      if (!track) return;
      track.style.width = activityOverview.clientWidth + 'px';
      track.style.transform = 'none';
    }

    function scheduleActivityOverview() {
      if (activityOverviewFrame !== null) return;
      activityOverviewFrame = requestAnimationFrame(() => {
        activityOverviewFrame = null;
        drawActivityOverview();
      });
    }

    function drawActivityOverview() {
      activityOverview.replaceChildren();
      const segments = activitySegmentWeights(activityPhaseSegments(activityEntriesInView()));
      if (!segments.length) {
        const empty = document.createElement('span');
        empty.className = 'activity-overview-empty';
        empty.textContent = 'Timeline appears here';
        activityOverview.appendChild(empty);
        return;
      }
      const track = document.createElement('div');
      track.className = 'activity-overview-track';
      for (const segment of segments) {
        const bar = document.createElement('span');
        bar.className = 'activity-segment kind-' + segment.kind;
        if (segment.weight <= 0) continue;
        bar.style.flexGrow = String(segment.weight);
        bar.title = segment.label + ' · ' + segment.kind + ' · ' + formatActivityDuration(segment.duration);
        track.appendChild(bar);
      }
      activityOverview.appendChild(track);
      syncActivityOverviewScroll();
    }

    function updateActivitySummary() {
      const parts = [];
      const stepCount = activityState.steps.size;
      if (stepCount) parts.push(stepCount + (stepCount === 1 ? " model step" : " model steps"));
      if (activityState.tools) parts.push(activityState.tools + (activityState.tools === 1 ? " tool run" : " tool runs"));
      if (activityState.errors) parts.push(activityState.errors + (activityState.errors === 1 ? " error" : " errors"));
      if (activityState.startedAt) {
        const elapsed = (activityState.finishedAt || Date.now()) - activityState.startedAt;
        if (elapsed >= 1000) parts.push(formatElapsed(elapsed));
      }
      activitySummary.textContent = parts.length ? parts.join(" · ") : "No activity yet";
      let className = "activity-summary";
      if (activityState.finishedWithError) className += " error";
      else if (activityState.finishedAt) className += " done";
      else if (activityState.startedAt) className += " running";
      activitySummary.className = className;
      statTools.textContent = activityState.tools ? String(activityState.tools) : "-";
    }

    function addActivityEntry(kind, title, options = {}) {
      activityEmpty.hidden = true;
      const startedAt = Number(options.startedAt) || Date.now();
      finalizeOpenActivities(startedAt);
      const entry = document.createElement("li");
      entry.className = "activity-entry kind-" + kind;
      entry._activityPhases = [];
      entry._activityTitle = title;
      entry.dataset.startedAt = String(startedAt);
      if (options.status) entry.dataset.status = String(options.status).toLowerCase();

      const marker = document.createElement("span");
      marker.className = "activity-marker";
      marker.textContent = options.marker || activityMarker(kind);
      entry.appendChild(marker);

      const card = document.createElement("div");
      card.className = "activity-card";
      const head = document.createElement("div");
      head.className = "activity-row-head";
      const label = document.createElement("span");
      label.className = "activity-label";
      label.textContent = title;
      const time = document.createElement("span");
      time.className = "activity-time";
      time.textContent = options.time || formatClock();
      const duration = document.createElement("span");
      duration.className = "activity-duration";
      duration.hidden = true;
      head.appendChild(label);
      head.appendChild(time);
      head.appendChild(duration);
      card.appendChild(head);

      const copy = document.createElement("div");
      copy.className = "activity-copy";
      card.appendChild(copy);

      const metaRow = document.createElement("div");
      metaRow.className = "activity-meta-row";
      const status = document.createElement("span");
      status.className = "activity-status";
      status.hidden = !options.status;
      const meta = document.createElement("span");
      meta.className = "activity-meta";
      metaRow.appendChild(status);
      metaRow.appendChild(meta);
      card.appendChild(metaRow);
      entry.appendChild(card);

      activityLog.appendChild(entry);
      beginActivityPhase(entry, kind, startedAt);
      if (options.detail !== undefined || options.command !== undefined) {
        updateActivityEntry(entry, { detail: options.detail, command: options.command });
      }
      if (options.status) updateActivityEntry(entry, { status: options.status, meta: options.meta });
      else if (options.meta) updateActivityEntry(entry, { meta: options.meta });
      if (options.output) updateActivityEntry(entry, { output: options.output });
      updateActivitySummary();
      scheduleActivityLayout();
      return entry;
    }

    function updateActivityEntry(entry, options = {}) {
      if (!entry) return;
      const label = entry.querySelector(".activity-label");
      const copy = entry.querySelector(".activity-copy");
      const status = entry.querySelector(".activity-status");
      const meta = entry.querySelector(".activity-meta");
      if (options.title !== undefined) {
        entry._activityTitle = String(options.title);
        if (label) label.textContent = options.title;
      }
      if (options.detail !== undefined && copy) copy.textContent = options.detail;
      if (options.command !== undefined && copy) {
        const fullCommand = "$ " + String(options.command || "");
        copy.replaceChildren();
        const preview = document.createElement("code");
        preview.className = "activity-command-preview";
        preview.textContent = "$ " + commandPreview(options.command);
        preview.title = fullCommand;
        const full = document.createElement("code");
        full.className = "activity-command-full";
        full.textContent = fullCommand;
        copy.append(preview, full);
        entry.dataset.command = String(options.command || "");
      }
      if (options.status !== undefined && status) {
        status.hidden = !options.status;
        status.textContent = options.status;
        status.className = "activity-status " + activityStatusTone(options.status);
        entry.dataset.status = String(options.status || "").toLowerCase();
        if (activityStatusIsFinished(options.status)) {
          endActivityPhase(entry);
          markActivityEnded(entry);
        }
      }
      if (options.meta !== undefined && meta) {
        meta.textContent = options.meta;
        meta.hidden = !options.meta;
      }
      if (options.output !== undefined) {
        let output = entry.querySelector(".activity-output");
        if (!output) {
          output = document.createElement("div");
          output.className = "activity-output";
          const label = document.createElement("span");
          label.className = "activity-output-label";
          label.textContent = "Output";
          const body = document.createElement("div");
          body.className = "activity-output-text";
          output.append(label, body);
          entry.querySelector(".activity-card").appendChild(output);
        }
        const body = output.querySelector(".activity-output-text");
        const text = String(options.output ?? "");
        const max = 20000;
        if (body) {
          body.textContent = text.length > max ? text.slice(0, max) + "\n\n... output truncated in GUI ..." : text || "(no output)";
        }
      }
      scheduleActivityLayout();
    }

    function addActivity(text, options = {}) {
      if (typeof options === "string") options = { detail: options };
      return addActivityEntry(options.kind || "info", text, options);
    }

    function markActivityStarted() {
      activityState.startedAt = Date.now();
      activityState.finishedAt = 0;
      activityState.finishedWithError = false;
      if (activityTimer) clearInterval(activityTimer);
      activityTimer = setInterval(() => {
        if (!activityState.finishedAt) {
          updateActivitySummary();
          drawActivityOverview();
          scheduleActivityLayout();
        }
      }, 1000);
      updateActivitySummary();
    }

    function markActivityFinished(withError = false) {
      activityState.finishedAt = Date.now();
      activityState.finishedWithError = withError;
      if (activityTimer) {
        clearInterval(activityTimer);
        activityTimer = null;
      }
      updateActivitySummary();
    }

    function resetActivity() {
      activityLog.querySelectorAll(".activity-entry").forEach((entry) => entry.remove());
      activityEmpty.hidden = false;
      activityStepEntries.clear();
      activityToolEntries.clear();
      currentActivityStep = null;
      currentActivityThinking = null;
      currentActivityStepNumber = null;
      activityLiveText = "";
      activityReasoningText = "";
      activityState = {
        steps: new Set(),
        tools: 0,
        errors: 0,
        segments: [],
        startedAt: 0,
        finishedAt: 0,
        finishedWithError: false,
      };
      if (activityTimer) {
        clearInterval(activityTimer);
        activityTimer = null;
      }
      activityScroller.scrollTop = 0;
      activityFollowOutput = true;
      setActivitySpineState(false, false);
      drawActivityOverview();
      updateActivitySummary();
      scheduleActivityLayout();
    }

    function addActivityStep(step) {
      let stepNumber = Number(step);
      if (!Number.isFinite(stepNumber) || stepNumber <= 0) stepNumber = activityState.steps.size + 1;
      if (activityStepEntries.has(stepNumber)) {
        currentActivityStep = activityStepEntries.get(stepNumber);
        currentActivityStepNumber = stepNumber;
        return currentActivityStep;
      }
      activityState.steps.add(stepNumber);
      const entry = addActivityEntry("step", "Step " + stepNumber + " · Planning", {
        marker: String(stepNumber),
        detail: "Model is deciding what to do next…",
        status: "thinking",
        meta: "step " + stepNumber,
      });
      entry.dataset.step = String(stepNumber);
      activityStepEntries.set(stepNumber, entry);
      currentActivityStep = entry;
      currentActivityStepNumber = stepNumber;
      updateActivitySummary();
      return entry;
    }

    function ensureActivityThinking(step, timestamp = Date.now()) {
      if (currentActivityThinking && !currentActivityThinking.dataset.endedAt) return currentActivityThinking;
      const stepNumber = Number(step || currentActivityStepNumber) || activityState.steps.size || 1;
      currentActivityThinking = addActivityEntry('thinking', 'Thinking', {
        detail: 'Step ' + stepNumber + ' is reasoning about the next response.',
        status: 'thinking',
        meta: 'step ' + stepNumber,
        startedAt: timestamp,
      });
      return currentActivityThinking;
    }

    function finishActivityThinking(timestamp = Date.now()) {
      if (!currentActivityThinking) return;
      endActivityPhase(currentActivityThinking, timestamp);
      updateActivityEntry(currentActivityThinking, { status: 'complete' });
      currentActivityThinking = null;
    }

    function updateActivityStep(step, options = {}) {
      const stepNumber = Number(step);
      let entry = Number.isFinite(stepNumber) ? activityStepEntries.get(stepNumber) : currentActivityStep;
      if (!entry) entry = addActivityStep(step);
      updateActivityEntry(entry, options);
      return entry;
    }

    function addToolActivity(event) {
      const toolCallId = event.tool_call_id || "";
      finishActivityThinking(activityEventTime(event));
      if (toolCallId && activityToolEntries.has(toolCallId)) return activityToolEntries.get(toolCallId);
      const entry = addActivityEntry("tool", "Bash command", {
        marker: "→",
        command: event.command || "",
        status: "running",
        meta: currentActivityStepNumber ? "step " + currentActivityStepNumber : "",
      });
      entry.dataset.startedAt = String(activityEventTime(event));
      entry._activityPhases = [];
      beginActivityPhase(entry, "tool", activityEventTime(event));
      if (toolCallId) {
        entry.dataset.toolCallId = toolCallId;
        activityToolEntries.set(toolCallId, entry);
      }
      activityState.tools += 1;
      if (currentActivityStepNumber) {
        updateActivityStep(currentActivityStepNumber, { status: "running" });
      }
      updateActivitySummary();
      return entry;
    }

    function findToolActivity(event) {
      const toolCallId = event.tool_call_id || "";
      if (toolCallId && activityToolEntries.has(toolCallId)) return activityToolEntries.get(toolCallId);
      const running = activityLog.querySelectorAll(".activity-entry.kind-tool[data-status='running']");
      return running.length ? running[running.length - 1] : null;
    }

    function updateToolActivity(event) {
      let entry = findToolActivity(event);
      if (!entry) entry = addToolActivity(event);
      const returncode = event.returncode;
      const failed = returncode != null && Number(returncode) !== 0;
      const status = returncode == null ? "finished" : "exit " + returncode;
      const output = String(event.output || "");
      const endedAt = activityEventTime(event);
      endActivityPhase(entry, endedAt);
      entry.dataset.endedAt = String(endedAt);
      const startedAt = Number(entry.dataset.startedAt || 0);
      const elapsed = startedAt ? endedAt - startedAt : 0;
      const timing = elapsed >= 100 ? " · " + formatElapsed(elapsed) : "";
      const meta = "exit " + (returncode ?? "?") + timing + " · " + formatTokens(output.length) + " chars";
      // Store the complete result on the activity entry.  CSS hides it in the
      // compact rail and reveals it when Activity Expand takes over the page.
      updateActivityEntry(entry, { status, meta, output });
      if (currentActivityStepNumber) {
        updateActivityStep(currentActivityStepNumber, { status: failed ? "tool failed" : "tool done" });
      }
      return entry;
    }

    function finalizeActivity(event) {
      const exitStatus = event.exit_status || "done";
      finishActivityThinking(activityEventTime(event));
      activityStepEntries.forEach((entry) => endActivityPhase(entry, activityEventTime(event)));
      activityToolEntries.forEach((entry) => endActivityPhase(entry, activityEventTime(event)));
      const successful = exitStatus === "Submitted";
      const stopped = exitStatus === "Stopped";
      const withError = !successful && !stopped;
      if (activityState.steps.size) {
        const lastStep = Math.max(...activityState.steps);
        updateActivityStep(lastStep, {
          title: "Step " + lastStep + " · " + (successful ? "Complete" : stopped ? "Stopped" : "Failed"),
          status: successful ? "complete" : stopped ? "stopped" : "failed",
        });
      }
      const meta = [
        event.n_calls != null ? event.n_calls + " calls" : "",
        event.calls_without_cost ? event.calls_without_cost + " without cost data" : "",
        event.cost != null ? "$" + Number(event.cost).toFixed(4) : "",
        (event.total_tokens ?? event.tokens) != null ? formatTokens(event.total_tokens ?? event.tokens) + " tokens" : "",
      ].filter(Boolean).join(" · ");
      const detail = event.submission
        ? "Submission: " + previewText(event.submission, 180)
        : (event.message ? previewText(event.message, 180) : "Exit status: " + exitStatus);
      addActivityEntry(withError ? "error" : "done", "Run finished · " + exitStatus, {
        marker: successful ? "✓" : stopped ? "■" : "!",
        detail,
        status: successful ? "submitted" : stopped ? "stopped" : "failed",
        meta,
      });
      finalizeOpenActivities(activityEventTime(event));
      markActivityFinished(withError);
      scheduleActivityLayout();
    }

    function clearFeed() {
      feed.innerHTML = "";
      currentAssistant = null;
      currentReasoning = null;
      pendingAssistantMessage = null;
      followOutput = true;
      updateScrollButtons();
    }

    function addRunMarker(kind, label, meta = "") {
      const marker = document.createElement("div");
      marker.className = "run-marker " + kind;
      const text = document.createElement("span");
      text.className = "run-marker-label";
      text.textContent = label;
      marker.appendChild(text);
      if (meta) {
        const detail = document.createElement("span");
        detail.className = "run-marker-meta";
        detail.textContent = meta;
        marker.appendChild(detail);
      }
      feed.appendChild(marker);
      return marker;
    }

    function commandPreview(command) {
      const lines = String(command || "").split(/\r?\n/);
      const nonEmpty = lines.filter((line) => line.trim().length > 0);
      const first = (nonEmpty[0] || "").trim().replace(/\s+/g, " ");
      const remaining = Math.max(0, nonEmpty.length - 1);
      return (first || "(empty command)") + (remaining ? "  +" + remaining + " more line" + (remaining === 1 ? "" : "s") : "");
    }

    function flashActivityEntry(entry) {
      entry.classList.remove('activity-highlight');
      void entry.offsetWidth;
      entry.classList.add('activity-highlight');
      entry.addEventListener('animationend', () => entry.classList.remove('activity-highlight'), { once: true });
    }

    function flashActivityEntryWhenSettled(entry) {
      const startedAt = performance.now();
      let previousScrollTop = activityScroller.scrollTop;
      let stableFrames = 0;
      function waitForScrollToSettle() {
        if (!entry.isConnected) return;
        const scrollTop = activityScroller.scrollTop;
        const elapsed = performance.now() - startedAt;
        stableFrames = Math.abs(scrollTop - previousScrollTop) < 1 ? stableFrames + 1 : 0;
        previousScrollTop = scrollTop;
        if ((stableFrames >= 4 && elapsed >= 100) || elapsed >= 2500) {
          flashActivityEntry(entry);
          return;
        }
        requestAnimationFrame(waitForScrollToSettle);
      }
      requestAnimationFrame(waitForScrollToSettle);
    }

    function openToolActivity(toolCallId) {
      setActivityExpanded(true);
      const entry = toolCallId ? activityToolEntries.get(String(toolCallId)) : null;
      if (!entry) return;
      // Wait for the expanded layout pass, then reveal and focus the matching
      // activity entry.  This is the only place command details are shown.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        entry.hidden = false;
        entry.scrollIntoView({ block: "center", behavior: "smooth" });
        flashActivityEntryWhenSettled(entry);
      }));
    }

    function makeToolCall(command, toolCallId = "") {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "tool-call";
      row.dataset.command = command || "";
      if (toolCallId) row.dataset.toolCallId = String(toolCallId);
      row.title = "Open command details in Activity";
      row.setAttribute("aria-label", "Open command details in Activity: " + commandPreview(command));

      const icon = document.createElement("span");
      icon.className = "tool-call-icon";
      icon.textContent = "\u25b8";
      const prompt = document.createElement("span");
      prompt.className = "tool-call-prompt";
      prompt.textContent = "$";
      const text = document.createElement("span");
      text.className = "tool-call-command";
      text.textContent = commandPreview(command);
      const status = document.createElement("span");
      status.className = "tool-call-status running";
      status.textContent = "running";

      row.append(icon, prompt, text, status);
      row.addEventListener("click", () => openToolActivity(toolCallId));
      feed.appendChild(row);
      return row;
    }

    function findToolCall(event) {
      const rows = Array.from(feed.querySelectorAll(".tool-call"));
      const toolCallId = event.tool_call_id == null ? "" : String(event.tool_call_id);
      if (toolCallId) {
        for (let i = rows.length - 1; i >= 0; i--) {
          if (rows[i].dataset.toolCallId === toolCallId) return rows[i];
        }
      }
      const command = event.command || "";
      for (let i = rows.length - 1; i >= 0; i--) {
        if (rows[i].dataset.command === command) return rows[i];
      }
      return null;
    }

    function updateToolCall(event) {
      let row = findToolCall(event);
      if (!row) row = makeToolCall(event.command || "", event.tool_call_id);
      const status = row.querySelector(".tool-call-status");
      if (!status) return row;
      if (event.returncode == null) {
        status.textContent = "done";
        status.className = "tool-call-status";
      } else {
        status.textContent = "exit " + event.returncode;
        status.className = "tool-call-status " + (Number(event.returncode) === 0 ? "ok" : "error");
      }
      return row;
    }

    function makeCard(kind, role, meta = "") {
      const card = document.createElement("div");
      card.className = "card " + kind;
      if (kind === "assistant") {
        // Assistant text is part of the transcript, not a widget.  Model
        // metadata (step, cost, tokens) is kept in the Activity ledger.
        card.setAttribute("aria-label", role || "Agent");
      } else if (kind !== "user") {
        const head = document.createElement("div");
        head.className = "card-head";
        const label = document.createElement("span");
        label.className = "role";
        label.textContent = role;
        head.appendChild(label);
        if (meta) {
          const m = document.createElement("span");
          m.className = "meta";
          m.textContent = meta;
          head.appendChild(m);
        }
        card.appendChild(head);
      }
      feed.appendChild(card);
      return card;
    }

    function escapeMarkdownHtml(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\x22/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function safeMarkdownUrl(value) {
      const url = String(value || '').trim();
      return /^(https?:\/\/|mailto:|#|\/(?!\/)|\.\.?\/)/i.test(url) ? url : '';
    }

    function renderMarkdownInline(value) {
      const tokens = [];
      const token = (html) => {
        tokens.push(html);
        return '\u0000' + (tokens.length - 1) + '\u0000';
      };
      let text = escapeMarkdownHtml(value);
      text = text.replace(/`([^`\n]+)`/g, (match, code) => token('<code>' + code + '</code>'));
      text = text.replace(/\[([^\]\n]+)\]\(([^\s)]+)\)/g, (match, label, url) => {
        const href = safeMarkdownUrl(url);
        return href
          ? token('<a href=\x22' + href + '\x22 target=\x22_blank\x22 rel=\x22noopener noreferrer\x22>' + label + '</a>')
          : label;
      });
      text = text.replace(/\*\*([^*\n]+)\*\*|__([^_\n]+)__/g, (match, stars, underscores) => {
        return '<strong>' + (stars || underscores) + '</strong>';
      });
      text = text.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
      text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
      text = text.replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, '$1<em>$2</em>');
      return text.replace(/\u0000(\d+)\u0000/g, (match, index) => tokens[Number(index)]);
    }

    function markdownToHtml(markdown) {
      const lines = String(markdown == null ? '' : markdown).replace(/\r\n?/g, '\n').split('\n');
      const html = [];
      let paragraph = [];
      let quote = [];
      let listType = '';
      let codeLanguage = '';
      let codeLines = null;

      function flushParagraph() {
        if (!paragraph.length) return;
        html.push('<p>' + renderMarkdownInline(paragraph.join('\n')).replace(/\n/g, '<br>') + '</p>');
        paragraph = [];
      }

      function flushQuote() {
        if (!quote.length) return;
        html.push('<blockquote><p>' + renderMarkdownInline(quote.join('\n')).replace(/\n/g, '<br>') + '</p></blockquote>');
        quote = [];
      }

      function closeList() {
        if (!listType) return;
        html.push('</' + listType + '>');
        listType = '';
      }

      function flushCode() {
        if (codeLines === null) return;
        const language = codeLanguage ? ' class=\x22language-' + codeLanguage + '\x22' : '';
        html.push('<pre><code' + language + '>' + escapeMarkdownHtml(codeLines.join('\n')) + '</code></pre>');
        codeLines = null;
        codeLanguage = '';
      }

      function markdownTableCells(line) {
        let value = String(line).trim();
        if (value.startsWith('|')) value = value.slice(1);
        if (value.endsWith('|')) value = value.slice(0, -1);
        return value.split('|').map((cell) => cell.trim());
      }

      function isMarkdownTableDivider(line) {
        if (line === undefined || !line.includes('|')) return false;
        const cells = markdownTableCells(line);
        return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
      }

      for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
        const line = lines[lineIndex];
        const fence = line.match(/^\s*```\s*([\w.+-]*)\s*$/);
        if (codeLines !== null) {
          if (fence) flushCode();
          else codeLines.push(line);
          continue;
        }
        if (fence) {
          flushParagraph();
          flushQuote();
          closeList();
          codeLanguage = fence[1];
          codeLines = [];
          continue;
        }
        if (!line.trim()) {
          flushParagraph();
          flushQuote();
          closeList();
          continue;
        }
        if (line.includes('|') && isMarkdownTableDivider(lines[lineIndex + 1])) {
          flushParagraph();
          flushQuote();
          closeList();
          const headers = markdownTableCells(line);
          html.push('<table>');
          html.push('<thead><tr>' + headers.map((cell) => '<th>' + renderMarkdownInline(cell) + '</th>').join('') + '</tr></thead>');
          html.push('<tbody>');
          lineIndex += 1;
          while (lineIndex + 1 < lines.length) {
            const row = lines[lineIndex + 1];
            if (!row.trim() || !row.includes('|')) break;
            lineIndex += 1;
            const cells = markdownTableCells(row);
            html.push('<tr>' + headers.map((_, cellIndex) => '<td>' + renderMarkdownInline(cells[cellIndex] || '') + '</td>').join('') + '</tr>');
          }
          html.push('</tbody></table>');
          continue;
        }
        const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
        if (heading) {
          flushParagraph();
          flushQuote();
          closeList();
          const level = heading[1].length;
          html.push('<h' + level + '>' + renderMarkdownInline(heading[2]) + '</h' + level + '>');
          continue;
        }
        if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
          flushParagraph();
          flushQuote();
          closeList();
          html.push('<hr>');
          continue;
        }
        const quoted = line.match(/^\s{0,3}>\s?(.*)$/);
        if (quoted) {
          flushParagraph();
          closeList();
          quote.push(quoted[1]);
          continue;
        }
        const unordered = line.match(/^\s*[-+*]\s+(.+)$/);
        const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
        if (unordered || ordered) {
          flushParagraph();
          flushQuote();
          const type = ordered ? 'ol' : 'ul';
          if (listType !== type) {
            closeList();
            html.push('<' + type + '>');
            listType = type;
          }
          html.push('<li>' + renderMarkdownInline((ordered || unordered)[1]) + '</li>');
          continue;
        }
        flushQuote();
        closeList();
        paragraph.push(line);
      }
      flushCode();
      flushParagraph();
      flushQuote();
      closeList();
      return html.join('\n');
    }

    function markdownSource(content) {
      return content && content.dataset.markdown !== undefined
        ? content.dataset.markdown
        : String(content ? content.textContent : '');
    }

    function setMarkdownContent(content, value) {
      const source = String(value == null ? '' : value);
      content.dataset.markdown = source;
      content.classList.add('markdown');
      content.innerHTML = markdownToHtml(source);
    }

    function appendText(card, text) {
      let content = card.querySelector('.content');
      if (!content) {
        content = document.createElement('div');
        content.className = 'content';
        card.appendChild(content);
      }
      const source = markdownSource(content) + String(text == null ? '' : text);
      if (card.classList.contains('user') || card.classList.contains('assistant')) {
        setMarkdownContent(content, source);
      } else {
        content.dataset.markdown = source;
        content.textContent = source;
      }
      return content;
    }

    function beginUserEdit(card) {
      if (card.querySelector('.user-editor')) return;
      const sourceId = selectedSessionId;
      const conversation = conversations.find((item) => conversationId(item) === sourceId);
      if (!sourceId || !conversation || !conversation.done) {
        window.alert('Wait for this conversation to finish before creating a branch.');
        return;
      }
      const content = card.querySelector('.content');
      const original = markdownSource(content);
      const gallery = card.querySelector('.card-images');
      card.classList.add('editing');
      content.hidden = true;
      if (gallery) gallery.hidden = true;
      const editor = document.createElement('div');
      editor.className = 'user-editor';
      const textarea = document.createElement('textarea');
      textarea.value = original;
      textarea.setAttribute('aria-label', 'Edit user message for a new branch');
      const actions = document.createElement('div');
      actions.className = 'user-editor-actions';
      const cancel = document.createElement('button');
      cancel.type = 'button';
      cancel.className = 'secondary';
      cancel.textContent = 'Cancel';
      const confirm = document.createElement('button');
      confirm.type = 'button';
      confirm.textContent = 'Create branch';
      actions.append(cancel, confirm);
      editor.append(textarea, actions);
      card.appendChild(editor);
      function resizeEditor() {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(Math.max(textarea.scrollHeight, 64), 320) + 'px';
      }
      textarea.addEventListener('input', resizeEditor);
      requestAnimationFrame(resizeEditor);
      textarea.addEventListener('contextmenu', (event) => event.stopPropagation());
      function closeEditor() {
        editor.remove();
        card.classList.remove('editing');
        content.hidden = false;
        if (gallery) gallery.hidden = false;
      }
      async function createBranch() {
        const task = textarea.value.trim();
        if (!task) {
          textarea.focus();
          return;
        }
        if (task === original.trim()) {
          closeEditor();
          return;
        }
        confirm.disabled = true;
        cancel.disabled = true;
        setStatus('creating branch', 'running');
        try {
          const data = await conversationRequest(
            '/api/conversations/' + encodeURIComponent(sourceId) + '/branch',
            {
              method: 'POST',
              body: JSON.stringify({ task, turn_index: Number(card.dataset.turnIndex) }),
            },
          );
          await loadConversations();
          await selectConversation(data.session_id);
        } catch (err) {
          confirm.disabled = false;
          cancel.disabled = false;
          setStatus('branch failed', 'error');
          window.alert(String(err.message || err));
        }
      }
      cancel.addEventListener('click', closeEditor);
      confirm.addEventListener('click', createBranch);
      textarea.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          closeEditor();
        } else if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          createBranch();
        }
      });
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    }

    function addUserCard(task, images) {
      const card = makeCard("user", "");
      card.setAttribute("aria-label", "Your message");
      card.dataset.turnIndex = String(feed.querySelectorAll(".card.user").length - 1);
      card.title = "Right-click to edit and create a branch";
      card.addEventListener("contextmenu", (event) => {
        const conversation = conversations.find((item) => conversationId(item) === selectedSessionId);
        openContextMenu(event, [{
          label: "Edit and create branch",
          disabled: !conversation || !conversation.done,
          action: () => beginUserEdit(card),
        }]);
      });
      appendText(card, task);
      const imageUrls = Array.isArray(images) ? images.filter(Boolean) : [];
      if (imageUrls.length) {
        const gallery = document.createElement("div");
        gallery.className = "card-images";
        imageUrls.forEach((url) => {
          const link = document.createElement("a");
          link.href = String(url);
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.title = "Click to preview image";
          link.addEventListener("click", (event) => {
            event.preventDefault();
            openImagePreview(url);
          });
          const img = document.createElement("img");
          img.src = String(url);
          img.alt = "Attached image";
          link.appendChild(img);
          gallery.appendChild(link);
        });
        card.appendChild(gallery);
      }
      if (pendingTreeNavigation
        && pendingTreeNavigation.sessionId === selectedSessionId
        && pendingTreeNavigation.turnIndex === Number(card.dataset.turnIndex)) {
        pendingTreeNavigation = null;
        requestAnimationFrame(() => highlightUserCard(card));
      }
      return card;
    }

    function renderPendingImages() {
      composerAttachments.replaceChildren();
      composerAttachments.hidden = pendingImages.length === 0;
      pendingImages.forEach((url, index) => {
        const item = document.createElement("div");
        item.className = "composer-attachment";
        const img = document.createElement("img");
        img.src = url;
        img.alt = "Attached image " + (index + 1);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "\u00d7";
        remove.title = "Remove image";
        remove.setAttribute("aria-label", "Remove image " + (index + 1));
        remove.addEventListener("click", () => {
          pendingImages.splice(index, 1);
          renderPendingImages();
        });
        item.appendChild(img);
        item.appendChild(remove);
        composerAttachments.appendChild(item);
      });
    }

    function readImageFile(file) {
      return new Promise((resolve, reject) => {
        if (!file.type || !file.type.startsWith("image/")) {
          reject(new Error("Only image files can be attached."));
          return;
        }
        if (file.size > MAX_IMAGE_BYTES) {
          reject(new Error("Image " + file.name + " is too large; the limit is 20 MB."));
          return;
        }
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("Could not read image " + file.name + "."));
        reader.readAsDataURL(file);
      });
    }

    async function addImageFiles(files) {
      const selected = Array.from(files || []).filter((file) => file && file.type && file.type.startsWith("image/"));
      if (!selected.length) return;
      const room = MAX_IMAGES - pendingImages.length;
      if (room <= 0) {
        setStatus("image limit reached", "error");
        return;
      }
      for (const file of selected.slice(0, room)) {
        try {
          pendingImages.push(await readImageFile(file));
        } catch (err) {
          setStatus(String(err.message || err), "error");
        }
      }
      renderPendingImages();
      taskEl.focus();
    }

    function ensureAssistantCard() {
      if (!currentAssistant) {
        currentAssistant = makeCard("assistant", "mini-SWE-agent");
      }
      return currentAssistant;
    }

    function normalizeCommandForComparison(value) {
      return String(value == null ? "" : value)
        .replace(/\r\n?/g, "\n")
        .split("\n")
        .map((line) => line.replace(/[ \t]+$/g, ""))
        .join("\n")
        .trim()
        .replace(/^\$\s+/, "");
    }

    function commandTextMatches(candidate, target) {
      return Boolean(target) && normalizeCommandForComparison(candidate) === target;
    }

    function tidyAssistantContent(content) {
      return String(content == null ? "" : content)
        .replace(/\r\n?/g, "\n")
        .split("\n")
        .map((line) => line.replace(/[ \t]+$/g, ""))
        .join("\n")
        .replace(/\n{3,}/g, "\n\n")
        .replace(/[ \t]{2,}/g, " ")
        .replace(/[ \t]+([.,!?;:])/g, "$1")
        .trim();
    }

    /**
     * Remove a command from an assistant message while preserving its prose.
     *
     * Models frequently repeat the command they are about to execute in their
     * textual response (often inside a fenced code block).  The tool-call row
     * is the authoritative representation of that command, so we remove matching
     * command fragments from the assistant text and leave only the explanation.
     */
    function stripCommandFromAssistantContent(content, command) {
      const original = String(content == null ? "" : content);
      const target = normalizeCommandForComparison(command);
      const result = { content: original, removed: false };
      if (!target) return result;

      let text = original;

      // Fenced code blocks can contain the command as their whole body or as one
      // line among surrounding comments/examples.
      text = text.replace(/```([^\n`]*)\r?\n?([\s\S]*?)```/g, (match, language, body) => {
        if (commandTextMatches(body, target)) {
          result.removed = true;
          return "";
        }
        const lines = body.split(/\r?\n/);
        const kept = lines.filter((line) => {
          if (commandTextMatches(line, target)) {
            result.removed = true;
            return false;
          }
          return true;
        });
        if (kept.length === lines.length) return match;
        const keptBody = kept.join("\n").trim();
        return keptBody ? "```" + language + "\n" + keptBody + "\n```" : "";
      });

      // Inline code spans are common in explanations such as
      // "I will run `sed -n ...` to inspect the file".
      text = text.replace(/`([^`\n]+)`/g, (match, body) => {
        if (commandTextMatches(body, target)) {
          result.removed = true;
          return "";
        }
        return match;
      });

      // Also handle plain command-only lines outside fenced blocks.
      const lines = text.split("\n");
      const keptLines = lines.filter((line) => {
        if (commandTextMatches(line, target)) {
          result.removed = true;
          return false;
        }
        return true;
      });
      text = keptLines.join("\n");

      // Finally remove raw occurrences embedded in prose.  Keep replacing until
      // every duplicate has been removed.
      let occurrence = text.indexOf(target);
      while (occurrence !== -1) {
        result.removed = true;
        text = text.slice(0, occurrence) + text.slice(occurrence + target.length);
        occurrence = text.indexOf(target);
      }

      result.content = result.removed ? tidyAssistantContent(text) : original;
      return result;
    }

    function hasVisibleReasoning(card) {
      const reasoning = card && card.querySelector(".reasoning .reasoning-body");
      return Boolean(reasoning && reasoning.textContent.trim());
    }

    function mergeMatchingCommandIntoPendingAssistant(command) {
      if (!pendingAssistantMessage || !pendingAssistantMessage.card) return false;
      const card = pendingAssistantMessage.card;
      if (!card.isConnected) {
        pendingAssistantMessage = null;
        return false;
      }
      const contentEl = card.querySelector(".content");
      if (!contentEl) return false;
      const merged = stripCommandFromAssistantContent(markdownSource(contentEl), command);
      if (!merged.removed) return false;

      setMarkdownContent(contentEl, merged.content);
      pendingAssistantMessage.content = merged.content;
      card.classList.add("command-merged");
      if (!merged.content && !hasVisibleReasoning(card)) {
        // The assistant message contained nothing but the command.  Show only
        // the tool-call row, which is the preferred representation.
        card.hidden = true;
        card.classList.add("command-only");
      }
      return true;
    }

    function flushPendingAssistantMessage() {
      // Associate a completed assistant message with the next tool calls.  Once
      // a non-adjacent event arrives, no further command de-duplication applies.
      pendingAssistantMessage = null;
    }

    function lastVisibleAssistantMessageContent() {
      const cards = feed.querySelectorAll(".card.assistant:not([hidden])");
      for (let i = cards.length - 1; i >= 0; i--) {
        const content = cards[i].querySelector(".content");
        const text = content ? markdownSource(content).trim() : "";
        if (text) return text;
      }
      return "";
    }

    function isRepeatedAssistantSubmission(submission) {
      // The agent emits the final assistant message and then repeats it as the
      // submission in the done event.  Render that answer only once.
      const text = String(submission == null ? "" : submission).trim();
      return Boolean(text) && lastVisibleAssistantMessageContent() === text;
    }

    function queueAssistantMessageForMerge(card, content, step) {
      const contentEl = card && card.querySelector(".content");
      pendingAssistantMessage = {
        card,
        content: contentEl ? markdownSource(contentEl) : String(content == null ? "" : content),
        step,
      };
    }

    function conversationId(conversation) {
      return conversation.session_id || conversation.id || "";
    }

    async function copyConversationId(id) {
      if (!id) return;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(id);
          return;
        }
        throw new Error("Clipboard API unavailable");
      } catch (error) {
        const textarea = document.createElement("textarea");
        textarea.value = id;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        const copied = document.execCommand("copy");
        textarea.remove();
        if (!copied) console.error("Failed to copy conversation ID", error);
      }
    }

    function conversationWorkspacePath(conversation) {
      return (conversation && (conversation.workspace || conversation.cwd || conversation.working_directory)) || defaultWorkspace || "";
    }

    function setConversationWorkspace(conversation) {
      const workspace = conversationWorkspacePath(conversation);
      conversationWorkspace.textContent = workspace;
      conversationWorkspace.title = workspace;
    }

    function rememberSession(sessionId) {
      if (sessionId) {
        sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
        history.replaceState(null, "", "#" + encodeURIComponent(sessionId));
      } else {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        history.replaceState(null, "", location.pathname + location.search);
      }
    }

    function rememberedSessionId() {
      return decodeURIComponent(location.hash.replace(/^#/, "")) || sessionStorage.getItem(SESSION_STORAGE_KEY) || "";
    }

    function conversationStatus(conversation) {
      if (!conversation) return "idle";
      return conversation.status || (conversation.done ? (conversation.exit_status || "done") : "running");
    }

    function conversationStateFromStatus(status, done) {
      if (!done) return "running";
      const normalized = String(status || "").trim().toLowerCase();
      if (["submitted", "complete", "completed", "success", "succeeded"].includes(normalized)) return "submitted";
      if (["stopped", "stop", "cancelled", "canceled"].includes(normalized)) return "stopped";
      if (["interrupted", "interrupt"].includes(normalized)) return "interrupted";
      return "failed";
    }

    function conversationState(conversation) {
      if (!conversation) return "";
      if (CONVERSATION_STATES.includes(conversation.state)) return conversation.state;
      return conversationStateFromStatus(conversationStatus(conversation), Boolean(conversation.done));
    }

    function applyConversationStatus(conversation) {
      if (!conversation) return;
      const status = conversationStatus(conversation);
      setStatus(status, statusClass(conversationState(conversation)));
      const running = Boolean(!conversation.done && String(status).toLowerCase() === "running");
      setComposerRunning(running);
    }

    function statusClass(state) {
      return CONVERSATION_STATES.includes(state) ? state : "";
    }

    function formatConversationTime(timestamp) {
      if (!timestamp) return "";
      return new Date(Number(timestamp) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function closeImagePreview() {
      imagePreview.hidden = true;
      imagePreviewImage.src = '';
      document.body.classList.remove('preview-open');
    }

    function openImagePreview(url) {
      closeContextMenu();
      imagePreviewImage.src = String(url);
      imagePreviewOpen.href = String(url);
      imagePreview.hidden = false;
      document.body.classList.add('preview-open');
      imagePreviewClose.focus();
    }

    imagePreviewClose.addEventListener('click', closeImagePreview);
    imagePreview.addEventListener('click', (event) => {
      if (event.target === imagePreview) closeImagePreview();
    });

    function closeContextMenu() {
      contextMenu.hidden = true;
      contextMenu.replaceChildren();
    }

    function openContextMenu(event, items) {
      event.preventDefault();
      event.stopPropagation();
      closeContextMenu();
      for (const item of items) {
        const button = document.createElement('button');
        button.type = 'button';
        button.role = 'menuitem';
        button.textContent = item.label;
        button.disabled = Boolean(item.disabled);
        if (item.danger) button.className = 'danger';
        button.addEventListener('click', () => {
          closeContextMenu();
          Promise.resolve(item.action()).catch((err) => window.alert(String(err.message || err)));
        });
        contextMenu.appendChild(button);
      }
      contextMenu.hidden = false;
      contextMenu.style.left = Math.max(8, Math.min(event.clientX, window.innerWidth - contextMenu.offsetWidth - 8)) + 'px';
      contextMenu.style.top = Math.max(8, Math.min(event.clientY, window.innerHeight - contextMenu.offsetHeight - 8)) + 'px';
    }

    document.addEventListener('click', closeContextMenu);
    document.addEventListener('contextmenu', closeContextMenu);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        closeContextMenu();
        closeImagePreview();
      }
    });
    window.addEventListener('resize', closeContextMenu);
    window.addEventListener('blur', closeContextMenu);

    async function conversationRequest(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
      return data;
    }

    async function updateConversationMetadata(id, changes) {
      const data = await conversationRequest('/api/conversations/' + encodeURIComponent(id), {
        method: 'PATCH',
        body: JSON.stringify(changes),
      });
      const conversation = conversations.find((item) => conversationId(item) === id);
      if (conversation) Object.assign(conversation, data.conversation);
      if (id === selectedSessionId) {
        conversationLabel.textContent = (data.conversation && data.conversation.title) || 'Untitled conversation';
      }
      renderConversationList();
      if ('pinned' in changes) await loadConversations();
    }

    function showConversationContextMenu(event, id) {
      const conversation = conversations.find((item) => conversationId(item) === id);
      if (!conversation) return;
      openContextMenu(event, [
        {
          label: 'Rename',
          action: async () => {
            const title = window.prompt('Rename conversation', conversation.title || conversation.task || '');
            if (title !== null && title.trim()) await updateConversationMetadata(id, { title: title.trim() });
          },
        },
        {
          label: conversation.pinned ? 'Unpin' : 'Pin to top',
          action: () => updateConversationMetadata(id, { pinned: !conversation.pinned }),
        },
        {
          label: 'Delete',
          danger: true,
          action: async () => {
            if (!window.confirm('Delete this conversation? Child branches will be kept.')) return;
            await conversationRequest('/api/conversations/' + encodeURIComponent(id), { method: 'DELETE' });
            if (id === selectedSessionId) startNewConversation();
            await loadConversations();
          },
        },
      ]);
    }

    function currentConversationTree() {
      const byId = new Map(conversations.map((conversation) => [conversationId(conversation), conversation]));
      let root = byId.get(selectedSessionId);
      if (!root) return { roots: [], count: 0 };
      const ancestors = new Set();
      while (root.parent_id && byId.has(root.parent_id) && !ancestors.has(conversationId(root))) {
        ancestors.add(conversationId(root));
        root = byId.get(root.parent_id);
      }
      const family = [];
      const familyIds = new Set();
      function gather(id) {
        if (familyIds.has(id) || !byId.has(id)) return;
        familyIds.add(id);
        family.push(byId.get(id));
        for (const conversation of conversations) {
          if (conversation.parent_id === id) gather(conversationId(conversation));
        }
      }
      gather(conversationId(root));
      family.sort((a, b) => Number(a.created_at || 0) - Number(b.created_at || 0));
      const roots = [];
      const nodeMap = new Map();
      let count = 0;
      for (const conversation of family) {
        const id = conversationId(conversation);
        const parent = conversation.parent_id && familyIds.has(conversation.parent_id)
          ? byId.get(conversation.parent_id)
          : null;
        const parentId = parent ? conversationId(parent) : '';
        const turns = Array.isArray(conversation.turns) && conversation.turns.length
          ? conversation.turns
          : [{ task: conversation.task || conversation.title || 'Untitled conversation' }];
        const startTurn = parent ? Math.max(0, Number(conversation.branch_turn_index ?? 0)) : 0;
        if (parent) {
          for (let index = 0; index < startTurn; index++) {
            const shared = nodeMap.get(parentId + ':' + index);
            if (shared) nodeMap.set(id + ':' + index, shared);
          }
        }
        let previous = null;
        for (let index = startTurn; index < turns.length; index++) {
          const node = {
            id: id + ':' + index,
            sessionId: id,
            turnIndex: index,
            task: String(turns[index].task || '(image input)'),
            branchTitle: conversation.title || conversation.task || 'Untitled conversation',
            state: conversationState(conversation),
            children: [],
          };
          nodeMap.set(node.id, node);
          count++;
          if (previous) {
            previous.children.push(node);
          } else {
            const anchor = parent && startTurn > 0 ? nodeMap.get(parentId + ':' + (startTurn - 1)) : null;
            (anchor ? anchor.children : roots).push(node);
          }
          previous = node;
        }
      }
      return { roots, count };
    }

    function drawConversationTreeEdges() {
      mainConversationTree.querySelector('.main-tree-connectors')?.remove();
      if (mainTreeView.hidden || !conversationTreeEdges.length) return;
      const bounds = mainConversationTree.getBoundingClientRect();
      const width = Math.max(mainConversationTree.clientWidth, mainConversationTree.scrollWidth);
      const height = mainConversationTree.scrollHeight;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.classList.add('main-tree-connectors');
      svg.setAttribute('width', width);
      svg.setAttribute('height', height);
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.setAttribute('aria-hidden', 'true');
      for (const edge of conversationTreeEdges) {
        const parentDot = conversationTreeDots.get(edge.parent);
        const childDot = conversationTreeDots.get(edge.child);
        if (!parentDot || !childDot) continue;
        const parent = parentDot.getBoundingClientRect();
        const child = childDot.getBoundingClientRect();
        const from = {
          x: parent.left + parent.width / 2 - bounds.left + mainTreeView.scrollLeft,
          y: parent.top + parent.height / 2 - bounds.top + mainTreeView.scrollTop,
        };
        const to = {
          x: child.left + child.width / 2 - bounds.left + mainTreeView.scrollLeft,
          y: child.top + child.height / 2 - bounds.top + mainTreeView.scrollTop,
        };
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'main-tree-edge');
        path.setAttribute('d', `M ${from.x} ${from.y} L ${to.x} ${to.y}`);
        svg.appendChild(path);
      }
      mainConversationTree.prepend(svg);
    }

    function scheduleConversationTreeEdges() {
      if (conversationTreeEdgesFrame !== null) cancelAnimationFrame(conversationTreeEdgesFrame);
      conversationTreeEdgesFrame = requestAnimationFrame(() => {
        conversationTreeEdgesFrame = null;
        drawConversationTreeEdges();
      });
    }

    window.addEventListener('resize', scheduleConversationTreeEdges);

    function renderConversationTree() {
      const { roots, count } = currentConversationTree();
      conversationTreeEdges = [];
      conversationTreeDots = new Map();
      mainConversationTree.replaceChildren();
      treeCount.textContent = count + (count === 1 ? ' input' : ' inputs');
      if (!roots.length) {
        mainConversationTree.textContent = 'Select a conversation to view its branches.';
        return;
      }
      function mainNode(node) {
        const family = document.createElement('div');
        family.className = 'main-tree-family';
        const row = document.createElement('div');
        row.className = 'main-tree-node-row';
        const dot = document.createElement('span');
        dot.className = 'main-tree-dot' + (node.sessionId === selectedSessionId ? ' active' : '');
        dot.setAttribute('aria-hidden', 'true');
        conversationTreeDots.set(node.id, dot);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'main-tree-node' + (node.sessionId === selectedSessionId ? ' active' : '');
        const task = document.createElement('span');
        task.className = 'main-tree-node-task';
        task.textContent = node.task;
        const meta = document.createElement('span');
        meta.className = 'main-tree-node-meta';
        meta.textContent = node.branchTitle + ' · input ' + (node.turnIndex + 1);
        button.append(task, meta);
        button.addEventListener('click', () => navigateToTreeNode(node));
        row.append(dot, button);
        family.appendChild(row);
        if (node.children.length) {
          const children = document.createElement('div');
          children.className = 'main-tree-children ' + (node.children.length === 1 ? 'single' : 'branched');
          for (const child of node.children) {
            conversationTreeEdges.push({ parent: node.id, child: child.id });
            children.appendChild(mainNode(child));
          }
          family.appendChild(children);
        }
        return family;
      }
      for (const node of roots) mainConversationTree.appendChild(mainNode(node));
      scheduleConversationTreeEdges();
    }

    function setConversationTreeExpanded(expanded) {
      treeToggle.setAttribute('aria-expanded', String(expanded));
      mainTreeView.hidden = !expanded;
      if (expanded) {
        mainTreeView.scrollTop = 0;
        renderConversationTree();
      }
    }

    function highlightUserCard(card) {
      if (!card) return;
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.classList.add('tree-highlight');
      setTimeout(() => card.classList.remove('tree-highlight'), 1400);
    }

    function scrollToUserTurn(node) {
      const card = Array.from(feed.querySelectorAll('.card.user')).find(
        (item) => Number(item.dataset.turnIndex) === node.turnIndex,
      );
      highlightUserCard(card);
    }

    async function navigateToTreeNode(node) {
      setConversationTreeExpanded(false);
      if (node.sessionId !== selectedSessionId) {
        pendingTreeNavigation = node;
        await selectConversation(node.sessionId);
      } else {
        scrollToUserTurn(node);
      }
    }

    treeToggle.addEventListener('click', () => {
      setConversationTreeExpanded(treeToggle.getAttribute('aria-expanded') !== 'true');
    });
    mainTreeClose.addEventListener('click', () => setConversationTreeExpanded(false));

    function renderConversationList() {
      conversationList.querySelectorAll(".conversation-item").forEach((item) => item.remove());
      conversationEmpty.hidden = conversations.length > 0;
      for (const conversation of conversations) {
        const id = conversationId(conversation);
        if (!id) continue;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "conversation-item" + (id === selectedSessionId ? " active" : "") + (conversation.pinned ? " pinned" : "");
        button.dataset.sessionId = id;
        const workspacePath = conversationWorkspacePath(conversation);
        button.title = [
          conversation.title || conversation.task || "",
          "Click to copy full ID: " + id,
          workspacePath,
        ].filter(Boolean).join("\n");
        button.addEventListener("click", () => selectConversation(id));
        button.addEventListener("contextmenu", (event) => showConversationContextMenu(event, id));

        const titleRow = document.createElement("span");
        titleRow.className = "conversation-title-row";
        const title = document.createElement("span");
        title.className = "conversation-title";
        title.textContent = conversation.title || conversation.task || "Untitled conversation";
        titleRow.appendChild(title);
        if (workspacePath) {
          const workspace = document.createElement("span");
          workspace.className = "conversation-workspace";
          workspace.textContent = workspacePath;
          workspace.title = workspacePath;
          titleRow.appendChild(workspace);
        }
        button.appendChild(titleRow);

        const idRow = document.createElement("span");
        idRow.className = "conversation-id-row";
        const idValue = document.createElement("span");
        idValue.className = "conversation-id";
        idValue.textContent = id.slice(0, 8);
        idValue.dataset.fullId = id;
        idValue.title = "Click to copy full conversation ID: " + id;
        idValue.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          copyConversationId(id);
        });
        idRow.appendChild(idValue);
        button.appendChild(idRow);

        const meta = document.createElement("span");
        meta.className = "conversation-meta";
        const time = document.createElement("span");
        time.textContent = formatConversationTime(conversation.created_at);
        const status = document.createElement("span");
        status.className = "conversation-status " + statusClass(conversationState(conversation));
        status.textContent = conversationStatus(conversation);
        meta.appendChild(time);
        meta.appendChild(status);
        button.appendChild(meta);
        conversationList.appendChild(button);
      }
      renderConversationTree();
    }

    function resetRunView(conversation) {
      clearFeed();
      resetActivity();
      currentAssistant = null;
      currentReasoning = null;
      conversationLabel.textContent = (conversation && (conversation.title || conversation.task)) || (conversation && conversationId(conversation) ? "Untitled conversation" : "New conversation");
      setConversationWorkspace(conversation);
      statStep.textContent = conversation && conversation.n_calls ? String(conversation.n_calls) : "-";
      statCost.textContent = "$" + Number((conversation && conversation.cost) || 0).toFixed(4);
      setTokenStatsFromConversation(conversation);
      renderModelSwitcher(
        conversation && conversation.model_name
          ? conversation.model_name
          : String(guiSettings.active_model || "")
      );
      applyConversationStatus(conversation);
    }

    function updateConversationStatus(sessionId, changes) {
      const conversation = conversations.find((item) => conversationId(item) === sessionId);
      if (!conversation) return;
      Object.assign(conversation, changes);
      renderConversationList();
    }

    async function loadConversations({ selectNewest = false } = {}) {
      try {
        const response = await fetch("/api/conversations", { cache: "no-store" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        conversations = Array.isArray(data) ? data : (data.conversations || data.sessions || data.items || []);
        renderConversationList();
        if (selectedSessionId) {
          const selectedConversation = conversations.find((item) => conversationId(item) === selectedSessionId);
          if (selectedConversation && selectedConversation.done) applyConversationStatus(selectedConversation);
        }
        if (selectNewest && !selectedSessionId && conversations.length) {
          await selectConversation(conversationId(conversations[0]));
        }
      } catch (err) {
        console.error("Could not load conversations", err);
      }
    }

    async function selectConversation(sessionId) {
      if (!sessionId) return;
      const selectionVersion = ++conversationSelectionVersion;
      setConversationTreeExpanded(false);
      if (sessionId === selectedSessionId && eventSource) return;
      closeEventStream();
      activeSessionId = sessionId;
      selectedSessionId = sessionId;
      rememberSession(sessionId);
      const conversation = conversations.find((item) => conversationId(item) === sessionId);
      renderConversationList();
      resetRunView(conversation || { session_id: sessionId });
      if (!conversation) setStatus("loading", "running");
      replayingHistory = true;
      try {
        const snapshot = await apiRequest("/api/sessions/" + encodeURIComponent(sessionId));
        if (selectionVersion !== conversationSelectionVersion || selectedSessionId !== sessionId) return;
        const events = Array.isArray(snapshot.events) ? snapshot.events : [];
        for (let index = 0; index < events.length; index++) {
          const event = events[index];
          try { handleEvent(event); }
          catch (err) { console.error("Bad event", err, event); }
          lastEventOffset = index + 1;
          lastEventType = event.type || "";
        }
        replayingHistory = false;
        const currentConversation = snapshot.conversation || snapshot.session || conversation;
        if (!currentConversation || !currentConversation.done) {
          connectEvents(sessionId, events.length);
        }
      } catch (err) {
        if (selectionVersion !== conversationSelectionVersion || selectedSessionId !== sessionId) return;
        replayingHistory = false;
        console.error("Could not load conversation", err);
        setStatus("load failed", "error");
      }
    }

    function handleEvent(event) {
      const shouldFollowOutput = followOutput;
      // Assistant messages are buffered for one adjacent tool-call window.  A
      // matching action removes the repeated command from the assistant text;
      // any other event simply ends the merge window and keeps the message.
      const keepsAssistantMergeWindow = ["tool_call_delta", "action_start", "action_output", "observation"];
      if (pendingAssistantMessage && !keepsAssistantMergeWindow.includes(event.type)) {
        flushPendingAssistantMessage();
      }
      switch (event.type) {
        case "run_start":
          currentActivityThinking = null;
          if (!replayingHistory) {
            runTokenUsage = { prompt: 0, completion: 0, total: 0 };
          }
          addRunMarker("start", "Run started", event.step_limit ? "limit " + event.step_limit + " steps" : "");
          addUserCard(event.task || "", event.images || []);
          setStatus("running", "running");
          setComposerRunning(true);
          markActivityStarted();
          addActivityEntry("run", "Run started", {
            marker: "▶",
            detail: event.task ? previewText(event.task, 180) : "Agent run accepted.",
            status: "running",
            segment: true,
          });
          break;
        case "plan": {
          const planBits = [
            event.intent ? String(event.intent).replace(/_/g, " ") : "",
            event.reply_directly ? "direct answer" : event.write_files ? "writes files" : "read-only",
            event.steps ? "budget " + event.steps + " steps" : "",
          ].filter(Boolean).join(" · ");
          addActivityEntry("plan", "Plan", {
            marker: "◎",
            detail: previewText(event.plan || "Planner unavailable; using the default step budget.", 220),
            status: "planned",
            meta: planBits,
          });
          break;
        }
        case "budget_reached":
          addActivityEntry("plan", "Step budget reached", {
            marker: "‖",
            detail: "Used " + event.budget + " steps. Summarizing the work before deciding whether to continue.",
            status: "checkpoint",
            meta: event.budget + " steps",
          });
          break;
        case "budget_extended":
          addActivityEntry("plan", "Continuing with a new plan", {
            marker: "↻",
            detail: previewText(event.plan || event.summary || "Continuing the task.", 220),
            status: "running",
            meta: "+" + event.steps + " steps (budget " + event.budget + ")",
          });
          break;
        case "step_start":
          statStep.textContent = String(event.step || "-");
          setStatus("running", "running");
          addActivityStep(event.step);
          break;
        case "assistant_start":
          activityLiveText = "";
          activityReasoningText = "";
          ensureAssistantCard();
          if (currentActivityStepNumber) {
            updateActivityStep(currentActivityStepNumber, {
              detail: 'Receiving the model response…',
              status: 'thinking',
            });
          }
          ensureActivityThinking(currentActivityStepNumber, activityEventTime(event));
          break;
        case "assistant_delta": {
          const timestamp = activityEventTime(event);
          finishActivityThinking(timestamp);
          beginActivityPhase(activityStepEntries.get(Number(event.step || currentActivityStepNumber)) || currentActivityStep, 'reply', timestamp);
          const card = ensureAssistantCard();
          activityLiveText += event.text || "";
          appendText(card, event.text || "");
          if (activityLiveText.trim()) {
            updateActivityStep(event.step || currentActivityStepNumber, {
              title: "Step " + (event.step || currentActivityStepNumber || "?") + " · Writing response",
              detail: previewText(activityLiveText, 220),
              status: "thinking",
            });
          }
          break;
        }
        case "reasoning_delta": {
          const thinkingEntry = ensureActivityThinking(event.step || currentActivityStepNumber, activityEventTime(event));
          const card = ensureAssistantCard();
          if (!currentReasoning) {
            const details = document.createElement("details");
            details.className = "reasoning";
            const summary = document.createElement("summary");
            summary.textContent = "Reasoning";
            details.appendChild(summary);
            const body = document.createElement("div");
            body.className = "reasoning-body";
            details.appendChild(body);
            card.appendChild(details);
            currentReasoning = body;
          }
          currentReasoning.textContent += event.text || "";
          activityReasoningText += event.text || "";
          if (!activityLiveText.trim() && activityReasoningText.trim()) {
            updateActivityEntry(thinkingEntry, {
              detail: previewText(activityReasoningText, 220),
              status: 'thinking',
            });
          }
          break;
        }
        case "tool_call_delta":
          if (!currentAssistant) ensureAssistantCard();
          break;
        case "assistant_end":
          finishActivityThinking(activityEventTime(event));
          endActivityPhase(activityStepEntries.get(Number(event.step || currentActivityStepNumber)) || currentActivityStep, activityEventTime(event));
          break;
        case "assistant_message": {
          finishActivityThinking(activityEventTime(event));
          endActivityPhase(activityStepEntries.get(Number(event.step || currentActivityStepNumber)) || currentActivityStep, activityEventTime(event));
          const card = ensureAssistantCard();
          if (event.content) {
            let content = card.querySelector(".content");
            if (!content) {
              content = document.createElement("div");
              content.className = "content";
              card.insertBefore(content, card.querySelector(".reasoning"));
            }
            setMarkdownContent(content, event.content);
          }
          const callTokens = updateTokenStatsFromAssistant(event);
          const callCostKnown = event.cost_known !== false;
          if (callCostKnown) {
            const currentCost = Number(statCost.textContent.replace("$", "")) || 0;
            statCost.textContent = "$" + (currentCost + Number(event.cost || 0)).toFixed(4);
          }
          const actions = Array.isArray(event.actions) ? event.actions : [];
          const activityTitle = "Step " + (event.step || currentActivityStepNumber || "?") + " · "
            + (actions.length ? "Command" + (actions.length === 1 ? "" : "s") + " requested" : "Model response");
          const activityDetail = previewText(event.content, 220)
            || (actions.length ? actions.length + " command" + (actions.length === 1 ? "" : "s") + " queued." : "(empty response)");
          const activityMeta = [
            callTokens ? formatTokens(callTokens) + " tokens" : "",
            event.total_tokens_used != null ? formatTokens(event.total_tokens_used) + " cumulative" : "",
            callCostKnown ? "$" + Number(event.cost || 0).toFixed(4) : "no cost data",
          ].filter(Boolean).join(" · ");
          updateActivityStep(event.step || currentActivityStepNumber, {
            title: activityTitle,
            detail: activityDetail,
            status: "responded",
            meta: activityMeta,
          });
          updateConversationStatus(activeSessionId, {
            prompt_tokens: currentTokenUsage.prompt,
            completion_tokens: currentTokenUsage.completion,
            total_tokens: currentTokenUsage.total,
            token_usage: {
              prompt_tokens: currentTokenUsage.prompt,
              completion_tokens: currentTokenUsage.completion,
              total_tokens: currentTokenUsage.total,
            },
          });
          queueAssistantMessageForMerge(card, event.content || "", event.step);
          currentAssistant = null;
          currentReasoning = null;
          break;
        }
        case "action_start": {
          // A tool call is the source of truth for an executed command.  Remove
          // the same command from the adjacent assistant message, then show one
          // quiet line in the transcript.  Full details stay in Activity.
          mergeMatchingCommandIntoPendingAssistant(event.command || "");
          makeToolCall(event.command || "", event.tool_call_id);
          addToolActivity(event);
          break;
        }
        case "action_output": {
          updateToolCall(event);
          updateToolActivity(event);
          break;
        }
        case "observation":
          break;
        case "format_error": {
          finishActivityThinking(activityEventTime(event));
          endActivityPhase(currentActivityStep, activityEventTime(event));
          currentAssistant = null;
          currentReasoning = null;
          activityLiveText = "";
          activityReasoningText = "";
          updateTokenStatsFromAssistant(event);
          activityState.errors += 1;
          const card = makeCard("error", "Format error", "retrying");
          appendText(card, event.message || "");
          if (event.step || currentActivityStepNumber) {
            updateActivityStep(event.step || currentActivityStepNumber, {
              title: "Step " + (event.step || currentActivityStepNumber) + " · Format error",
              detail: previewText(event.message, 220) || "The model response could not be parsed.",
              status: "retrying",
            });
          }
          addActivityEntry("error", "Model response format error", {
            detail: previewText(event.message, 220) || "The response could not be parsed; retrying.",
            status: "retrying",
            meta: event.consecutive_errors ? event.consecutive_errors + " consecutive error(s)" : "",
          });
          updateActivitySummary();
          break;
        }
        case "error": {
          finishActivityThinking(activityEventTime(event));
          currentAssistant = null;
          currentReasoning = null;
          activityLiveText = "";
          activityReasoningText = "";
          activityState.errors += 1;
          const card = makeCard("error", "Error");
          appendText(card, event.message || "Unknown error");
          setStatus("error", "error");
          if (currentActivityStepNumber) {
            updateActivityStep(currentActivityStepNumber, {
              title: "Step " + currentActivityStepNumber + " · Error",
              detail: previewText(event.message, 220) || "The agent reported an error.",
              status: "failed",
            });
          }
          addActivityEntry("error", "Agent error", {
            detail: previewText(event.message, 220) || "The agent reported an error.",
            status: "failed",
          });
          updateActivitySummary();
          break;
        }
        case "done": {
          if (!replayingHistory) {
            statCost.textContent = "$" + Number(event.conversation_cost ?? event.cost ?? 0).toFixed(4);
            statStep.textContent = String(event.n_calls ?? statStep.textContent);
            updateTokenStatsFromDone(event);
          }
          const exitStatus = event.exit_status || "done";
          const doneState = conversationStateFromStatus(exitStatus, true);
          const headerState = doneState === "failed" ? "error" : doneState;
          setStatus(exitStatus, headerState);
          if (event.submission && !isRepeatedAssistantSubmission(event.submission)) {
            const card = makeCard("assistant", "Submission");
            appendText(card, event.submission);
          }
          addRunMarker("end", "Run finished", event.exit_status || "done");
          finalizeActivity(event);
          updateConversationStatus(activeSessionId, {
            done: true,
            status: exitStatus,
            state: doneState,
            exit_status: event.exit_status || "",
            cost: Number(event.cost || 0),
            n_calls: Number(event.n_calls || 0),
            prompt_tokens: currentTokenUsage.prompt,
            completion_tokens: currentTokenUsage.completion,
            total_tokens: currentTokenUsage.total,
            token_usage: {
              prompt_tokens: currentTokenUsage.prompt,
              completion_tokens: currentTokenUsage.completion,
              total_tokens: currentTokenUsage.total,
            },
          });
          if (!replayingHistory) {
            closeEventStream();
            setTimeout(() => loadConversations(), 250);
          }
          setComposerRunning(false);
          break;
        }
      }
      if (shouldFollowOutput && eventAffectsVisibleFeed(event)) scrollFeedToBottom();
      updateScrollButtons();
    }

    function closeEventStream() {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      activeSessionId = null;
    }

    function connectEvents(sessionId, eventOffset = 0) {
      closeEventStream();
      activeSessionId = sessionId;
      lastEventOffset = eventOffset;
      lastEventType = "";
      const suffix = eventOffset > 0 ? "?since=" + encodeURIComponent(eventOffset) : "";
      const source = new EventSource("/api/events/" + encodeURIComponent(sessionId) + suffix);
      eventSource = source;
      source.onmessage = (message) => {
        if (eventSource !== source) return;
        try {
          const event = JSON.parse(message.data);
          handleEvent(event);
          lastEventOffset += 1;
          lastEventType = event.type || "";
        } catch (err) { console.error("Bad event", err, message.data); }
      };
      source.onerror = async () => {
        if (eventSource !== source) return;
        closeEventStream();
        // The stream can close at the same moment the run finishes. Refresh the
        // server summaries before claiming the connection was lost, otherwise a
        // completed run is reported as an error.
        await loadConversations();
        const conversation = conversations.find((item) => conversationId(item) === sessionId);
        if (conversation && !conversation.done) {
          setStatus("disconnected", "error");
          addActivityEntry("error", "Connection lost", {
            detail: "The event stream disconnected before the run finished. Reconnecting reloads the conversation.",
            status: "disconnected",
          });
          activityState.errors += 1;
          markActivityFinished(true);
        } else if (conversation) {
          // The summary is authoritative for finished runs.
          applyConversationStatus(conversation);
          if (conversation.done && lastEventType !== "done") {
            // The terminal event was lost with the connection. Replay the
            // missing tail so the activity panel also shows "Submitted".
            try {
              const snapshot = await apiRequest("/api/sessions/" + encodeURIComponent(sessionId));
              const events = Array.isArray(snapshot.events) ? snapshot.events : [];
              const wasReplaying = replayingHistory;
              replayingHistory = true;
              for (let index = lastEventOffset; index < events.length; index++) {
                const event = events[index];
                try { handleEvent(event); }
                catch (err) { console.error("Bad event", err, event); }
                lastEventOffset = index + 1;
                lastEventType = event.type || "";
              }
              replayingHistory = wasReplaying;
            } catch (err) {
              console.error("Could not replay missing events", err);
            }
          }
        }
      };
    }

    async function startRun() {
      const images = pendingImages.slice();
      let task = taskEl.value.trim();
      if (!task && !images.length) {
        taskEl.focus();
        return;
      }
      if (!task) task = "Please analyze the attached image.";
      const conversation = selectedSessionId
        ? conversations.find((item) => conversationId(item) === selectedSessionId)
        : null;
      // A finished or stopped conversation accepts follow-up messages.  Keep
      // the transcript on screen and let the server append the next turn to the
      // same session instead of creating a new conversation.
      const continuing = Boolean(conversation && conversation.done);
      setComposerRunning(true);
      closeEventStream();
      if (!continuing) {
        conversationLabel.textContent = task;
        clearFeed();
        resetActivity();
        statStep.textContent = "-";
        statCost.textContent = "$0.0000";
        currentTokenUsage = { prompt: 0, completion: 0, total: 0 };
        completedTokenUsage = { prompt: 0, completion: 0, total: 0 };
        runTokenUsage = { prompt: 0, completion: 0, total: 0 };
        setTokenStats(0, 0, 0);
        setContextStats(0);
      }
      setStatus("starting", "running");
      stopButton.disabled = true;
      try {
        const payload = { task, images };
        if (modelSwitcher.value) payload.model_name = modelSwitcher.value;
        if (continuing) payload.session_id = selectedSessionId;
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        taskEl.value = "";
        resizeTaskInput();
        pendingImages = [];
        renderPendingImages();
        imageInput.value = "";
        taskEl.focus();
        const sessionId = data.session_id;
        selectedSessionId = sessionId;
        rememberSession(sessionId);
        if (data.conversation) {
          conversations = conversations.filter((item) => conversationId(item) !== sessionId);
          conversations.unshift(data.conversation);
          setConversationWorkspace(data.conversation);
        } else {
          const newConversation = { session_id: sessionId, task, status: "running", state: "running", created_at: Date.now() / 1000, image_count: images.length };
          conversations.unshift(newConversation);
        }
        renderConversationList();
        setComposerRunning(true);
        replayingHistory = false;
        connectEvents(sessionId, Number(data.event_offset || 0));
        setTimeout(() => loadConversations(), 250);
      } catch (err) {
        setComposerRunning(false);
        setStatus("error", "error");
        const card = makeCard("error", "Error");
        appendText(card, String(err));
        addActivityEntry("error", "Could not start run", {
          detail: String(err),
          status: "failed",
        });
      }
    }

    async function stopRun() {
      // Prefer the currently connected session, but fall back to the selected
      // conversation if the event stream has disconnected.  Otherwise a
      // transient SSE error makes the Stop button silently do nothing.
      const sessionId = activeSessionId || selectedSessionId;
      if (!sessionId) {
        setComposerRunning(false);
        return;
      }
      stopButton.disabled = true;
      setStatus("stopping", "running");
      addActivity("Stopping run…", { kind: "info", status: "stopping", segment: false });
      try {
        const response = await fetch("/api/stop/" + encodeURIComponent(sessionId), { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        // Switch back to Send immediately, even if the SSE connection has
        // already dropped.  The server waits for the stopped run to finish
        // before starting a follow-up sent to this same conversation.
        updateConversationStatus(sessionId, {
          done: true,
          status: "Stopped",
          state: "stopped",
          exit_status: "Stopped",
        });
        setStatus("Stopped", "stopped");
        setComposerRunning(false);
        taskEl.focus();
      } catch (err) {
        stopButton.disabled = false;
        setStatus("running", "running");
        addActivity("Could not stop run", { kind: "error", detail: String(err), status: "failed" });
      }
    }

    function startNewConversation() {
      closeEventStream();
      setConversationTreeExpanded(false);
      selectedSessionId = null;
      rememberSession("");
      renderConversationList();
      clearFeed();
      resetActivity();
      conversationLabel.textContent = "New conversation";
      setConversationWorkspace(null);
      statStep.textContent = "-";
      statCost.textContent = "$0.0000";
      currentTokenUsage = { prompt: 0, completion: 0, total: 0 };
      completedTokenUsage = { prompt: 0, completion: 0, total: 0 };
      runTokenUsage = { prompt: 0, completion: 0, total: 0 };
      setTokenStats(0, 0, 0);
      setContextStats(0);
      renderModelSwitcher(String(guiSettings.active_model || ""));
      setStatus("idle", "");
      setComposerRunning(false);
      taskEl.value = "";
      resizeTaskInput();
      taskEl.focus();
    }

    attachImageButton.addEventListener("click", () => imageInput.click());
    imageInput.addEventListener("change", async () => {
      await addImageFiles(imageInput.files);
      imageInput.value = "";
    });
    taskEl.addEventListener("paste", async (event) => {
      const files = Array.from((event.clipboardData && event.clipboardData.items) || [])
        .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
        .map((item) => item.getAsFile())
        .filter(Boolean);
      if (files.length) await addImageFiles(files);
    });
    ["dragenter", "dragover"].forEach((name) => {
      composerEl.addEventListener(name, (event) => {
        event.preventDefault();
        composerEl.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      composerEl.addEventListener(name, (event) => {
        event.preventDefault();
        composerEl.classList.remove("dragging");
      });
    });
    composerEl.addEventListener("drop", async (event) => {
      const files = event.dataTransfer ? event.dataTransfer.files : [];
      await addImageFiles(files);
    });
    sendButton.addEventListener("click", startRun);
    stopButton.addEventListener("click", stopRun);
    newConversationButton.addEventListener("click", startNewConversation);
    modelSwitcher.addEventListener("change", () => switchModel(modelSwitcher.value));
    settingsButton.addEventListener("click", () => openSettings("general"));
    settingsCloseButton.addEventListener("click", closeSettings);
    settingsModal.addEventListener("click", (event) => {
      if (event.target === settingsModal) closeSettings();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !settingsModal.hidden) closeSettings();
    });
    for (const tab of settingsTabs) {
      tab.addEventListener("click", () => switchSettingsTab(tab.dataset.settingsTab));
    }
    addProviderButton.addEventListener("click", () => showProviderForm(null));
    providerCancelButton.addEventListener("click", hideProviderForm);
    providerFetchModelsButton.addEventListener("click", saveAndLoadProviderModels);
    providerForm.addEventListener("submit", saveProviderForm);
    addModelButton.addEventListener("click", () => showModelForm(null, Array.from(expandedProviderIds)[0] || ""));
    modelCancelButton.addEventListener("click", hideModelForm);
    modelForm.addEventListener("submit", saveModelForm);
    taskEl.addEventListener("input", resizeTaskInput);
    taskEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) startRun();
    });
    updateScrollButtons();

    fetch("/api/config").then((r) => r.json()).then((config) => {
      defaultWorkspace = config.workspace || "";
      if (!selectedSessionId) setConversationWorkspace(null);
      if (config.initial_task) {
        taskEl.value = config.initial_task;
        resizeTaskInput();
      }
      databasePath = config.database_path || "";
      const databaseElement = document.getElementById("settings-database");
      databaseElement.textContent = databasePath ? "SQLite: " + databasePath : "-";
      databaseElement.title = databasePath;
    }).catch(() => {});

    async function restoreConversation() {
      const remembered = rememberedSessionId();
      await loadConversations();
      if (remembered && conversations.some((conversation) => conversationId(conversation) === remembered)) {
        await selectConversation(remembered);
      } else if (conversations.length) {
        await selectConversation(conversationId(conversations[0]));
      }
    }

    loadGuiSettings({ selectActive: true }).then(() => {
      if (!selectedSessionId) renderModelSwitcher(String(guiSettings.active_model || ""));
      restoreConversation();
    });
    setInterval(() => loadConversations(), 5000);
    window.addEventListener("hashchange", () => {
      const sessionId = rememberedSessionId();
      if (sessionId && sessionId !== selectedSessionId) {
        selectConversation(sessionId);
      }
    });
  </script>
</body>
</html>
"""


def _workspace_from_config(agent_config: dict[str, Any]) -> str:
    """Return the directory in which GUI-triggered commands run."""
    environment = agent_config.get("environment") or {}
    return str(environment.get("cwd") or agent_config.get("workspace") or Path.cwd())


@dataclass
class GuiConfig:
    """Configuration shared by every GUI-triggered agent run."""

    agent_config: dict[str, Any]
    initial_task: str = ""
    model_name: str = ""
    workspace: str = ""

    def __post_init__(self) -> None:
        self.workspace = self.workspace or _workspace_from_config(self.agent_config)


def build_gui_config(
    *,
    config_spec: list[str],
    model_name: str | None = None,
    model_class: str | None = None,
    agent_class: str | None = None,
    environment_class: str | None = None,
    output: Path | None = None,
    initial_task: str = "",
) -> GuiConfig:
    """Merge CLI config specs into the configuration used for each GUI run."""
    configs: list[dict[str, Any]] = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "agent": {
                # A browser has no terminal, so confirmation prompts would block.
                # The GUI runs non-interactively; use the default agent unless the
                # caller explicitly selected another one.
                "agent_class": agent_class or "default",
                "mode": "yolo",
                "confirm_exit": False,
                "output_path": output or UNSET,
            },
            "model": {
                "model_class": model_class or UNSET,
                "model_name": model_name or UNSET,
            },
            "environment": {
                "environment_class": environment_class or UNSET,
            },
        }
    )
    config = recursive_merge(*configs)
    # The GUI cannot ask the user to intervene mid-run, so the agent classifies
    # each request and negotiates its own step budget before using tools.
    config.setdefault("agent", {}).setdefault("auto_budget", True)
    return GuiConfig(
        agent_config=config,
        initial_task=initial_task,
        model_name=str(config.get("model", {}).get("model_name") or ""),
    )


def create_agent_from_config(config: GuiConfig) -> Any:
    """Create a fresh model, environment, and agent for one GUI session."""
    model = get_model(config=config.agent_config.get("model", {}))
    env = get_environment(config.agent_config.get("environment", {}), default_type="local")
    return get_agent(model, env, config.agent_config.get("agent", {}), default_type="default")


def _json_safe(value: Any) -> Any:
    """Convert arbitrary event payloads into JSON-compatible values."""
    return json.loads(json.dumps(value, default=str))


def _callable_accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    """Return whether *func* can be called with a keyword argument named *name*.

    Test doubles and builtins sometimes do not expose a Python signature.  In
    that case optimistically try the keyword: the richer call is what real
    agents support, and a genuinely incompatible callable still raises and is
    reported through the normal GUI error path.
    """
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


class GuiSession:
    """One browser-triggered agent run and its replayable event stream."""

    _TRAJECTORY_EVENT_TYPES = {"assistant_message", "observation", "done", "error", "format_error"}

    def __init__(
        self,
        task: str,
        *,
        config: GuiConfig,
        model_name: str = "",
        images: list[str] | None = None,
        agent_factory: Callable[[], Any] | None = None,
        store: ConversationStore | None = None,
    ):
        self.session_id = uuid.uuid4().hex
        self.task = task
        self.title = task
        self.pinned = False
        self.parent_id = ''
        self.branch_turn_index = -1
        self.images = list(images or [])
        # ``task`` is the conversation title (its first user message), while
        # ``current_*`` describes the turn that is currently executing.  Keeping
        # them separate lets a conversation contain several follow-up runs.
        self.current_task = task
        self.current_images = list(images or [])
        self.config = config
        self.model_name = str(model_name or config.model_name or "").strip()
        self.workspace = config.workspace
        self.agent_factory = agent_factory or (lambda: create_agent_from_config(config))
        self.events: list[dict[str, Any]] = []
        self._next_event_index = 0
        self.condition = threading.Condition()
        self.stop_requested = threading.Event()
        self.agent: Any | None = None
        self.done = False
        self._done_emitted = False
        self.created_at = time.time()
        self.exit_status = ""
        self.cost = 0.0
        self.completed_cost = 0.0
        self.n_calls = 0
        self.tokens = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Token usage of finished turns plus the current turn, so the Run panel
        # can show whole-conversation consumption instead of only the last run.
        self.completed_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.run_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.context_tokens = 0
        self.error = ""
        self.trajectory: dict[str, Any] | None = None
        # Cumulative message history for this conversation.  It is seeded into
        # each fresh agent on a follow-up turn so the model keeps the context of
        # all earlier user instructions, tool calls, and observations.
        self.messages: list[dict[str, Any]] = []
        self._store = store
        self.thread = threading.Thread(target=self._run, name=f"mini-gui-{self.session_id[:8]}", daemon=True)

    @classmethod
    def restore(
        cls,
        summary: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        config: GuiConfig,
        store: ConversationStore | None = None,
        event_indices: list[int] | None = None,
    ) -> GuiSession:
        """Recreate a completed or interrupted session loaded from SQLite."""
        session = cls.__new__(cls)
        session.session_id = str(summary["session_id"])
        session.task = str(summary.get("task") or "")
        session.title = str(summary.get("title") or session.task)
        session.pinned = bool(summary.get("pinned"))
        session.parent_id = str(summary.get("parent_id") or "")
        session.branch_turn_index = int(summary.get("branch_turn_index", -1))
        # Persisted event indices can have gaps (older branch copies did not
        # store their prefix). Continue after the highest stored index so new
        # events never overwrite surviving history.
        session._next_event_index = (max(event_indices) + 1) if event_indices else len(events)
        session.images = list(
            next((event.get("images") for event in events if event.get("type") == "run_start"), []) or []
        )
        session.config = config
        run_start_events = [event for event in events if event.get("type") == "run_start"]
        session.model_name = str(
            summary.get("model_name")
            or (run_start_events[-1].get("model_name") if run_start_events else "")
            or config.model_name
            or ""
        ).strip()
        session.workspace = str(summary.get("workspace") or config.workspace)
        session.agent_factory = None
        session.events = list(events)
        session.condition = threading.Condition()
        session.stop_requested = threading.Event()
        session.agent = None
        session.done = bool(summary.get("done", True))
        session._done_emitted = True
        session.created_at = float(summary.get("created_at") or time.time())
        session.exit_status = str(summary.get("exit_status") or "")
        session.cost = float(summary.get("cost") or 0.0)
        session.n_calls = int(summary.get("n_calls") or 0)
        usage = summary.get("token_usage") or {}
        session.token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or summary.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or summary.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or summary.get("total_tokens") or 0),
        }
        # Summaries written by older versions only stored the last run. Rebuild
        # whole-conversation totals from the persisted terminal events.
        completed = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        completed_cost = 0.0
        context_tokens = 0
        saw_done = False
        for event in events:
            event_type = event.get("type")
            if event_type == "done":
                saw_done = True
                event_usage = event.get("token_usage") or event.get("cumulative_usage") or {}
                if isinstance(event_usage, dict):
                    for key in completed:
                        completed[key] += int(event_usage.get(key) or 0)
                completed_cost += float(event.get("cost") or 0.0)
            elif event_type in ("assistant_message", "format_error"):
                event_usage = event.get("usage") or {}
                usage_total = event_usage.get("total_tokens") if isinstance(event_usage, dict) else 0
                context_tokens = int(event.get("total_tokens") or usage_total or 0) or context_tokens
        if not saw_done:
            completed = {key: int(session.token_usage.get(key) or 0) for key in completed}
        session.completed_token_usage = completed
        session.run_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        session.token_usage = dict(completed)
        session.tokens = completed["total_tokens"]
        session.completed_cost = completed_cost
        session.cost = completed_cost
        session.context_tokens = context_tokens
        session.error = str(summary.get("error") or "")
        session.trajectory = (
            store.load_trajectory(session.session_id)
            if store is not None and hasattr(store, "load_trajectory")
            else None
        )
        session.messages = copy.deepcopy((session.trajectory or {}).get("messages") or [])
        if not session.messages:
            session.messages = session._history_from_events()
        session._store = store
        session.thread = None
        if session.done and (not session.events or session.events[-1].get("type") != "done"):
            # A process can stop after persisting intermediate events but before
            # the agent emits its terminal event.  The summary is still marked
            # done for such restored runs, so synthesize the missing terminal
            # event to keep event replay consistent with the conversation state.
            session.events.append(
                {
                    "type": "done",
                    "exit_status": session.exit_status or str(summary.get("status") or "Interrupted"),
                    "submission": "",
                    "cost": session.cost,
                    "n_calls": session.n_calls,
                    "prompt_tokens": session.token_usage["prompt_tokens"],
                    "completion_tokens": session.token_usage["completion_tokens"],
                    "total_tokens": session.token_usage["total_tokens"],
                    "tokens": session.tokens,
                    "token_usage": dict(session.token_usage),
                }
            )
        return session

    def start(self) -> None:
        self.thread.start()

    def set_model(self, model_name: str, agent_factory: Callable[[], Any] | None = None) -> None:
        """Select the model used by subsequent turns in this conversation."""
        self.model_name = str(model_name or "").strip()
        if agent_factory is not None:
            self.agent_factory = agent_factory

    def request_stop(self) -> bool:
        """Ask the running agent to stop and return whether it was still running."""
        with self.condition:
            if self.done:
                return False
            self.stop_requested.set()
            agent = self.agent
        if agent is not None:
            if hasattr(agent, "request_stop"):
                agent.request_stop()
            elif hasattr(agent, "stop"):
                agent.stop()
        return True

    def stop(self) -> bool:
        """Alias for :meth:`request_stop`."""
        return self.request_stop()

    def continue_run(self, task: str, images: list[str] | None = None) -> int:
        """Append a follow-up user turn and start another run in this conversation.

        Returns the event index at which the new run starts.  The browser uses
        that offset to skip already-rendered events when it reconnects to the
        session stream.
        """
        task = task.strip()
        images = list(images or [])
        if not task and not images:
            raise ValueError("Task must not be empty")
        if not task:
            task = "Please analyze the attached image."
        with self.condition:
            if not self.done:
                # A user may click Stop and immediately send the next message.
                # Give the previous agent a moment to observe the stop request
                # before rejecting the follow-up.
                if not self.stop_requested.is_set():
                    raise RuntimeError("Conversation is still running")
                deadline = time.monotonic() + 5.0
                while not self.done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("Conversation is still stopping")
                    self.condition.wait(remaining)
            self.current_task = task
            self.current_images = images
            self._ensure_previous_instructions()
            self.done = False
            self._done_emitted = False
            self.stop_requested.clear()
            self.exit_status = ""
            self.error = ""
            self.agent = None
            event_offset = len(self.events)
            self.thread = threading.Thread(target=self._run, name=f"mini-gui-{self.session_id[:8]}", daemon=True)
        self._persist_summary()
        self.thread.start()
        return event_offset

    def _history_from_events(self) -> list[dict[str, Any]]:
        """Build a minimal transcript for agents that do not expose messages.

        Built-in agents keep the exact protocol messages in ``agent.messages``
        and take precedence over this fallback.  Custom agents and conversations
        restored from older databases may not have that data available, so we
        reconstruct the user instructions and visible assistant answers from
        the persisted event stream.  Passing this transcript to the next turn
        is still far better than silently starting a brand-new conversation.
        """
        with self.condition:
            events = list(self.events)
        messages: list[dict[str, Any]] = []
        for event in events:
            event_type = event.get("type")
            if event_type == "run_start":
                task = str(event.get("task") or "").strip()
                if task:
                    messages.append({"role": "user", "content": task})
            elif event_type == "assistant_message":
                content = str(event.get("content") or "").strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
        return messages

    def _ensure_previous_instructions(self) -> None:
        """Seed older user turns that are missing from a restored trajectory.

        Conversations created before follow-ups preserved history only stored
        the latest agent trajectory, even though the event stream still records
        every ``run_start`` task.  When such a conversation is continued, add a
        compact list of the earlier instructions so the model does not treat the
        new message as an isolated, new conversation.
        """
        with self.condition:
            events = list(self.events)
            messages = self.messages
        instructions: list[str] = []
        for event in events:
            if event.get("type") != "run_start":
                continue
            task = str(event.get("task") or "").strip()
            if task and task not in instructions:
                instructions.append(task)
        if not instructions:
            return
        user_text = "\n".join(
            str(message.get("content") or "") for message in messages if message.get("role") == "user"
        )
        missing = [instruction for instruction in instructions if instruction not in user_text]
        if not missing:
            return
        context = (
            "Earlier instructions in this same conversation (use them as "
            "context; the latest user message below is the current request):\n"
            + "\n".join(f"- {instruction}" for instruction in missing)
        )
        insert_at = 1 if messages and messages[0].get("role") == "system" else 0
        messages.insert(insert_at, {"role": "user", "content": context})

    @staticmethod
    def _usage_dict(prompt: Any = 0, completion: Any = 0, total: Any = 0) -> dict[str, int]:
        prompt_tokens = int(prompt or 0)
        completion_tokens = int(completion or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(total or prompt_tokens + completion_tokens),
        }

    def _refresh_token_usage(self) -> None:
        self.token_usage = {
            key: self.completed_token_usage[key] + self.run_token_usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        self.tokens = self.token_usage["total_tokens"]

    def _set_token_usage(self, prompt: Any = 0, completion: Any = 0, total: Any = 0) -> None:
        """Set the current turn's usage; the summary always exposes conversation totals."""
        self.run_token_usage = self._usage_dict(prompt, completion, total)
        self._refresh_token_usage()

    def _add_token_usage(self, usage: dict, *, completed: bool) -> None:
        target = self.completed_token_usage if completed else self.run_token_usage
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            target[key] += int(usage.get(key) or 0)
        self._refresh_token_usage()

    def _summary_locked(self) -> dict[str, Any]:
        status = self.exit_status or ("running" if not self.done else "done")
        state = conversation_state(status, done=self.done)
        return {
            "id": self.session_id,
            "session_id": self.session_id,
            "task": self.task,
            "title": self.title,
            "pinned": self.pinned,
            "parent_id": self.parent_id,
            "branch_turn_index": self.branch_turn_index,
            "turns": [
                {"task": str(event.get("task") or ""), "image_count": len(event.get("images") or [])}
                for event in self.events if event.get("type") == "run_start"
            ],
            "image_count": len(self.images),
            "workspace": self.workspace,
            "model_name": self.model_name,
            "status": status,
            "state": state.value,
            "created_at": self.created_at,
            "done": self.done,
            "exit_status": self.exit_status,
            "cost": self.cost,
            "n_calls": self.n_calls,
            "prompt_tokens": self.token_usage["prompt_tokens"],
            "completion_tokens": self.token_usage["completion_tokens"],
            "total_tokens": self.tokens,
            "tokens": self.tokens,
            "total_prompt_tokens": self.token_usage["prompt_tokens"],
            "total_completion_tokens": self.token_usage["completion_tokens"],
            "total_tokens_used": self.tokens,
            "context_tokens": self.context_tokens,
            "token_usage": dict(self.token_usage),
            "event_count": len(self.events),
            "error": self.error,
        }

    def summary(self) -> dict[str, Any]:
        """Return the metadata needed to render this conversation in the sidebar."""
        with self.condition:
            return self._summary_locked()

    def _persist_summary(self) -> None:
        if self._store is None:
            return
        try:
            self._store.save_session(self.summary())
        except Exception:  # noqa: BLE001 - persistence must not crash an agent run
            logger.exception("Could not persist GUI conversation summary")

    def _persist_trajectory(self) -> None:
        agent = self.agent
        if self._store is None or agent is None or not hasattr(agent, "serialize"):
            return
        try:
            trajectory = agent.serialize()
            if isinstance(trajectory, dict):
                self.trajectory = trajectory
                self._store.save_trajectory(self.session_id, trajectory)
        except Exception:  # noqa: BLE001 - persistence must not crash an agent run
            logger.exception("Could not persist GUI conversation trajectory")

    def _persist_event(
        self, summary: dict[str, Any], event_index: int, event: dict[str, Any], event_type: str | None
    ) -> None:
        if self._store is None:
            return
        try:
            self._store.record_event(summary, event_index, event)
            if event_type in self._TRAJECTORY_EVENT_TYPES:
                self._persist_trajectory()
        except Exception:  # noqa: BLE001 - persistence must not crash an agent run
            logger.exception("Could not persist GUI conversation event")

    def emit(self, event: dict[str, Any]) -> None:
        safe_event = _json_safe(event)
        safe_event.setdefault("timestamp", time.time())
        with self.condition:
            self.events.append(safe_event)
            event_type = safe_event.get("type")
            if event_type in ("assistant_message", "format_error"):
                cumulative = safe_event.get("cumulative_usage")
                if isinstance(cumulative, dict):
                    self._set_token_usage(
                        cumulative.get("prompt_tokens"),
                        cumulative.get("completion_tokens"),
                        cumulative.get("total_tokens"),
                    )
                else:
                    usage = safe_event.get("usage") or {}
                    self._add_token_usage(
                        self._usage_dict(
                            usage.get("prompt_tokens") or safe_event.get("prompt_tokens") or 0,
                            usage.get("completion_tokens") or safe_event.get("completion_tokens") or 0,
                            usage.get("total_tokens")
                            or safe_event.get("tokens")
                            or safe_event.get("total_tokens")
                            or 0,
                        ),
                        completed=False,
                    )
                self.context_tokens = int(
                    safe_event.get("total_tokens") or (safe_event.get("usage") or {}).get("total_tokens") or 0
                )
            if event_type == "done":
                self._done_emitted = True
                self.exit_status = str(safe_event.get("exit_status") or "done")
                self.n_calls = int(safe_event.get("n_calls") or 0)
                usage = safe_event.get("token_usage") or safe_event.get("cumulative_usage") or safe_event.get("usage")
                if not isinstance(usage, dict):
                    usage = {}
                self._add_token_usage(
                    self._usage_dict(
                        safe_event.get(
                            "total_prompt_tokens", usage.get("prompt_tokens", safe_event.get("prompt_tokens", 0))
                        ),
                        safe_event.get(
                            "total_completion_tokens",
                            usage.get("completion_tokens", safe_event.get("completion_tokens", 0)),
                        ),
                        safe_event.get(
                            "total_tokens_used",
                            usage.get("total_tokens", safe_event.get("total_tokens", safe_event.get("tokens", 0))),
                        ),
                    ),
                    completed=True,
                )
                self.run_token_usage = self._usage_dict()
                self._refresh_token_usage()
                self.completed_cost += float(safe_event.get("cost") or 0.0)
                self.cost = self.completed_cost
            elif event_type == "error":
                self.error = str(safe_event.get("message") or "Unknown error")
            safe_event["conversation_usage"] = dict(self.token_usage)
            safe_event["conversation_cost"] = self.cost
            event_index = self._next_event_index
            self._next_event_index += 1
            summary = self._summary_locked()
            self.condition.notify_all()
        self._persist_event(summary, event_index, safe_event, event_type)

    def _run(self) -> None:
        result: dict[str, Any] = {
            "exit_status": "Error",
            "submission": "",
            "cost": 0.0,
            "n_calls": 0,
            "tokens": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        try:
            # A follow-up starts with a fresh model/environment so model
            # switches and per-turn limits work, but it must carry the complete
            # transcript from all previous turns.  Without this, each user
            # instruction was sent to the model as if it were a brand-new
            # conversation and appeared to overwrite the previous one.
            history = copy.deepcopy(self.messages)
            agent = self.agent_factory()
            self.agent = agent
            if hasattr(agent, "set_event_callback"):
                agent.set_event_callback(self.emit)
            if self.stop_requested.is_set():
                if hasattr(agent, "request_stop"):
                    agent.request_stop()
                elif hasattr(agent, "stop"):
                    agent.stop()
            run_kwargs: dict[str, Any] = {}
            continue_method = getattr(agent, "continue_run", None) if history else None
            if callable(continue_method):
                if history and _callable_accepts_keyword(continue_method, "messages"):
                    run_kwargs["messages"] = history
                if self.current_images and _callable_accepts_keyword(continue_method, "images"):
                    run_kwargs["images"] = self.current_images
                returned = continue_method(self.current_task, **run_kwargs)
            else:
                run_method = getattr(agent, "run", None)
                if run_method is None:
                    raise TypeError(f"{type(agent).__name__} has no run() method")
                # Custom agents can opt into conversation history by accepting
                # a messages/history keyword on run().  Built-in DefaultAgent
                # uses continue_run() instead.
                if history:
                    for history_kwarg in ("messages", "history", "message_history"):
                        if _callable_accepts_keyword(run_method, history_kwarg):
                            run_kwargs[history_kwarg] = history
                            break
                if self.current_images and _callable_accepts_keyword(run_method, "images"):
                    run_kwargs["images"] = self.current_images
                returned = run_method(self.current_task, **run_kwargs)
            if isinstance(returned, dict):
                result = {**result, **returned}
            agent_messages = getattr(agent, "messages", None)
            if isinstance(agent_messages, list) and agent_messages:
                self.messages = copy.deepcopy(agent_messages)
            else:
                self.messages = self._history_from_events()
        except UserInterruption:
            # Some custom agents may propagate the stop signal rather than
            # converting it into an exit message. Treat it as a clean user stop
            # instead of surfacing a scary error and leaving the conversation
            # in the failed state.
            result = {**result, "exit_status": "Stopped", "submission": ""}
        except BaseException as exc:  # noqa: BLE001 - report all failures to the browser
            self.emit(
                {
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            logger.exception("GUI agent run failed")
        finally:
            agent = self.agent
            agent_messages = getattr(agent, "messages", None)
            if isinstance(agent_messages, list) and agent_messages:
                self.messages = copy.deepcopy(agent_messages)
            else:
                self.messages = self._history_from_events()
            if not self._done_emitted:
                self.emit({"type": "done", **result})
            with self.condition:
                self.done = True
                self.condition.notify_all()
            self._persist_trajectory()
            self._persist_summary()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the run is complete. Returns ``True`` if it finished."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while not self.done:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    return False
                self.condition.wait(remaining)
            return True


class GuiSessionManager:
    """Owns GUI sessions, restoring and persisting them through SQLite."""

    def __init__(
        self,
        config: GuiConfig,
        *,
        agent_factory: Callable[[], Any] | None = None,
        db_path: str | Path | None = None,
        database_path: str | Path | None = None,
        store: ConversationStore | None = None,
        settings_path: str | Path | None = None,
    ):
        self.config = config
        self.agent_factory = agent_factory
        resolved_db_path = db_path or database_path or os.getenv("MSWEA_GUI_DB_PATH") or DEFAULT_DATABASE_FILE
        self.store = store or ConversationStore(resolved_db_path)
        resolved_settings_path = settings_path or self._default_settings_path(
            resolved_db_path, database_path=database_path, store=store
        )
        model_config = config.agent_config.get("model", {}) if isinstance(config.agent_config, dict) else {}
        self.model_settings = GuiModelSettings(
            resolved_settings_path,
            default_model=config.model_name,
            default_model_class=str(model_config.get("model_class") or ""),
        )
        # Short alias used by templates and integrations.
        self.settings = self.model_settings
        self.sessions: dict[str, GuiSession] = {}
        self._lock = threading.Lock()
        self._restore_sessions()

    @staticmethod
    def _default_settings_path(
        db_path: str | Path,
        *,
        database_path: str | Path | None = None,
        store: ConversationStore | None = None,
    ) -> Path:
        """Keep GUI settings beside an explicitly selected conversation DB.

        This keeps test databases and multiple GUI profiles isolated while the
        normal ``gui_conversations.db`` still resolves to the global config
        directory. ``MSWEA_GUI_SETTINGS_PATH`` always takes precedence.
        """
        configured = os.getenv("MSWEA_GUI_SETTINGS_PATH")
        if configured:
            return Path(configured).expanduser()
        candidate = db_path or database_path or (store.path if store is not None else "")
        if candidate and str(candidate) != ":memory:":
            return Path(str(candidate)).expanduser().with_name("gui_settings.json")
        return DEFAULT_SETTINGS_FILE

    @property
    def active_model_name(self) -> str:
        """Return the model selected in the GUI settings."""
        return str(getattr(self.model_settings, "active_model", "") or "")

    def model_settings_snapshot(self) -> dict[str, Any]:
        """Return provider/model state for both the API and the switcher."""
        payload = self.model_settings.snapshot()
        payload["active_model"] = self.active_model_name
        payload["model_name"] = self.active_model_name
        return payload

    def get_settings(self) -> dict[str, Any]:
        """Alias for :meth:`model_settings_snapshot`."""
        return self.model_settings_snapshot()

    def list_enabled_models(self) -> list[dict[str, Any]]:
        return self.model_settings.list_models(enabled_only=True)

    def switch_model(self, model_name: str) -> dict[str, Any]:
        """Alias for :meth:`set_active_model`."""
        return self.set_active_model(model_name)

    def _config_for_model(self, model_name: str) -> GuiConfig:
        if not model_name:
            return self.config
        config = self.config if model_name == self.config.model_name else copy.deepcopy(self.config)
        model_config = config.agent_config.setdefault("model", {})
        model_config["model_name"] = model_name
        config.model_name = model_name
        try:
            model_entry = self.model_settings.get_model(model_name)
        except SettingsNotFoundError:
            return config
        if model_entry.get("model_class"):
            model_config["model_class"] = model_entry["model_class"]
        provider_id = str(model_entry.get("provider") or "")
        provider = next(
            (item for item in self.model_settings.providers if item.get("id") == provider_id),
            None,
        )
        if provider is not None:
            model_class = str(model_entry.get("model_class") or "")
            if not model_class or "litellm" in model_class.lower():
                qualified_name = _qualified_model_name(provider_id, model_name, provider.get("base_url", ""))
                model_config["model_name"] = qualified_name
                config.model_name = qualified_name
            model_kwargs = model_config.setdefault("model_kwargs", {})
            if provider.get("base_url"):
                model_kwargs.setdefault("api_base", provider["base_url"])
            if provider.get("api_key"):
                model_kwargs.setdefault("api_key", provider["api_key"])
                api_key_env = str(provider.get("api_key_env") or "")
                if api_key_env and not os.getenv(api_key_env):
                    # LiteLLM and several provider integrations resolve keys
                    # from the environment.  GUI-managed keys should behave the
                    # same way for every model class.
                    os.environ[api_key_env] = str(provider["api_key"])
            # Provider-discovered models are usually not in LiteLLM's pricing
            # registry, so fall back to recording the call without cost instead
            # of aborting the run.  Known models still report real costs.
            model_config.setdefault("cost_tracking", "ignore_errors")
        return config

    def _agent_factory_for_model(self, model_name: str) -> Callable[[], Any] | None:
        if self.agent_factory is not None:
            return self.agent_factory
        config = self._config_for_model(model_name)
        return lambda: create_agent_from_config(config)

    def _resolve_model_name(self, model_name: str | None) -> str:
        requested = str(model_name or "").strip()
        if requested:
            try:
                model = self.model_settings.get_model(requested)
            except SettingsNotFoundError:
                model = None
            if model is not None and not model.get("enabled", True):
                raise ValueError(f"Model is disabled: {requested}")
            if model is not None and not _model_is_enabled_by_provider(model, self.model_settings.providers):
                raise ValueError(f"Provider is disabled for model: {requested}")
            return requested
        return self.active_model_name

    def list_providers(self) -> list[dict[str, Any]]:
        return self.model_settings.list_providers()

    def list_models(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        return self.model_settings.list_models(enabled_only=enabled_only)

    def add_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.model_settings.add_provider(payload)

    def fetch_available_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        return self.model_settings.fetch_available_models(provider_id, overrides, timeout=timeout)

    def query_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        return self.model_settings.query_provider_models(provider_id, overrides, timeout=timeout)

    def fetch_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        return self.model_settings.fetch_provider_models(provider_id, overrides, timeout=timeout)

    def load_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        return self.model_settings.load_provider_models(provider_id, overrides, timeout=timeout)

    def refresh_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        return self.model_settings.refresh_provider_models(provider_id, overrides, timeout=timeout)

    def sync_provider_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        return self.model_settings.sync_provider_models(provider_id, overrides, timeout=timeout)

    def fetch_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        return self.model_settings.fetch_models(provider_id, overrides, timeout=timeout)

    def query_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        return self.model_settings.query_models(provider_id, overrides, timeout=timeout)

    def discover_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> list[dict[str, str]]:
        return self.model_settings.discover_models(provider_id, overrides, timeout=timeout)

    def load_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        return self.model_settings.load_models(provider_id, overrides, timeout=timeout)

    def load_models_from_provider(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        return self.model_settings.load_models_from_provider(provider_id, overrides, timeout=timeout)

    def refresh_models(
        self,
        provider_id: str,
        overrides: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        return self.model_settings.refresh_models(provider_id, overrides, timeout=timeout)

    def update_provider(self, provider_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.model_settings.update_provider(provider_id, payload)

    def delete_provider(self, provider_id: str) -> dict[str, Any]:
        return self.model_settings.delete_provider(provider_id)

    def add_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.model_settings.add_model(payload)

    def update_model(self, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.model_settings.update_model(model_name, payload)

    def delete_model(self, model_name: str) -> dict[str, Any]:
        return self.model_settings.delete_model(model_name)

    def set_model_enabled(self, model_name: str, enabled: bool) -> dict[str, Any]:
        return self.model_settings.set_model_enabled(model_name, enabled)

    def set_provider_enabled(self, provider_id: str, enabled: bool) -> dict[str, Any]:
        return self.model_settings.set_provider_enabled(provider_id, enabled)

    def set_active_model(self, model_name: str) -> dict[str, Any]:
        model = self.model_settings.set_active_model(model_name)
        # Existing idle conversations can use the new model on their next turn.
        for session in self.sessions.values():
            if session.done and not session.model_name:
                session.set_model(model["model_name"], self._agent_factory_for_model(model["model_name"]))
        return model

    @property
    def database_path(self) -> str:
        """Return the absolute path to the SQLite conversation database."""
        if self.store.path == ":memory:":
            return self.store.path
        return str(Path(self.store.path).expanduser().resolve())

    def database_info(self) -> dict[str, Any]:
        """Return information that helps users locate the conversation database."""
        path = self.database_path
        exists = path != ":memory:" and Path(path).exists()
        return {
            "path": path,
            "url": f"sqlite:///{path}" if path != ":memory:" else "sqlite:///:memory:",
            "exists": exists,
            "size_bytes": Path(path).stat().st_size if exists else 0,
        }

    def _restore_sessions(self) -> None:
        for summary in self.store.load_sessions():
            entries = self.store.load_event_entries(summary["session_id"])
            events = [event for _, event in entries]
            last_event_type = events[-1].get("type") if events else ""
            if last_event_type == "done":
                if not summary.get("done"):
                    summary["done"] = True
                    summary["exit_status"] = str(events[-1].get("exit_status") or summary.get("exit_status") or "")
                    summary["status"] = summary["exit_status"] or "done"
                    self.store.save_session(summary)
            elif not summary.get("done"):
                # A process restart cannot resume a previous agent thread.
                summary["done"] = True
                summary["status"] = "Interrupted"
                summary["exit_status"] = "Interrupted"
                self.store.save_session(summary)
            session = GuiSession.restore(
                summary,
                events,
                config=self.config,
                store=self.store,
                event_indices=[index for index, _ in entries],
            )
            if self.agent_factory is not None:
                # A custom factory (used by tests and integrations) should also
                # be available when continuing a conversation restored from
                # disk.  The default factory is rebuilt lazily by
                # continue_session() so it can honor a model switch.
                session.agent_factory = self.agent_factory
            self.sessions[session.session_id] = session

    def start_session(self, task: str, images: list[str] | None = None, model_name: str | None = None) -> GuiSession:
        task = task.strip()
        images = list(images or [])
        if not task and not images:
            raise ValueError("Task must not be empty")
        if not task:
            task = "Please analyze the attached image."
        selected_model = self._resolve_model_name(model_name)
        session_config = self._config_for_model(selected_model)
        session = GuiSession(
            task,
            config=session_config,
            model_name=selected_model,
            images=images,
            agent_factory=self._agent_factory_for_model(selected_model),
            store=self.store,
        )
        with self._lock:
            self.sessions[session.session_id] = session
        self.store.save_session(session.summary())
        session.start()
        return session

    def continue_session(
        self,
        session_id: str,
        task: str,
        images: list[str] | None = None,
        model_name: str | None = None,
    ) -> tuple[GuiSession, int] | None:
        """Append a follow-up message to an existing conversation."""
        session = self.get_session(session_id)
        if session is None:
            return None
        selected_model = self._resolve_model_name(model_name) if model_name else session.model_name
        if selected_model and selected_model != session.model_name:
            session.config = self._config_for_model(selected_model)
            session.set_model(selected_model, self._agent_factory_for_model(selected_model))
        elif session.agent_factory is None:
            # Sessions restored from SQLite intentionally have no live agent or
            # factory. Build one for the next turn while preserving the model
            # recorded with the conversation.
            session.config = self._config_for_model(selected_model)
            session.set_model(selected_model, self._agent_factory_for_model(selected_model))
        return session, session.continue_run(task, images=images)

    def rename_session(self, session_id: str, title: str) -> GuiSession | None:
        session = self.get_session(session_id)
        title = title.strip()
        if not title:
            raise ValueError('Title must not be empty')
        if session is not None:
            with session.condition:
                session.title = title
            session._persist_summary()
        return session

    def set_session_pinned(self, session_id: str, pinned: bool) -> GuiSession | None:
        session = self.get_session(session_id)
        if session is not None:
            with session.condition:
                session.pinned = bool(pinned)
            session._persist_summary()
        return session

    def delete_session(self, session_id: str) -> GuiSession | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        if not session.done:
            session.request_stop()
            if not session.wait(timeout=5.0):
                raise RuntimeError('Conversation is still stopping')
        with self._lock:
            self.sessions.pop(session_id, None)
        self.store.delete_session(session_id)
        return session


    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return '\n'.join(
                str(part.get('text') or '') for part in content if isinstance(part, dict)
            )
        return str(content or '')

    def branch_session(self, session_id: str, turn_index: int, task: str) -> GuiSession | None:
        source = self.get_session(session_id)
        task = task.strip()
        if not task:
            raise ValueError('Task must not be empty')
        if source is None:
            return None
        with source.condition:
            if not source.done:
                raise RuntimeError('Wait for the conversation to finish before branching')
            events = copy.deepcopy(source.events)
            source_messages = copy.deepcopy(source.messages)
            source_config = copy.deepcopy(source.config)
            factory = source.agent_factory
        run_starts = [index for index, event in enumerate(events) if event.get('type') == 'run_start']
        if not isinstance(turn_index, int) or isinstance(turn_index, bool):
            raise ValueError('turn_index must be an integer')
        if turn_index < 0 or turn_index >= len(run_starts):
            raise ValueError('Invalid conversation turn')
        event_offset = run_starts[turn_index]
        prefix = events[:event_offset]
        target = events[event_offset]
        tasks = [str(events[index].get('task') or '') for index in run_starts]
        history: list[dict[str, Any]] = []
        if turn_index:
            search_from = 0
            for index, original_task in enumerate(tasks[: turn_index + 1]):
                position = next(
                    (
                        position
                        for position in range(search_from, len(source_messages))
                        if source_messages[position].get('role') == 'user'
                        and original_task.strip() in self._message_text(source_messages[position].get('content')).strip()
                    ),
                    None,
                )
                if position is None:
                    break
                if index == turn_index:
                    history = source_messages[:position]
                    break
                search_from = position + 1
            if not history:
                for event in prefix:
                    if event.get('type') == 'run_start' and str(event.get('task') or '').strip():
                        history.append({'role': 'user', 'content': str(event['task'])})
                    elif event.get('type') == 'assistant_message' and str(event.get('content') or '').strip():
                        history.append({'role': 'assistant', 'content': str(event['content'])})
        selected_model = source.model_name
        branch = GuiSession(
            task,
            config=source_config,
            model_name=selected_model,
            images=list(target.get('images') or []),
            agent_factory=factory or self._agent_factory_for_model(selected_model),
            store=None,
        )
        branch.parent_id = source.session_id
        branch.branch_turn_index = turn_index
        for event in prefix:
            branch.emit(event)
        branch.done = True
        branch._done_emitted = True
        branch.messages = history
        branch._store = self.store
        self.store.save_events(branch.session_id, list(enumerate(branch.events)))
        with self._lock:
            self.sessions[branch.session_id] = branch
        self.store.save_session(branch.summary())
        branch.continue_run(task, images=list(target.get('images') or []))
        return branch

    def get_session(self, session_id: str) -> GuiSession | None:
        with self._lock:
            return self.sessions.get(session_id)

    def list_sessions(self) -> list[GuiSession]:
        """Return every conversation, newest first."""
        with self._lock:
            return sorted(self.sessions.values(), key=lambda session: (not session.pinned, -session.created_at))

    def list_conversations(self) -> list[GuiSession]:
        """Alias for :meth:`list_sessions`."""
        return self.list_sessions()

    def state_counts(self) -> dict[str, int]:
        """Return the number of conversations in each canonical state."""
        counts = {state.value: 0 for state in ConversationState}
        for session in self.list_sessions():
            counts[session.summary()["state"]] += 1
        return counts

    def status_counts(self) -> dict[str, int]:
        """Alias for :meth:`state_counts`."""
        return self.state_counts()

    def export_session(self, session_id: str) -> dict[str, Any] | None:
        """Return a self-contained JSON export for a conversation."""
        return self.store.export_conversation(session_id)

    def export_conversation(self, session_id: str) -> dict[str, Any] | None:
        """Alias for :meth:`export_session`."""
        return self.export_session(session_id)

    def stop_session(self, session_id: str) -> GuiSession | None:
        session = self.get_session(session_id)
        if session is not None:
            session.request_stop()
        return session

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.store.close()


class GuiHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with a reference to the session manager."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], manager: GuiSessionManager):
        self.manager = manager
        super().__init__(server_address, GuiRequestHandler)


class GuiRequestHandler(BaseHTTPRequestHandler):
    server: GuiHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_conversation_export(self, session_id: str) -> None:
        payload = self.server.manager.export_session(session_id)
        if payload is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
            return
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="mini-swe-agent-{session_id}.traj.json"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_settings_snapshot(
        self, *, status: HTTPStatus = HTTPStatus.OK, extra: dict[str, Any] | None = None
    ) -> None:
        payload = self.server.manager.model_settings_snapshot()
        if extra:
            payload.update(extra)
        self._send_json(status, payload)

    @staticmethod
    def _settings_path_id(path: str, prefixes: tuple[str, ...]) -> tuple[str, str] | None:
        """Return ``(resource_id, action)`` for a settings sub-path."""
        clean = path.rstrip("/")
        action = ""
        for candidate in ("enable", "disable", "toggle"):
            suffix = "/" + candidate
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                action = candidate
                break
        for prefix in prefixes:
            if clean.startswith(prefix):
                resource_id = unquote(clean.removeprefix(prefix).strip("/"))
                if resource_id:
                    return resource_id, action
        return None

    @staticmethod
    def _provider_models_target(path: str) -> tuple[str, str] | None:
        """Return ``(provider_id, action)`` for provider model-query endpoints."""
        clean = path.rstrip("/")
        suffixes = (
            ("/models/refresh", "refresh"),
            ("/models/load", "load"),
            ("/models/fetch", "fetch"),
            ("/models/discover", "discover"),
            ("/fetch-models", "fetch"),
            ("/load-models", "load"),
            ("/refresh-models", "refresh"),
            ("/discover-models", "discover"),
            ("/query-models", "query"),
            ("/import-models", "import"),
            ("/available-models", "available"),
            ("/model-list", "models"),
            ("/fetch_models", "fetch"),
            ("/load_models", "load"),
            ("/refresh_models", "refresh"),
            ("/fetch", "fetch"),
            ("/load", "load"),
            ("/refresh", "refresh"),
            ("/query", "query"),
            ("/models", "models"),
        )
        for suffix, action in suffixes:
            if not clean.endswith(suffix):
                continue
            root = clean[: -len(suffix)]
            for prefix in ("/api/settings/providers/", "/api/providers/"):
                if root.startswith(prefix):
                    provider_id = unquote(root.removeprefix(prefix).strip("/"))
                    if provider_id:
                        return provider_id, action
        return None

    def _dispatch_model_settings_write(self, method: str, path: str) -> bool:
        """Handle provider/model/settings writes shared by POST, PUT and PATCH.

        Returns ``True`` when the request was handled.  Keeping all mutation
        paths in one place makes the small API easy to extend without adding a
        web framework dependency.
        """
        active_model_paths = {
            "/api/settings/model",
            "/api/models/active",
            "/api/model",
            "/api/model/switch",
            "/api/model/select",
            "/api/models/select",
            "/api/models/activate",
            "/api/settings/models/active",
        }
        provider_collection_paths = {"/api/providers", "/api/settings/providers"}
        model_collection_paths = {"/api/models", "/api/settings/models"}
        provider_prefixes = ("/api/settings/providers/", "/api/providers/")
        model_prefixes = ("/api/settings/models/", "/api/models/")

        try:
            provider_models_target = self._provider_models_target(path)
            if method in ("POST", "PUT", "PATCH") and provider_models_target is not None:
                provider_id, _ = provider_models_target
                payload = self._read_json_body() if self.headers.get("Content-Length", "0") not in ("", "0") else {}
                self._send_json(HTTPStatus.OK, self.server.manager.load_provider_models(provider_id, payload))
                return True
            if method == "POST" and path in provider_collection_paths:
                payload = self._read_json_body()
                provider = self.server.manager.add_provider(payload)
                self._send_settings_snapshot(status=HTTPStatus.CREATED, extra={"provider": provider})
                return True
            if method == "POST" and path in model_collection_paths:
                payload = self._read_json_body()
                model = self.server.manager.add_model(payload)
                self._send_settings_snapshot(status=HTTPStatus.CREATED, extra={"model": model})
                return True
            if method == "POST" and path in active_model_paths:
                payload = self._read_json_body()
                model_name = payload.get("model_name") or payload.get("model") or payload.get("active_model")
                if not isinstance(model_name, str) or not model_name.strip():
                    raise ValueError("model_name must be a non-empty string")
                model = self.server.manager.set_active_model(model_name.strip())
                self._send_settings_snapshot(extra={"model": model, "active_model": model["model_name"]})
                return True
            if method in ("PUT", "PATCH") and path.rstrip("/") in ("/api/settings", "/api/settings/model"):
                payload = self._read_json_body()
                if path.rstrip("/") == "/api/settings" and (
                    isinstance(payload.get("providers"), list) or isinstance(payload.get("models"), list)
                ):
                    self.server.manager.model_settings.apply_settings(payload)
                    self._send_settings_snapshot()
                    return True
                model_name = payload.get("model_name") or payload.get("model") or payload.get("active_model")
                if model_name is not None:
                    if not isinstance(model_name, str) or not model_name.strip():
                        raise ValueError("model_name must be a non-empty string")
                    self.server.manager.set_active_model(model_name.strip())
                self._send_settings_snapshot()
                return True

            provider_target = self._settings_path_id(path, provider_prefixes)
            if provider_target is not None:
                provider_id, action = provider_target
                if method == "DELETE":
                    provider = self.server.manager.delete_provider(provider_id)
                    self._send_settings_snapshot(extra={"deleted_provider": provider})
                    return True
                if method in ("PUT", "PATCH") or (method == "POST" and action == ""):
                    payload = self._read_json_body() if action == "" else {}
                    if action == "enable":
                        payload["enabled"] = True
                    elif action == "disable":
                        payload["enabled"] = False
                    provider = self.server.manager.update_provider(provider_id, payload)
                    self._send_settings_snapshot(extra={"provider": provider})
                    return True
                if method == "POST" and action:
                    if action == "toggle":
                        provider = self.server.manager.get_provider(provider_id)
                        enabled = not bool(provider.get("enabled", True))
                    else:
                        enabled = action == "enable"
                    provider = self.server.manager.set_provider_enabled(provider_id, enabled)
                    self._send_settings_snapshot(extra={"provider": provider})
                    return True

            model_target = self._settings_path_id(path, model_prefixes)
            if model_target is not None:
                model_name, action = model_target
                if method == "DELETE":
                    model = self.server.manager.delete_model(model_name)
                    self._send_settings_snapshot(extra={"deleted_model": model})
                    return True
                if method in ("PUT", "PATCH") or (method == "POST" and action == ""):
                    payload = self._read_json_body() if action == "" else {}
                    if action == "enable":
                        payload["enabled"] = True
                    elif action == "disable":
                        payload["enabled"] = False
                    model = self.server.manager.update_model(model_name, payload)
                    self._send_settings_snapshot(extra={"model": model})
                    return True
                if method == "POST" and action:
                    if action == "toggle":
                        model = self.server.manager.get_model(model_name)
                        enabled = not bool(model.get("enabled", True))
                    else:
                        enabled = action == "enable"
                    model = self.server.manager.set_model_enabled(model_name, enabled)
                    self._send_settings_snapshot(extra={"model": model})
                    return True
        except SettingsConflictError as exc:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return True
        except SettingsNotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return True
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return True
        return False

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_html(_HTML)
        elif parsed.path in ("/health", "/api/health"):
            logger.info(
                "Health probe from %s:%s (User-Agent: %s)",
                self.client_address[0],
                self.client_address[1],
                self.headers.get("User-Agent", ""),
            )
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        elif parsed.path == "/api/config":
            settings_payload = self.server.manager.model_settings_snapshot()
            self._send_json(
                HTTPStatus.OK,
                {
                    "initial_task": self.server.manager.config.initial_task,
                    "model_name": self.server.manager.active_model_name,
                    "active_model": self.server.manager.active_model_name,
                    "workspace": self.server.manager.config.workspace,
                    "database_path": self.server.manager.database_path,
                    "db_path": self.server.manager.database_path,
                    "database": self.server.manager.database_info(),
                    "providers": settings_payload.get("providers", []),
                    "models": settings_payload.get("models", []),
                    "enabled_models": settings_payload.get("enabled_models", []),
                    "enabled_model_names": settings_payload.get("enabled_model_names", []),
                    "settings": settings_payload,
                },
            )
        elif parsed.path in ("/api/settings", "/api/settings/", "/api/providers", "/api/settings/providers"):
            self._send_settings_snapshot()
        elif parsed.path in ("/api/models", "/api/settings/models"):
            payload = self.server.manager.model_settings_snapshot()
            enabled_values = parse_qs(parsed.query).get("enabled", [])
            if enabled_values and enabled_values[0].strip().lower() in ("1", "true", "yes", "on"):
                payload["models"] = payload.get("enabled_models", [])
            self._send_json(HTTPStatus.OK, payload)
        elif parsed.path in ("/api/models/enabled", "/api/settings/models/enabled"):
            payload = self.server.manager.model_settings_snapshot()
            payload["models"] = payload.get("enabled_models", [])
            self._send_json(HTTPStatus.OK, payload)
        elif (provider_models_target := self._provider_models_target(parsed.path)) is not None:
            provider_id, _ = provider_models_target
            try:
                payload = self.server.manager.load_provider_models(provider_id)
            except SettingsNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, payload)
        elif parsed.path.startswith(("/api/providers/", "/api/settings/providers/")):
            prefixes = ("/api/settings/providers/", "/api/providers/")
            target = self._settings_path_id(parsed.path, prefixes)
            if target is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown provider"})
                return
            provider_id, _ = target
            try:
                provider = self.server.manager.model_settings.get_provider(provider_id)
            except SettingsNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_settings_snapshot(extra={"provider": provider})
        elif parsed.path.startswith(("/api/models/", "/api/settings/models/")):
            prefixes = ("/api/settings/models/", "/api/models/")
            target = self._settings_path_id(parsed.path, prefixes)
            if target is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown model"})
                return
            model_name, _ = target
            try:
                model = self.server.manager.model_settings.get_model(model_name)
            except SettingsNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_settings_snapshot(extra={"model": model})
        elif parsed.path in ("/api/database", "/api/database/path", "/api/info"):
            self._send_json(
                HTTPStatus.OK,
                {
                    "database_path": self.server.manager.database_path,
                    "db_path": self.server.manager.database_path,
                    "database": self.server.manager.database_info(),
                },
            )
        elif parsed.path.startswith("/api/export/"):
            self._send_conversation_export(parsed.path.removeprefix("/api/export/").strip("/"))
        elif parsed.path in ("/api/conversations/statuses", "/api/statuses", "/api/states"):
            counts = self.server.manager.state_counts()
            states = [state.value for state in ConversationState]
            self._send_json(
                HTTPStatus.OK,
                {
                    "count": len(states),
                    "states": states,
                    "state_counts": counts,
                    "status_counts": counts,
                },
            )
        elif parsed.path in (
            "/api/conversations",
            "/api/conversation",
            "/api/sessions",
            "/api/session",
            "/api/history",
        ):
            conversations = [session.summary() for session in self.server.manager.list_sessions()]
            counts = self.server.manager.state_counts()
            states = [state.value for state in ConversationState]
            self._send_json(
                HTTPStatus.OK,
                {
                    "conversations": conversations,
                    "sessions": conversations,
                    "items": conversations,
                    "states": states,
                    "state_count": len(states),
                    "state_counts": counts,
                    "status_counts": counts,
                },
            )
        elif parsed.path.startswith(("/api/conversations/", "/api/conversation/", "/api/sessions/", "/api/session/")):
            prefix = next(
                prefix
                for prefix in ("/api/conversations/", "/api/conversation/", "/api/sessions/", "/api/session/")
                if parsed.path.startswith(prefix)
            )
            session_id = parsed.path.removeprefix(prefix).strip("/")
            if session_id.endswith("/export"):
                self._send_conversation_export(session_id.removesuffix("/export").strip("/"))
                return
            session = self.server.manager.get_session(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
                return
            with session.condition:
                events = list(session.events)
            self._send_json(
                HTTPStatus.OK,
                {"conversation": session.summary(), "session": session.summary(), "events": events},
            )
        elif parsed.path.startswith("/api/events/"):
            session_id = parsed.path.removeprefix("/api/events/").strip("/")
            session = self.server.manager.get_session(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
                return
            try:
                since = max(0, int(parse_qs(parsed.query).get("since", ["0"])[0]))
            except ValueError:
                since = 0
            self._stream_events(session, since=since)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > 64_000_000:
            raise ValueError("Invalid request body")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _stop_session(self, session_id: str) -> None:
        session = self.server.manager.get_session(session_id)
        if session is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
            return
        session.request_stop()
        self._send_json(HTTPStatus.ACCEPTED, {"status": "stopping", "session_id": session.session_id})

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if self._dispatch_model_settings_write("POST", parsed.path):
            return
        if parsed.path.startswith("/api/stop/"):
            session_id = parsed.path.removeprefix("/api/stop/").strip("/")
            self._stop_session(session_id)
            return
        if parsed.path == "/api/stop":
            try:
                session_id = self._read_json_body().get("session_id", "")
                if not isinstance(session_id, str):
                    raise ValueError("session_id must be a string")
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._stop_session(session_id)
            return
        branch_match = re.fullmatch(r'/api/(?:conversations|conversation|sessions|session)/([^/]+)/branch', parsed.path)
        if branch_match:
            source_id = unquote(branch_match.group(1))
            try:
                payload = self._read_json_body()
                task = payload.get('task', '')
                turn_index = payload.get('turn_index')
                if not isinstance(task, str):
                    raise ValueError('task must be a string')
                branch = self.server.manager.branch_session(source_id, turn_index, task)
                if branch is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {'error': 'Unknown session'})
                    return
            except RuntimeError as exc:
                self._send_json(HTTPStatus.CONFLICT, {'error': str(exc)})
                return
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {'error': str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - surface setup errors to UI
                logger.exception('Could not branch GUI session')
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': str(exc)})
                return
            self._send_json(
                HTTPStatus.ACCEPTED,
                {
                    'session_id': branch.session_id,
                    'events_url': f'/api/events/{branch.session_id}',
                    'conversation': branch.summary(),
                    'event_offset': 0,
                    'branched_from': source_id,
                    'turn_index': turn_index,
                },
            )
            return
        if parsed.path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            payload = self._read_json_body()
            task = payload.get("task", "")
            images = payload.get("images", [])
            session_id = payload.get("session_id") or ""
            model_name = payload.get("model_name") or payload.get("model") or ""
            if not isinstance(task, str):
                raise ValueError("task must be a string")
            if not isinstance(session_id, str):
                raise ValueError("session_id must be a string")
            if not isinstance(model_name, str):
                raise ValueError("model_name must be a string")
            if images is None:
                images = []
            if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
                raise ValueError("images must be a list of data URLs or image URLs")
            if len(images) > 8:
                raise ValueError("At most 8 images can be attached")
            for image in images:
                if not image.startswith(("data:image/", "http://", "https://")):
                    raise ValueError("images must be data URLs or http(s) URLs")
            if session_id:
                continued = self.server.manager.continue_session(
                    session_id, task, images=images, model_name=model_name or None
                )
                if continued is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
                    return
                session, event_offset = continued
            else:
                session = self.server.manager.start_session(task, images=images, model_name=model_name or None)
                event_offset = 0
        except RuntimeError as exc:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - surface setup errors to UI
            logger.exception("Could not start GUI session")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "session_id": session.session_id,
                "events_url": f"/api/events/{session.session_id}",
                "conversation": session.summary(),
                "event_offset": event_offset,
                "continued": bool(session_id),
            },
        )

    def do_PUT(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if not self._dispatch_model_settings_write("PUT", parsed.path):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if self._dispatch_model_settings_write("PATCH", parsed.path):
            return
        match = re.fullmatch(r'/api/(?:conversations|conversation|sessions|session)/([^/]+)', parsed.path)
        if not match:
            self._send_json(HTTPStatus.NOT_FOUND, {'error': 'Not found'})
            return
        session_id = unquote(match.group(1))
        try:
            payload = self._read_json_body()
            if 'title' not in payload and 'pinned' not in payload:
                raise ValueError('Provide title or pinned')
            if 'title' in payload:
                if not isinstance(payload['title'], str):
                    raise ValueError('title must be a string')
                session = self.server.manager.rename_session(session_id, payload['title'])
            else:
                session = self.server.manager.get_session(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {'error': 'Unknown session'})
                return
            if 'pinned' in payload:
                if not isinstance(payload['pinned'], bool):
                    raise ValueError('pinned must be a boolean')
                session = self.server.manager.set_session_pinned(session_id, payload['pinned'])
            self._send_json(HTTPStatus.OK, {'conversation': session.summary(), 'session': session.summary()})
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {'error': str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if self._dispatch_model_settings_write("DELETE", parsed.path):
            return
        match = re.fullmatch(r'/api/(?:conversations|conversation|sessions|session)/([^/]+)', parsed.path)
        if not match:
            self._send_json(HTTPStatus.NOT_FOUND, {'error': 'Not found'})
            return
        session_id = unquote(match.group(1))
        try:
            session = self.server.manager.delete_session(session_id)
        except RuntimeError as exc:
            self._send_json(HTTPStatus.CONFLICT, {'error': str(exc)})
            return
        if session is None:
            self._send_json(HTTPStatus.NOT_FOUND, {'error': 'Unknown session'})
            return
        self._send_json(HTTPStatus.OK, {'deleted': session_id})

    def _stream_events(self, session: GuiSession, *, since: int = 0) -> None:
        try:
            cursor = int(self.headers.get("Last-Event-ID", "0") or "0")
        except ValueError:
            cursor = 0
        # A stream opened for a follow-up run can skip the events that the
        # browser has already rendered.  Automatic reconnects send
        # Last-Event-ID, which takes precedence over the initial offset.
        if cursor <= 0:
            cursor = since
        with session.condition:
            # Event ids are 1-based; keep reconnects from replaying events or
            # waiting forever when they send an out-of-range id.
            cursor = max(0, min(cursor, len(session.events)))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                with session.condition:
                    while cursor >= len(session.events) and not session.done:
                        session.condition.wait(timeout=15.0)
                    batch = session.events[cursor:]
                    finished = session.done
                    cursor = len(session.events)
                if batch:
                    first_event_id = cursor - len(batch) + 1
                    for offset, event in enumerate(batch):
                        payload = json.dumps(event)
                        self.wfile.write(f"id: {first_event_id + offset}\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                elif not finished:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                if finished and cursor >= len(session.events):
                    break
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("GUI event stream disconnected")
        finally:
            self.close_connection = True


def create_server(host: str, port: int, manager: GuiSessionManager) -> GuiHTTPServer:
    """Create (but do not start) the GUI HTTP server."""
    return GuiHTTPServer((host, port), manager)


def serve(
    manager: GuiSessionManager,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Serve the GUI until interrupted."""
    server = create_server(host, port, manager)
    actual_host, actual_port = server.server_address[:2]
    url_host = "127.0.0.1" if actual_host in ("0.0.0.0", "::") else actual_host
    url = f"http://{url_host}:{actual_port}"
    Console().print(f"[bold green]mini-SWE-agent GUI:[/bold green] {url}", soft_wrap=True)
    Console().print(f"[bold green]Conversation database:[/bold green] {manager.database_path}", soft_wrap=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        Console().print("\n[dim]Shutting down GUI...[/dim]")
    finally:
        server.server_close()


app = typer.Typer(rich_markup_mode="rich", add_completion=False)


@app.command(help="Launch a basic web GUI for mini-SWE-agent with streaming responses.")
def main(
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use"),
    agent_class: str | None = typer.Option(
        None, "--agent-class", help="Agent class to use (the default agent is recommended for the GUI)"
    ),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use"),
    task: str | None = typer.Option(None, "-t", "--task", help="Initial task to prefill in the GUI"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Path to config files or key=value config overrides"
    ),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Trajectory output path"),
    database: Path = typer.Option(
        DEFAULT_DATABASE_FILE,
        "--database",
        "--db",
        help="SQLite conversation database path (default: gui_conversations.db in the global config directory)",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind"),
    port: int = typer.Option(8000, "--port", help="Port to bind (0 chooses a free port)"),
    open_browser: bool = typer.Option(True, "--browser/--no-browser", help="Open the GUI in a browser"),
) -> None:
    configure_if_first_time()
    gui_config = build_gui_config(
        config_spec=config_spec,
        model_name=model_name,
        model_class=model_class,
        agent_class=agent_class,
        environment_class=environment_class,
        output=output,
        initial_task=task or "",
    )
    manager = GuiSessionManager(gui_config, db_path=database)
    serve(manager, host=host, port=port, open_browser=open_browser)


if __name__ == "__main__":
    app()
