"""Provider-agnostic text generation clients for live model evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
    "gemini": "gemini-2.5-flash",
    "codex": "gpt-5-codex",
}


PROVIDER_PACKAGE_EXTRAS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "ollama": "openai",
    "gemini": "gemini",
    "codex": "openai",
}


PROVIDER_AUTH_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY or GOOGLE_API_KEY",
    "codex": "OPENAI_API_KEY",
}


@dataclass
class GeneratedResponse:
    text: str
    raw: Any
    input_tokens: int = 0
    output_tokens: int = 0


class ModelClient(Protocol):
    provider: str

    def generate(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        **kwargs: Any,
    ) -> GeneratedResponse: ...


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _extract_openai_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_obj = getattr(item, "text", None)
                if isinstance(text_obj, str):
                    parts.append(text_obj)
                elif text_obj is not None and hasattr(text_obj, "value"):
                    parts.append(str(text_obj.value))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _extract_usage_value(usage: Any, *names: str) -> int:
    if usage is None:
        return 0
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def _read_claude_code_oauth() -> dict[str, Any] | None:
    """Read Claude Code OAuth credentials from the macOS Keychain.

    Returns the OAuth dict with accessToken/refreshToken/expiresAt,
    or None if unavailable (not macOS, not logged in, etc.).
    """
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-s", "Claude Code-credentials",
                "-w",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0:
            return None
        creds = json.loads(result.stdout.strip())
        oauth = creds.get("claudeAiOauth")
        if not isinstance(oauth, dict) or "accessToken" not in oauth:
            return None
        # Check if token is expired (expiresAt is ms since epoch)
        expires_at = oauth.get("expiresAt", 0)
        if expires_at and time.time() * 1000 > expires_at:
            return None
        return oauth
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


class AnthropicModelClient:
    provider = "anthropic"

    def __init__(self, client: Any | None = None, oauth_token: str | None = None) -> None:
        if client is None:
            from anthropic import Anthropic
            if oauth_token:
                # Use OAuth Bearer token from Claude Code session
                client = Anthropic(auth_token=oauth_token)
            else:
                client = Anthropic()
        self._client = client

    def generate(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        **kwargs: Any,
    ) -> GeneratedResponse:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        if system:
            request["system"] = system
        response = self._client.messages.create(**request)
        text = response.content[0].text if getattr(response, "content", None) else ""
        usage = getattr(response, "usage", None)
        return GeneratedResponse(
            text=text,
            raw=response,
            input_tokens=_extract_usage_value(usage, "input_tokens"),
            output_tokens=_extract_usage_value(usage, "output_tokens"),
        )


class OpenAIModelClient:
    provider = "openai"

    def __init__(self, client: Any | None = None, base_url: str | None = None) -> None:
        if client is None:
            from openai import OpenAI
            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
                # Local servers (Ollama, LM Studio, vLLM) don't need a real key
                if not os.environ.get("OPENAI_API_KEY"):
                    kwargs["api_key"] = "not-needed"
            client = OpenAI(**kwargs)
        self._client = client

    def generate(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        **kwargs: Any,
    ) -> GeneratedResponse:
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}] + payload

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": str(message.get("role", "user")),
                    "content": _coerce_text(message.get("content", "")),
                }
                for message in payload
            ],
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        choice = response.choices[0].message
        text = _extract_openai_text(choice)
        usage = getattr(response, "usage", None)
        return GeneratedResponse(
            text=text,
            raw=response,
            input_tokens=_extract_usage_value(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_extract_usage_value(usage, "completion_tokens", "output_tokens"),
        )


class GeminiModelClient:
    """Google Gemini model client using the google-genai SDK."""

    provider = "gemini"

    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client

    def generate(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        **kwargs: Any,
    ) -> GeneratedResponse:
        try:
            from google.genai import types
        except ImportError:
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                gemini_role = "model" if role == "assistant" else "user"
                text = _coerce_text(msg.get("content", ""))
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": text}],
                })
            config: Any = {"max_output_tokens": max_tokens}
            if system:
                config["system_instruction"] = system
        else:
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                # Gemini uses "user" and "model" roles
                gemini_role = "model" if role == "assistant" else "user"
                text = _coerce_text(msg.get("content", ""))
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part(text=text)],
                ))

            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
            )
            if system:
                config.system_instruction = system

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return GeneratedResponse(
            text=text,
            raw=response,
            input_tokens=_extract_usage_value(
                usage, "prompt_token_count", "input_tokens",
            ),
            output_tokens=_extract_usage_value(
                usage, "candidates_token_count", "output_tokens",
            ),
        )


class CodexModelClient:
    """OpenAI Codex client using the Responses API."""

    provider = "codex"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    def generate(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        system: str | None = None,
        **kwargs: Any,
    ) -> GeneratedResponse:
        input_items: list[dict[str, Any]] = []
        if system:
            input_items.append({"role": "system", "content": system})
        for msg in messages:
            input_items.append({
                "role": str(msg.get("role", "user")),
                "content": _coerce_text(msg.get("content", "")),
            })

        response = self._client.responses.create(
            model=model,
            input=input_items,
            max_output_tokens=max_tokens,
            **kwargs,
        )

        # Extract text from response output items
        text_parts: list[str] = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(part.text)
        text = "\n".join(text_parts)

        usage = getattr(response, "usage", None)
        return GeneratedResponse(
            text=text,
            raw=response,
            input_tokens=_extract_usage_value(usage, "input_tokens"),
            output_tokens=_extract_usage_value(usage, "output_tokens"),
        )


def create_model_client(
    provider: str = "anthropic",
    *,
    client: Any | None = None,
    base_url: str | None = None,
) -> ModelClient:
    chosen = provider.lower()
    if chosen == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY") and client is None:
            # Try Claude Code OAuth session before giving up
            oauth = _read_claude_code_oauth()
            if oauth:
                return AnthropicModelClient(oauth_token=oauth["accessToken"])
            raise RuntimeError("ANTHROPIC_API_KEY not set and no Claude Code session found")
        return AnthropicModelClient(client=client)
    if chosen == "openai":
        if not base_url and not os.environ.get("OPENAI_API_KEY") and client is None:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIModelClient(client=client, base_url=base_url)
    if chosen == "ollama":
        url = base_url or "http://localhost:11434/v1"
        return OpenAIModelClient(base_url=url)
    if chosen == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key and client is None:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return GeminiModelClient(client=client, api_key=api_key)
    if chosen == "codex":
        if not os.environ.get("OPENAI_API_KEY") and client is None:
            raise RuntimeError("OPENAI_API_KEY not set (required for Codex)")
        return CodexModelClient(client=client)
    raise ValueError(f"Unsupported model provider: {provider}")
