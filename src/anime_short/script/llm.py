"""OpenAI-compatible structured-output client for the script/plan stages (SPEC §5).

Structured output: both OpenAI and the local vLLM server (qwen3-vl-8b, checked directly against
`http://localhost:8093/v1`) honour `response_format={"type": "json_schema", ...}` and constrain
decoding to the given JSON Schema — including optional fields with defaults and `minItems`/
`maxItems` on arrays (used by the plan stage to force an exact shot count). We rely on that
grammar-level constraint rather than prompt-only "return JSON" instructions, and layer Pydantic
validation + retry on top for anything outside what the schema itself can express.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from anime_short.config import LLMConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM fails to produce a schema-valid response after all retries."""


class ChatClient(Protocol):
    """The slice of the `openai` client `LLMClient` needs — fakeable in tests."""

    chat: Any


class LLMClient:
    """Thin structured-output wrapper. Pass `client=` to inject a fake for tests."""

    def __init__(self, config: LLMConfig, client: ChatClient | None = None) -> None:
        self._client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )
        self._model = config.model
        self._temperature = config.temperature
        self._max_retries = config.max_retries
        self._reasoning_effort = config.reasoning_effort

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
    ) -> T:
        """Call the chat completion API and validate the response against `schema`.

        Retries up to `max_retries` times on invalid JSON / schema mismatch, feeding the previous
        output and the error back to the model.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        json_schema = schema.model_json_schema()
        json_schema["additionalProperties"] = False
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True},
        }

        extra: dict[str, Any] = {}
        if self._reasoning_effort:
            extra["reasoning_effort"] = self._reasoning_effort

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **extra,
            )
            content = response.choices[0].message.content or ""
            try:
                data = json.loads(content)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                log.warning("LLM response failed validation (attempt %d): %s", attempt + 1, exc)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That response was invalid: {exc}\n"
                            "Return ONLY corrected JSON matching the schema."
                        ),
                    }
                )
        raise LLMError(
            f"LLM did not produce a schema-valid response after {self._max_retries + 1} attempts: "
            f"{last_error}"
        )
