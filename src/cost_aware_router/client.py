"""LLM client interface plus the Anthropic implementation and offline fakes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from cost_aware_router.models import LLMRequest, LLMResponse, Usage


def estimate_tokens(text: str) -> int:
    """Cheap pre-flight estimate (~4 chars/token). Real usage is charged after each call."""
    return max(1, -(-len(text) // 4))


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class AnthropicClient:
    """Thin wrapper over the official `anthropic` SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system
        msg = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return LLMResponse(
            text=text,
            usage=Usage(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens),
            model=request.model,
        )


class FakeClient:
    """Scriptable offline client. `handler` maps a request to response text; tokens are estimated.

    Records every request in `calls` so tests can assert which models were used.
    """

    def __init__(self, handler: Callable[[LLMRequest], str] | str = "ok") -> None:
        self._handler = handler if callable(handler) else (lambda _r: handler)
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        text = self._handler(request)
        out = min(estimate_tokens(text), request.max_tokens)
        return LLMResponse(
            text=text,
            usage=Usage(input_tokens=estimate_tokens((request.system or "") + request.prompt), output_tokens=out),
            model=request.model,
        )
