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
    """Claude via `langchain-anthropic` (ChatAnthropic), which wraps the official `anthropic` SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def complete(self, request: LLMRequest) -> LLMResponse:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = ChatAnthropic(model=request.model, max_tokens=request.max_tokens, api_key=self._api_key)
        messages = ([SystemMessage(request.system)] if request.system else []) + [HumanMessage(request.prompt)]
        msg = chat.invoke(messages)
        usage = msg.usage_metadata or {}
        text = msg.text() if callable(msg.text) else msg.text
        return LLMResponse(
            text=text,
            usage=Usage(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0)),
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
