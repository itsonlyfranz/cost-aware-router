import sys
from types import SimpleNamespace

from cost_aware_router.client import AnthropicClient
from cost_aware_router.models import LLMRequest


def test_anthropic_client_maps_request_and_usage(monkeypatch):
    seen = {}

    class FakeChat:
        def __init__(self, **kw):
            seen["init"] = kw

        def invoke(self, messages):
            seen["messages"] = [(type(m).__name__, m.content) for m in messages]
            return SimpleNamespace(
                text=lambda: "hello", usage_metadata={"input_tokens": 11, "output_tokens": 7}
            )

    monkeypatch.setitem(sys.modules, "langchain_anthropic", SimpleNamespace(ChatAnthropic=FakeChat))
    r = AnthropicClient("k").complete(LLMRequest(model="m", prompt="p", system="s", max_tokens=5))
    assert (r.text, r.usage.input_tokens, r.usage.output_tokens) == ("hello", 11, 7)
    assert seen["init"] == {"model": "m", "max_tokens": 5, "api_key": "k"}
    assert seen["messages"] == [("SystemMessage", "s"), ("HumanMessage", "p")]
