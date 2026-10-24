import sys
from types import SimpleNamespace

from cost_aware_router.client import AnthropicClient
from cost_aware_router.models import LLMRequest


def test_anthropic_client_maps_request_and_usage(monkeypatch):
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="hello")],
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    fake_sdk = SimpleNamespace(Anthropic=lambda api_key=None: SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
    r = AnthropicClient("k").complete(LLMRequest(model="m", prompt="p", system="s", max_tokens=5))
    assert (r.text, r.usage.input_tokens, r.usage.output_tokens) == ("hello", 11, 7)
    assert seen == {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "p"}], "system": "s"}
