import json

import pytest

from cost_aware_router.client import FakeClient
from cost_aware_router.config import Settings
from cost_aware_router.models import LLMRequest


@pytest.fixture
def settings() -> Settings:
    return Settings()


def reply(confidence: float | None, answer: str = "an answer " * 20) -> str:
    obj: dict = {"answer": answer}
    if confidence is not None:
        obj["confidence"] = confidence
    return json.dumps(obj)


def by_model(settings: Settings, conf: dict[str, float | None]) -> FakeClient:
    """FakeClient whose reported confidence depends on the model tier name (fast/balanced/premium)."""
    names = {m.name: m.tier.name.lower() for m in settings.models()}
    return FakeClient(lambda r: reply(conf[names[r.model]]))


EASY = "What is the capital of France?"
HARD = (
    "Design the architecture for a distributed lock service. Analyze the trade-off between safety and "
    "availability, then debug race conditions.\n1. model\n2. failures\n3. rollout"
)
