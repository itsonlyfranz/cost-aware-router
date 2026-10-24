"""Deterministic offline simulation used by the demo and tests (no network, no API key)."""

from __future__ import annotations

import json

from cost_aware_router.client import FakeClient
from cost_aware_router.models import LLMRequest, Task, Tier

CAPABILITY = {"fast": 0.45, "balanced": 0.75, "premium": 1.0}


def simulated_client(difficulty: dict[str, float], settings) -> FakeClient:
    """A client whose models answer confidently only when capability >= the prompt's hidden difficulty.

    A model below the needed capability reports low confidence, mimicking calibrated self-assessment.
    """
    tier_of = {m.name: m.tier for m in settings.models()}
    cap = {Tier.FAST: CAPABILITY["fast"], Tier.BALANCED: CAPABILITY["balanced"], Tier.PREMIUM: CAPABILITY["premium"]}

    def handler(req: LLMRequest) -> str:
        d = difficulty.get(req.prompt, 0.5)
        capable = cap[tier_of[req.model]] >= d
        words = 12 + int(d * 160)  # harder tasks produce longer answers
        answer = " ".join(["word"] * words)
        return json.dumps({"answer": answer, "confidence": 0.93 if capable else 0.4})

    return FakeClient(handler)


def demo_workload() -> list[tuple[Task, float]]:
    """A realistic skewed mix: mostly easy traffic, a tail of hard tasks. Returns (task, hidden difficulty)."""
    easy = [
        "What is the capital of Australia?",
        "Define idempotent in one sentence.",
        "Translate 'good morning' to Spanish.",
        "Convert 72 degrees Fahrenheit to Celsius.",
        "What is the HTTP status code for Not Found?",
        "List three primary colors.",
        "Is 17 prime? yes or no",
        "What is a JSON object?",
    ]
    medium = [
        "Summarize the main differences between TCP and UDP and when to use each in a production service, with examples.",
        "Write a Python function that merges overlapping intervals and explain its time complexity briefly for a junior engineer.",
        "Explain how a database index speeds up queries and name two situations where an index hurts write performance.",
        "Draft a short incident postmortem outline for a 30 minute API outage caused by an expired certificate.",
    ]
    hard = [
        "Design the architecture for a multi-tenant rate limiter. Analyze the trade-off between accuracy and latency, "
        "then explain how you would handle clock skew and race conditions.\n1. data model\n2. failure modes\n3. rollout plan",
        "Debug and optimize this code:\n```python\ndef f(xs):\n    out = []\n    for x in xs:\n        if x in out:\n            continue\n        out.append(x)\n    return out\n```\n"
        "Then prove the complexity of your fix step by step and analyze concurrency safety.",
        "Analyze the security trade-offs of migrating session tokens from cookies to signed JWTs and design a safe migration plan "
        "with a formal threat model.\n- threats\n- mitigations\n- rollout",
    ]
    out: list[tuple[Task, float]] = []
    for i in range(3):  # repeat easy/medium so the mix is easy-heavy (~70% easy, 20% medium, 10% hard)
        for p in easy:
            out.append((Task(prompt=f"{p} (#{i})"), 0.25))
    for i in range(2):
        for p in medium:
            out.append((Task(prompt=f"{p} (#{i})"), 0.7))
    for p in hard:
        out.append((Task(prompt=p), 0.95))
    # one task hard to the router's heuristic: looks easy, is actually hard -> exercises escalation
    out.append((Task(prompt="What is the best way to resolve this merge?"), 0.9))
    return out

