"""Heuristic prompt-complexity scoring (no LLM call, so routing itself costs nothing)."""

from __future__ import annotations

import re

from pydantic import BaseModel

from cost_aware_router.models import Tier

HARD_WORDS = (
    "prove", "derive", "architecture", "design", "trade-off", "tradeoff", "optimize", "debug",
    "refactor", "analyze", "analyse", "step by step", "algorithm", "concurrency", "race condition",
    "security", "migrate", "formal", "complexity of",
)
MEDIUM_WORDS = ("explain", "summarize", "summarise", "write a", "draft", "compare", "function", "describe", "outline")
EASY_WORDS = ("what is", "define", "translate", "capital of", "spell", "convert", "list ", "yes or no")

FAST_BELOW = 0.30
BALANCED_BELOW = 0.60


class Complexity(BaseModel):
    score: float
    tier: Tier
    reason: str


def score_complexity(prompt: str) -> Complexity:
    text = prompt.lower()
    words = len(prompt.split())
    parts: list[tuple[float, str]] = []

    parts.append((min(words / 200, 1.0) * 0.35, f"{words} words"))
    hard = sum(w in text for w in HARD_WORDS)
    if hard:
        parts.append((min(hard * 0.18, 0.45), f"{hard} reasoning keywords"))
    medium = sum(w in text for w in MEDIUM_WORDS)
    if medium:
        parts.append((min(medium * 0.17, 0.34), f"{medium} task keywords"))
    if "```" in prompt or re.search(r"\bdef \w+\(|\bclass \w+|\{\s*\n", prompt):
        parts.append((0.2, "contains code"))
    steps = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s", prompt)) + text.count(" then ")
    if steps >= 2:
        parts.append((min(steps * 0.07, 0.25), f"{steps} steps"))
    if sum(w in text for w in EASY_WORDS) and words < 25:
        parts.append((-0.15, "simple lookup phrasing"))

    score = max(0.0, min(1.0, sum(p for p, _ in parts)))
    tier = Tier.FAST if score < FAST_BELOW else Tier.BALANCED if score < BALANCED_BELOW else Tier.PREMIUM
    return Complexity(score=round(score, 3), tier=tier, reason=", ".join(r for _, r in parts) or "trivial prompt")
