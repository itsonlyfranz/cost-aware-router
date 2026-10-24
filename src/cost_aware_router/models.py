"""Pydantic data models shared across the router."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class Tier(IntEnum):
    """Model capability/cost tiers, ordered cheapest to most capable."""

    FAST = 0
    BALANCED = 1
    PREMIUM = 2


class ModelSpec(BaseModel):
    name: str
    tier: Tier
    input_per_mtok: float = Field(ge=0, description="USD per million input tokens")
    output_per_mtok: float = Field(ge=0, description="USD per million output tokens")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMRequest(BaseModel):
    model: str
    prompt: str
    system: str | None = None
    max_tokens: int = 1024


class LLMResponse(BaseModel):
    text: str
    usage: Usage
    model: str


class Task(BaseModel):
    prompt: str
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    budget_tokens: int | None = Field(default=None, gt=0, description="Max input+output tokens for the whole task")
    budget_usd: float | None = Field(default=None, gt=0, description="Max spend in USD for the whole task")


class Attempt(BaseModel):
    model: str
    tier: Tier
    usage: Usage
    cost_usd: float
    confidence: float | None = None
    answer: str = ""
    accepted: bool = False


class ExitReason:
    CONFIDENT = "confident_early_exit"
    TOP_TIER = "top_tier_reached"
    BUDGET = "budget_exhausted"
    NO_BUDGET = "no_affordable_model"


class Decision(BaseModel):
    """One routing decision for one task: what ran, what it cost, and why it stopped."""

    task_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_preview: str
    complexity_score: float
    complexity_tier: Tier
    route_reason: str
    attempts: list[Attempt] = Field(default_factory=list)
    final_model: str | None = None
    answer: str = ""
    confidence: float | None = None
    exit_reason: str
    total_tokens: int = 0
    cost_usd: float = 0.0
    baseline_model: str
    baseline_cost_usd: float = 0.0
    budget_tokens: int | None = None
    budget_usd: float | None = None

    @property
    def savings_usd(self) -> float:
        return self.baseline_cost_usd - self.cost_usd
