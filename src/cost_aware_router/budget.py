"""Per-task token / USD budget enforcement."""

from __future__ import annotations

from cost_aware_router.models import ModelSpec, Usage

MIN_USEFUL_OUTPUT = 32  # below this many output tokens a call is not worth making


class BudgetExceeded(Exception):
    """Raised when a call cannot be afforded and the caller asked for strict enforcement."""


class TokenBudget:
    def __init__(self, max_tokens: int | None = None, max_usd: float | None = None) -> None:
        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.spent_tokens = 0
        self.spent_usd = 0.0

    @property
    def remaining_tokens(self) -> int | None:
        return None if self.max_tokens is None else max(0, self.max_tokens - self.spent_tokens)

    @property
    def remaining_usd(self) -> float | None:
        return None if self.max_usd is None else max(0.0, self.max_usd - self.spent_usd)

    def max_output_for(self, spec: ModelSpec, input_tokens: int, wanted: int) -> int:
        """Largest output allowance (<= wanted) that keeps this call inside both budgets; 0 if none."""
        allowed = wanted
        if (rt := self.remaining_tokens) is not None:
            allowed = min(allowed, rt - input_tokens)
        if (ru := self.remaining_usd) is not None:
            input_cost = spec.cost(input_tokens, 0)
            if spec.output_per_mtok > 0:
                allowed = min(allowed, int((ru - input_cost) * 1_000_000 / spec.output_per_mtok))
            elif input_cost > ru:
                allowed = 0
        return allowed if allowed >= MIN_USEFUL_OUTPUT else 0

    def charge(self, spec: ModelSpec, usage: Usage) -> float:
        """Record actual usage (even if it overshoots: the money is already spent). Returns USD cost."""
        cost = spec.cost(usage.input_tokens, usage.output_tokens)
        self.spent_tokens += usage.total
        self.spent_usd += cost
        return cost

    @property
    def exhausted(self) -> bool:
        return (self.remaining_tokens == 0) or (self.remaining_usd == 0)
