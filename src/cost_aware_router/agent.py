"""The router: complexity -> starting tier -> confidence-gated escalation under a token budget."""

from __future__ import annotations

from cost_aware_router.analytics import DecisionStore
from cost_aware_router.budget import MIN_USEFUL_OUTPUT, BudgetExceeded, TokenBudget
from cost_aware_router.client import LLMClient, estimate_tokens
from cost_aware_router.complexity import score_complexity
from cost_aware_router.confidence import SYSTEM_PROMPT, parse_response
from cost_aware_router.config import Settings
from cost_aware_router.models import Attempt, Decision, ExitReason, LLMRequest, ModelSpec, Task, Tier


class CostAwareRouter:
    def __init__(
        self,
        client: LLMClient,
        settings: Settings | None = None,
        store: DecisionStore | None = None,
        max_output_tokens: int = 512,
        strict_budget: bool = False,
    ) -> None:
        self.client = client
        self.settings = settings or Settings()
        self.store = store
        self.max_output_tokens = max_output_tokens
        self.strict_budget = strict_budget
        self.by_tier: dict[Tier, ModelSpec] = {m.tier: m for m in self.settings.models()}

    def run(self, task: Task | str) -> Decision:
        if isinstance(task, str):
            task = Task(prompt=task)
        cx = score_complexity(task.prompt)
        budget = TokenBudget(task.budget_tokens, task.budget_usd)
        baseline = self.by_tier[self.settings.baseline_tier]
        input_est = estimate_tokens(SYSTEM_PROMPT + task.prompt)

        decision = Decision(
            task_id=task.id,
            prompt_preview=task.prompt[:80],
            complexity_score=cx.score,
            complexity_tier=cx.tier,
            route_reason=f"{cx.tier.name} tier: {cx.reason}",
            exit_reason=ExitReason.NO_BUDGET,
            baseline_model=baseline.name,
            budget_tokens=task.budget_tokens,
            budget_usd=task.budget_usd,
        )

        tier = cx.tier
        last_output_tokens = 0
        while True:
            floor = Tier(decision.attempts[-1].tier + 1) if decision.attempts else Tier.FAST
            spec, allowance = self._affordable(tier, floor, budget, input_est)
            if spec is None:
                decision.exit_reason = ExitReason.BUDGET if decision.attempts else ExitReason.NO_BUDGET
                if self.strict_budget:
                    raise BudgetExceeded(f"task {task.id}: no model fits the remaining budget")
                if decision.attempts:  # fall back to the best answer we already paid for
                    self._accept(decision, decision.attempts[-1])
                break
            if spec.tier != tier:
                decision.route_reason += f"; downgraded {tier.name}->{spec.tier.name} to fit budget"
                tier = spec.tier

            resp = self.client.complete(
                LLMRequest(model=spec.name, prompt=task.prompt, system=SYSTEM_PROMPT, max_tokens=allowance)
            )
            answer, conf = parse_response(resp.text)
            cost = budget.charge(spec, resp.usage)
            attempt = Attempt(
                model=spec.name, tier=spec.tier, usage=resp.usage, cost_usd=cost, confidence=conf, answer=answer
            )
            decision.attempts.append(attempt)
            last_output_tokens = resp.usage.output_tokens

            if conf is not None and conf >= self.settings.confidence_threshold:
                decision.exit_reason = ExitReason.CONFIDENT
                self._accept(decision, attempt)
                break
            if tier == Tier.PREMIUM:
                decision.exit_reason = ExitReason.TOP_TIER
                self._accept(decision, attempt)
                break
            tier = Tier(tier + 1)  # not confident enough: escalate one tier

        decision.total_tokens = budget.spent_tokens
        decision.cost_usd = budget.spent_usd
        # Counterfactual: the same prompt and answer length on the baseline model, one call, no cascade.
        out = last_output_tokens or self.max_output_tokens // 2
        decision.baseline_cost_usd = baseline.cost(input_est, out)
        if self.store:
            self.store.append(decision)
        return decision

    def _affordable(self, tier: Tier, floor: Tier, budget: TokenBudget, input_est: int) -> tuple[ModelSpec | None, int]:
        """Pick the requested tier, or the nearest cheaper tier (not below `floor`) that fits the budget."""
        for t in range(tier, floor - 1, -1):
            spec = self.by_tier[Tier(t)]
            allowance = budget.max_output_for(spec, input_est, self.max_output_tokens)
            if allowance >= MIN_USEFUL_OUTPUT:
                return spec, allowance
        return None, 0

    @staticmethod
    def _accept(decision: Decision, attempt: Attempt) -> None:
        attempt.accepted = True
        decision.final_model = attempt.model
        decision.answer = attempt.answer
        decision.confidence = attempt.confidence
