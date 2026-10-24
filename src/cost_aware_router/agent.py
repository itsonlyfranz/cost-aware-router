"""The router as a LangGraph StateGraph: classify -> plan_call -> call_model -> evaluate -> (loop | finalize)."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from cost_aware_router.analytics import DecisionStore
from cost_aware_router.budget import MIN_USEFUL_OUTPUT, BudgetExceeded, TokenBudget
from cost_aware_router.client import LLMClient, estimate_tokens
from cost_aware_router.complexity import score_complexity
from cost_aware_router.confidence import SYSTEM_PROMPT, parse_response
from cost_aware_router.config import Settings
from cost_aware_router.models import Attempt, Decision, ExitReason, LLMRequest, ModelSpec, Task, Tier

# Each escalation round is plan_call + call_model + evaluate (3 steps); classify and finalize add 2.
# Three tiers therefore need at most 11 steps; the limit is a safety net against routing loops.
RECURSION_LIMIT = 25

# Pass to a checkpointer (e.g. MemorySaver(serde=CHECKPOINT_SERDE)) so RouterState round-trips.
CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[("cost_aware_router.models", n) for n in ("Task", "Tier", "ModelSpec", "Attempt", "Usage", "Decision")]
)


class RouterState(BaseModel):
    """Everything the graph knows about one task; checkpointed after every node when a checkpointer is set."""

    task: Task
    complexity_score: float = 0.0
    complexity_tier: Tier = Tier.FAST
    route_reason: str = ""
    tier: Tier = Tier.FAST  # tier to try next
    input_est: int = 0
    spec: ModelSpec | None = None  # model chosen for the pending call (None = nothing affordable)
    allowance: int = 0
    attempts: list[Attempt] = Field(default_factory=list)
    spent_tokens: int = 0
    spent_usd: float = 0.0
    exit_reason: str | None = None
    decision: Decision | None = None


class CostAwareRouter:
    def __init__(
        self,
        client: LLMClient,
        settings: Settings | None = None,
        store: DecisionStore | None = None,
        max_output_tokens: int = 512,
        strict_budget: bool = False,
        checkpointer: Any = None,
    ) -> None:
        self.client = client
        self.settings = settings or Settings()
        self.store = store
        self.max_output_tokens = max_output_tokens
        self.strict_budget = strict_budget
        self.by_tier: dict[Tier, ModelSpec] = {m.tier: m for m in self.settings.models()}
        self.checkpointer = checkpointer
        self.graph = self._build()

    # ---- graph ---------------------------------------------------------------------------------

    def _build(self):
        g = StateGraph(RouterState)
        g.add_node("classify", self._classify)
        g.add_node("plan_call", self._plan_call)
        g.add_node("call_model", self._call_model)
        g.add_node("evaluate", self._evaluate)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "classify")
        g.add_edge("classify", "plan_call")
        g.add_conditional_edges("plan_call", lambda s: "call_model" if s.spec else "finalize", ["call_model", "finalize"])
        g.add_edge("call_model", "evaluate")
        g.add_conditional_edges("evaluate", lambda s: "finalize" if s.exit_reason else "plan_call", ["plan_call", "finalize"])
        g.add_edge("finalize", END)
        return g.compile(checkpointer=self.checkpointer)

    def run(self, task: Task | str) -> Decision:
        if isinstance(task, str):
            task = Task(prompt=task)
        out = self.graph.invoke(
            {"task": task},
            config={"configurable": {"thread_id": task.id}, "recursion_limit": RECURSION_LIMIT},
        )
        return out["decision"]

    def history(self, task_id: str) -> list[RouterState]:
        """Checkpointed states for a finished task, oldest first (one per graph step)."""
        if self.checkpointer is None:
            raise RuntimeError("history() needs a checkpointer: pass checkpointer=MemorySaver(serde=CHECKPOINT_SERDE)")
        config = {"configurable": {"thread_id": task_id}}
        return [snap.values for snap in reversed(list(self.graph.get_state_history(config)))]  # type: ignore[misc]

    # ---- nodes ---------------------------------------------------------------------------------

    def _budget(self, s: RouterState) -> TokenBudget:
        b = TokenBudget(s.task.budget_tokens, s.task.budget_usd)
        b.spent_tokens, b.spent_usd = s.spent_tokens, s.spent_usd
        return b

    def _classify(self, s: RouterState) -> dict:
        cx = score_complexity(s.task.prompt)
        return {
            "complexity_score": cx.score,
            "complexity_tier": cx.tier,
            "tier": cx.tier,
            "route_reason": f"{cx.tier.name} tier: {cx.reason}",
            "input_est": estimate_tokens(SYSTEM_PROMPT + s.task.prompt),
        }

    def _plan_call(self, s: RouterState) -> dict:
        """Budget gate: the wanted tier, else the nearest cheaper tier (not below ones already tried) that fits."""
        budget = self._budget(s)
        floor = Tier(s.attempts[-1].tier + 1) if s.attempts else Tier.FAST
        for t in range(s.tier, floor - 1, -1):
            spec = self.by_tier[Tier(t)]
            allowance = budget.max_output_for(spec, s.input_est, self.max_output_tokens)
            if allowance >= MIN_USEFUL_OUTPUT:
                reason = s.route_reason
                if t != s.tier:
                    reason += f"; downgraded {s.tier.name}->{Tier(t).name} to fit budget"
                return {"spec": spec, "allowance": allowance, "tier": Tier(t), "route_reason": reason}
        if self.strict_budget:
            raise BudgetExceeded(f"task {s.task.id}: no model fits the remaining budget")
        return {
            "spec": None,
            "exit_reason": ExitReason.BUDGET if s.attempts else ExitReason.NO_BUDGET,
        }

    def _call_model(self, s: RouterState) -> dict:
        spec = s.spec
        assert spec is not None
        resp = self.client.complete(
            LLMRequest(model=spec.name, prompt=s.task.prompt, system=SYSTEM_PROMPT, max_tokens=s.allowance)
        )
        answer, conf = parse_response(resp.text)
        budget = self._budget(s)
        cost = budget.charge(spec, resp.usage)
        attempt = Attempt(
            model=spec.name, tier=spec.tier, usage=resp.usage, cost_usd=cost, confidence=conf, answer=answer
        )
        return {
            "attempts": [*s.attempts, attempt],
            "spent_tokens": budget.spent_tokens,
            "spent_usd": budget.spent_usd,
        }

    def _evaluate(self, s: RouterState) -> dict:
        """Early exit on confidence, accept at the top tier, otherwise escalate one tier."""
        last = s.attempts[-1]
        if last.confidence is not None and last.confidence >= self.settings.confidence_threshold:
            return {"exit_reason": ExitReason.CONFIDENT}
        if s.tier == Tier.PREMIUM:
            return {"exit_reason": ExitReason.TOP_TIER}
        return {"tier": Tier(s.tier + 1)}

    def _finalize(self, s: RouterState) -> dict:
        baseline = self.by_tier[self.settings.baseline_tier]
        decision = Decision(
            task_id=s.task.id,
            prompt_preview=s.task.prompt[:80],
            complexity_score=s.complexity_score,
            complexity_tier=s.complexity_tier,
            route_reason=s.route_reason,
            attempts=[a.model_copy() for a in s.attempts],
            exit_reason=s.exit_reason or ExitReason.NO_BUDGET,
            total_tokens=s.spent_tokens,
            cost_usd=s.spent_usd,
            baseline_model=baseline.name,
            budget_tokens=s.task.budget_tokens,
            budget_usd=s.task.budget_usd,
        )
        if decision.attempts:  # accept the last answer (confident, top-tier, or best already paid for)
            last = decision.attempts[-1]
            last.accepted = True
            decision.final_model, decision.answer, decision.confidence = last.model, last.answer, last.confidence
        # Counterfactual: same prompt and answer length on the baseline model, one call, no cascade.
        out = decision.attempts[-1].usage.output_tokens if decision.attempts else self.max_output_tokens // 2
        decision.baseline_cost_usd = baseline.cost(s.input_est, out)
        if self.store:
            self.store.append(decision)
        return {"decision": decision}
