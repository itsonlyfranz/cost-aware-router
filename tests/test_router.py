import pytest
from conftest import EASY, HARD, by_model, reply

from cost_aware_router.agent import CostAwareRouter
from cost_aware_router.budget import BudgetExceeded
from cost_aware_router.client import FakeClient
from cost_aware_router.models import ExitReason, Task, Tier


def test_routes_easy_task_to_cheapest_model(settings):
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    d = CostAwareRouter(client, settings).run(EASY)
    assert [c.model for c in client.calls] == [settings.fast_model]
    assert d.final_model == settings.fast_model
    assert d.complexity_tier == Tier.FAST


def test_routes_hard_task_straight_to_premium(settings):
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    d = CostAwareRouter(client, settings).run(HARD)
    assert [c.model for c in client.calls] == [settings.premium_model]
    assert d.complexity_tier == Tier.PREMIUM


def test_early_exit_when_confident_skips_higher_tiers(settings):
    client = by_model(settings, {"fast": 0.9, "balanced": 0.99, "premium": 0.99})
    d = CostAwareRouter(client, settings).run(EASY)
    assert d.exit_reason == ExitReason.CONFIDENT
    assert len(d.attempts) == 1 and d.attempts[0].accepted


def test_low_confidence_escalates_one_tier_at_a_time(settings):
    client = by_model(settings, {"fast": 0.3, "balanced": 0.5, "premium": 0.9})
    d = CostAwareRouter(client, settings).run(EASY)
    assert [a.tier for a in d.attempts] == [Tier.FAST, Tier.BALANCED, Tier.PREMIUM]
    assert d.final_model == settings.premium_model
    assert [a.accepted for a in d.attempts] == [False, False, True]


def test_top_tier_answer_is_accepted_even_if_unconfident(settings):
    client = by_model(settings, {"fast": 0.1, "balanced": 0.1, "premium": 0.2})
    d = CostAwareRouter(client, settings).run(EASY)
    assert d.exit_reason == ExitReason.TOP_TIER
    assert d.final_model == settings.premium_model


def test_threshold_is_configurable(settings):
    settings.confidence_threshold = 0.5
    client = by_model(settings, {"fast": 0.6, "balanced": 0.0, "premium": 0.0})
    assert len(CostAwareRouter(client, settings).run(EASY).attempts) == 1


def test_unparseable_reply_counts_as_not_confident(settings):
    client = FakeClient(lambda r: "plain text, no json" if r.model == settings.fast_model else reply(0.95))
    d = CostAwareRouter(client, settings).run(EASY)
    assert len(d.attempts) == 2 and d.attempts[0].confidence is None


def test_cost_is_sum_of_all_attempts(settings):
    client = by_model(settings, {"fast": 0.1, "balanced": 0.95, "premium": 0.95})
    d = CostAwareRouter(client, settings).run(EASY)
    assert d.cost_usd == pytest.approx(sum(a.cost_usd for a in d.attempts))
    assert d.total_tokens == sum(a.usage.total for a in d.attempts)


# --- budgets -------------------------------------------------------------------------------------


def test_token_budget_caps_max_tokens_sent_to_model(settings):
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    CostAwareRouter(client, settings, max_output_tokens=2000).run(Task(prompt=EASY, budget_tokens=200))
    assert client.calls[0].max_tokens < 200


def test_budget_too_small_makes_no_call(settings):
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    d = CostAwareRouter(client, settings).run(Task(prompt=EASY, budget_tokens=10))
    assert client.calls == [] and d.exit_reason == ExitReason.NO_BUDGET and d.final_model is None


def test_strict_budget_raises(settings):
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    with pytest.raises(BudgetExceeded):
        CostAwareRouter(client, settings, strict_budget=True).run(Task(prompt=EASY, budget_tokens=10))


def test_usd_budget_downgrades_premium_task_to_cheaper_model(settings):
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    d = CostAwareRouter(client, settings).run(Task(prompt=HARD, budget_usd=0.0009))
    assert d.complexity_tier == Tier.PREMIUM
    assert client.calls[0].model != settings.premium_model
    assert "downgraded" in d.route_reason
    assert d.cost_usd <= 0.0009 * 1.5  # estimates are approximate; overshoot stays small


def test_escalation_stops_when_next_tier_unaffordable_and_keeps_best_answer(settings):
    client = by_model(settings, {"fast": 0.1, "balanced": 0.1, "premium": 0.99})
    prompt = EASY
    # enough for one fast call, not enough for another
    budget = 140
    d = CostAwareRouter(client, settings, max_output_tokens=64).run(Task(prompt=prompt, budget_tokens=budget))
    assert d.exit_reason == ExitReason.BUDGET
    assert d.final_model == settings.fast_model  # never re-called a lower tier, never exceeded the cap by a new call
    assert len(client.calls) == 1


def test_total_spend_never_triggers_call_beyond_budget(settings):
    client = by_model(settings, {"fast": 0.1, "balanced": 0.1, "premium": 0.1})
    d = CostAwareRouter(client, settings, max_output_tokens=64).run(Task(prompt=EASY, budget_tokens=400))
    assert d.total_tokens <= 400 + 64  # a call is only made if its allowance fits; actual usage can slightly differ


# --- LangGraph structure -------------------------------------------------------------------------


def test_graph_has_explicit_nodes_and_conditional_edges(settings):
    router = CostAwareRouter(by_model(settings, {"fast": 0.9, "balanced": 0.9, "premium": 0.9}), settings)
    g = router.graph.get_graph()
    assert {"classify", "plan_call", "call_model", "evaluate", "finalize"} <= set(g.nodes)
    conditional = {(e.source, e.target) for e in g.edges if e.conditional}
    assert {("plan_call", "call_model"), ("plan_call", "finalize"),
            ("evaluate", "plan_call"), ("evaluate", "finalize")} <= conditional


def test_checkpointer_records_step_by_step_state(settings):
    client = by_model(settings, {"fast": 0.1, "balanced": 0.95, "premium": 0.95})
    router = CostAwareRouter(client, settings)
    d = router.run(Task(id="t1", prompt=EASY))
    states = router.history("t1")
    assert len(states) > 5
    assert states[-1]["decision"].task_id == "t1"
    spent = [s.get("spent_usd", 0.0) for s in states]
    assert spent == sorted(spent) and spent[-1] == pytest.approx(d.cost_usd)


def test_recursion_limit_bounds_the_worst_case_path(settings):
    from cost_aware_router.agent import RECURSION_LIMIT
    client = by_model(settings, {"fast": 0.0, "balanced": 0.0, "premium": 0.0})
    d = CostAwareRouter(client, settings).run(EASY)  # full fast->balanced->premium cascade
    assert len(d.attempts) == 3 and RECURSION_LIMIT >= 2 + 3 * 3
