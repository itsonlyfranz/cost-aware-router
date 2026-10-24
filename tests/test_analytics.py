import pytest
from conftest import EASY, HARD, by_model

from cost_aware_router.agent import CostAwareRouter
from cost_aware_router.analytics import DecisionStore, render, summarize
from cost_aware_router.models import Task


def _run_mix(settings, store):
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    router = CostAwareRouter(client, settings, store)
    for _ in range(4):
        router.run(EASY)
    router.run(HARD)
    return router


def test_decisions_persist_to_jsonl_and_round_trip(settings, tmp_path):
    store = DecisionStore(tmp_path / "sub" / "d.jsonl")
    _run_mix(settings, store)
    loaded = store.load()
    assert len(loaded) == 5
    assert loaded[0].attempts[0].model == settings.fast_model


def test_empty_store_loads_empty(tmp_path):
    assert DecisionStore(tmp_path / "none.jsonl").load() == []


def test_summary_totals_match_decisions(settings, tmp_path):
    store = DecisionStore(tmp_path / "d.jsonl")
    _run_mix(settings, store)
    ds = store.load()
    s = summarize(ds)
    assert s.decisions == 5
    assert s.cost_usd == pytest.approx(sum(d.cost_usd for d in ds))
    assert sum(g.decisions for g in s.by_model.values()) == 5
    assert s.p95_cost_usd >= s.p50_cost_usd > 0


def test_routing_beats_always_premium_baseline_on_easy_heavy_traffic(settings, tmp_path):
    store = DecisionStore(tmp_path / "d.jsonl")
    _run_mix(settings, store)
    s = summarize(store.load())
    assert s.baseline_model == settings.premium_model
    assert s.savings_usd > 0 and 40 <= s.savings_pct <= 100


def test_savings_negative_when_hard_task_baseline_is_cheaper(settings):
    settings.baseline_tier = settings.baseline_tier.FAST
    client = by_model(settings, {"fast": 0.95, "balanced": 0.95, "premium": 0.95})
    d = CostAwareRouter(client, settings).run(HARD)
    assert d.savings_usd < 0  # honest accounting: routing to premium costs more than a fast-only baseline


def test_render_contains_headline_numbers(settings, tmp_path):
    store = DecisionStore(tmp_path / "d.jsonl")
    _run_mix(settings, store)
    out = render(summarize(store.load()))
    assert "savings" in out and "cost / decision" in out and settings.fast_model in out


def test_summary_of_nothing_is_zeroed():
    s = summarize([])
    assert s.decisions == 0 and s.savings_pct == 0.0


def test_escalation_rate(settings):
    client = by_model(settings, {"fast": 0.1, "balanced": 0.95, "premium": 0.95})
    router = CostAwareRouter(client, settings)
    ds = [router.run(Task(prompt=EASY)) for _ in range(2)]
    assert summarize(ds).escalation_rate == 1.0
