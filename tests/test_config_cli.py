from cost_aware_router.cli import main
from cost_aware_router.config import Settings, load_dotenv
from cost_aware_router.models import Tier


def test_defaults():
    s = Settings.from_env({})
    assert s.default_model == "claude-sonnet-5-5" and s.baseline_tier == Tier.PREMIUM


def test_env_overrides():
    s = Settings.from_env({
        "CAR_DEFAULT_MODEL": "x-model", "CAR_PRICE_FAST": "0.5,2.5", "CAR_CONFIDENCE_THRESHOLD": "0.6",
        "CAR_BASELINE_TIER": "balanced", "ANTHROPIC_API_KEY": "k",
    })
    assert s.default_model == "x-model" and s.prices[Tier.FAST] == (0.5, 2.5)
    assert s.confidence_threshold == 0.6 and s.baseline_tier == Tier.BALANCED and s.api_key == "k"
    assert {m.tier: m.name for m in s.models()}[Tier.BALANCED] == "x-model"


def test_dotenv_does_not_override_existing_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nCAR_A=from_file\nCAR_B='quoted'\n")
    monkeypatch.setenv("CAR_A", "from_env")
    monkeypatch.delenv("CAR_B", raising=False)
    load_dotenv(f)
    import os
    assert os.environ["CAR_A"] == "from_env" and os.environ["CAR_B"] == "quoted"
    monkeypatch.delenv("CAR_B")


def test_demo_runs_offline_and_reports_savings(capsys, tmp_path):
    assert main(["demo", "--store", str(tmp_path / "d.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "savings" in out and "budget_exhausted" in out and "no_affordable_model" in out
    assert main(["report", "--store", str(tmp_path / "d.jsonl")]) == 0


def test_demo_hits_headline_savings_range(tmp_path):
    from cost_aware_router.analytics import DecisionStore, summarize
    from cost_aware_router.agent import CostAwareRouter
    from cost_aware_router.simulation import demo_workload, simulated_client
    s = Settings()
    wl = demo_workload()
    router = CostAwareRouter(simulated_client({t.prompt: d for t, d in wl}, s), s, DecisionStore(tmp_path / "d.jsonl"))
    for t, _ in wl:
        router.run(t)
    assert 40 <= summarize(router.store.load()).savings_pct <= 70


def test_run_without_api_key_fails_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir("/")
    assert main(["run", "hi"]) == 2
