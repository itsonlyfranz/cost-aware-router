from conftest import EASY, HARD

from cost_aware_router.complexity import score_complexity
from cost_aware_router.models import Tier


def test_simple_lookup_routes_to_fast():
    assert score_complexity(EASY).tier == Tier.FAST


def test_design_and_debug_prompt_routes_to_premium():
    c = score_complexity(HARD)
    assert c.tier == Tier.PREMIUM
    assert "reasoning keywords" in c.reason


def test_medium_task_routes_to_balanced():
    c = score_complexity("Explain how a database index works and write a function that illustrates it for a junior.")
    assert c.tier == Tier.BALANCED


def test_score_is_monotonic_in_difficulty_and_bounded():
    assert score_complexity(EASY).score < score_complexity(HARD).score <= 1.0
    assert score_complexity("").score >= 0.0
