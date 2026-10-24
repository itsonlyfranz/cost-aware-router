import pytest

from cost_aware_router.budget import MIN_USEFUL_OUTPUT, TokenBudget
from cost_aware_router.models import ModelSpec, Tier, Usage

SPEC = ModelSpec(name="m", tier=Tier.BALANCED, input_per_mtok=3.0, output_per_mtok=15.0)


def test_unlimited_budget_allows_requested_output():
    assert TokenBudget().max_output_for(SPEC, 100, 512) == 512


def test_token_budget_caps_output_allowance():
    b = TokenBudget(max_tokens=300)
    assert b.max_output_for(SPEC, 100, 512) == 200


def test_returns_zero_when_remaining_below_useful_minimum():
    b = TokenBudget(max_tokens=100 + MIN_USEFUL_OUTPUT - 1)
    assert b.max_output_for(SPEC, 100, 512) == 0


def test_usd_budget_caps_output_allowance():
    b = TokenBudget(max_usd=0.001)
    out = b.max_output_for(SPEC, 100, 10_000)
    assert 0 < out < 10_000
    assert SPEC.cost(100, out) <= 0.001 + 1e-9


def test_charge_accumulates_and_reduces_remaining():
    b = TokenBudget(max_tokens=1000, max_usd=1.0)
    cost = b.charge(SPEC, Usage(input_tokens=100, output_tokens=50))
    assert cost == pytest.approx(SPEC.cost(100, 50))
    assert b.remaining_tokens == 850
    assert b.spent_usd == pytest.approx(cost)


def test_overshoot_clamps_remaining_at_zero():
    b = TokenBudget(max_tokens=10)
    b.charge(SPEC, Usage(input_tokens=50, output_tokens=50))
    assert b.remaining_tokens == 0 and b.exhausted
