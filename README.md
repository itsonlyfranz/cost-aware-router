# cost-aware-router

A cost-aware agent router: **per-task token budgets**, **complexity-based model routing**, **early exit on confidence**, and **cost-per-decision analytics**, built on the Anthropic Claude API.

> Cost-aware agent router: per-task token budgets, complexity-based model routing, confidence early exit, and cost-per-decision analytics on the Anthropic Claude API.

**Technologies:** Python 3.11+, Anthropic SDK, Pydantic v2, pytest · **Category:** AI agents / cost optimization

**What it demonstrates:** most agent traffic doesn't need your most expensive model. Routing each task to the cheapest model that can handle it, stopping as soon as the answer is confident, and capping spend per task cuts inference cost substantially. See [the caveat](#what-the-numbers-do-and-dont-mean) before quoting any number below.

## Highlights

- **Complexity routing at zero cost:** a heuristic scorer (length, reasoning keywords, code, step count) picks a starting tier from fast, balanced or premium with no LLM call.
- **Confidence-gated cascade:** a cheap model answers first with a self-reported confidence; the router exits early at the threshold or escalates one tier at a time.
- **Hard per-task budgets:** token and USD caps size `max_tokens` before each call, downgrade tiers under pressure, and never make a call whose allowance doesn't fit.
- **Cost-per-decision analytics:** every decision (attempts, tokens, USD, exit reason, counterfactual baseline cost) is logged to local JSONL and summarized with p50/p95, per-model and per-tier breakdowns.
- **Measured in the bundled demo (simulated models, 36 tasks):** routed cost $0.0646 vs $0.1148 always-premium, a 43.7% reduction, with 14% of tasks escalating; 44 offline tests pass.

## Architecture

```
            Task(prompt, budget_tokens?, budget_usd?)
                         │
                         ▼
              ┌─────────────────────┐
              │ complexity scorer   │  heuristic, zero-cost (length, reasoning keywords, code, steps)
              └─────────┬───────────┘
                        │ starting tier: FAST / BALANCED / PREMIUM
                        ▼
   ┌──────────────► budget gate ── can this tier's call fit the remaining token/USD budget?
   │                    │ no → downgrade to the nearest cheaper tier that fits (never below
   │                    │      tiers already tried); none fits → stop (keep best paid answer)
   │                    ▼ yes
   │              LLM call (max_tokens capped to budget allowance)
   │                    │
   │                    ▼
   │         parse {"answer", "confidence"}
   │           │                    │
   │   conf ≥ threshold        conf < threshold (or unparseable)
   │           ▼                    ▼
   │      EARLY EXIT          at PREMIUM? ── yes → accept answer
   │                                │ no
   └────────────────────────────────┘ escalate one tier
                        │
                        ▼
        Decision (attempts, tokens, USD, baseline USD, exit reason) → JSONL log → analytics
```

| Component | Responsibility |
|---|---|
| `models.py` | Pydantic v2 models: `Task`, `Attempt`, `Decision`, `ModelSpec`, `Tier` |
| `complexity.py` | Heuristic prompt-complexity score → starting tier |
| `budget.py` | `TokenBudget`: token + USD limits, output allowance, charging actual usage |
| `confidence.py` | Structured-output prompt and robust parser for the confidence signal |
| `agent.py` | `CostAwareRouter`: the routing / escalation / early-exit loop |
| `analytics.py` | Local JSONL `DecisionStore`, `summarize()`, text `render()` |
| `client.py` | `LLMClient` protocol, `AnthropicClient`, offline `FakeClient` |
| `simulation.py` | Deterministic simulated models and demo workload |
| `cli.py` | `car demo`, `car run`, `car report` |

## Quickstart

Requires Python 3.11+.

```bash
uv sync --extra dev          # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
uv run pytest                # fully offline, no API key needed
uv run car demo              # offline demo with simulated models
```

For live calls, copy `.env.example` to `.env` (gitignored) and set `ANTHROPIC_API_KEY`:

```bash
uv run car run "Explain how a B-tree index works" --budget-usd 0.01
uv run car report            # analytics over .car/decisions.jsonl
```

### Configuration (env vars or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Needed only for `car run` |
| `CAR_DEFAULT_MODEL` | `claude-sonnet-5-5` | Balanced tier |
| `CAR_MODEL_FAST` / `CAR_MODEL_PREMIUM` | `claude-haiku-5-5` / `claude-opus-5-5` | Cheap and top tiers |
| `CAR_PRICE_FAST` / `_BALANCED` / `_PREMIUM` | illustrative | `"input,output"` USD per million tokens |
| `CAR_CONFIDENCE_THRESHOLD` | `0.8` | Early-exit threshold |
| `CAR_BASELINE_TIER` | `PREMIUM` | What "no router" would have used, for savings accounting |
| `CAR_DATA_DIR` | `.car` | Where `decisions.jsonl` lives |

Built-in prices are **illustrative placeholders** — set the `CAR_PRICE_*` variables to your actual rates.

## Demo

`car demo` runs 36 tasks (a skewed mix: ~2/3 trivial lookups, some medium, a few hard, one that *looks* easy but isn't) through the router against simulated models whose confidence is calibrated to a hidden per-task difficulty. It prints each routing path, then the analytics report, then shows budget enforcement on a hard task:

```
  FAST     haiku(0.93)                               $0.00045  confident_early_exit
  FAST     haiku(0.40) -> sonnet(0.93)               $0.00369  confident_early_exit
  PREMIUM  opus(0.93)                                $0.00600  confident_early_exit
  FAST     haiku(0.40) -> sonnet(0.40) -> opus(0.93) $0.00992  confident_early_exit
  ...
routed cost          $0.0646
baseline cost        $0.1148  (every task on claude-opus-5-5)
savings              $0.0502  (43.7%)

0.001 USD cap: budget_exhausted, model=claude-sonnet-5-5 ... downgraded PREMIUM->BALANCED to fit budget
120 token cap: no_affordable_model, model=None, 0 tokens
```

Note the router pays for misrouting: the deceptive prompt costs more than going straight to premium (three calls). That cost is included in the savings figure.

### What the numbers do and don't mean

The demo is a **simulation**: model confidence and answer lengths are scripted, so the ~44% is a property of the workload mix and price ratios, not a measurement of real model quality. It shows the mechanism and the accounting. Real savings depend on your traffic mix, how well the models' self-reported confidence is calibrated, and your actual prices; measure with `car report` on real traffic. The 40–60% figure is plausible for easy-heavy workloads, not guaranteed.

## Design decisions and trade-offs

- **Heuristic complexity scoring, not an LLM classifier.** Routing must be nearly free; an LLM router adds cost and latency to every task. The price is misroutes, which the confidence cascade corrects.
- **Self-reported confidence as the early-exit signal.** Cheap and model-agnostic, but LLM self-confidence is imperfectly calibrated — a confidently wrong cheap model is the main risk. Mitigate by raising `CAR_CONFIDENCE_THRESHOLD`, and validate calibration on your own data. Unparseable output counts as "not confident".
- **Escalate one tier at a time, from the starting tier.** Minimises spend when the heuristic was right; the cost of a wrong guess is the wasted lower-tier calls (visible in `escalation rate`).
- **Budgets are pre-flight estimates plus post-hoc charging.** Input tokens are estimated (~4 chars/token) to size `max_tokens` before each call; real usage is charged afterwards. A call is only made if a useful output allowance fits, so overshoot is bounded to estimation error. If the next tier doesn't fit, the router stops and returns the best answer already paid for rather than silently exceeding the budget (`strict_budget=True` raises `BudgetExceeded` instead).
- **Budget pressure downgrades before it fails.** A premium-complexity task under a tight USD cap runs on a cheaper tier and is flagged in `route_reason`.
- **Savings are measured against a counterfactual.** Baseline = one call on the baseline tier with the same prompt and the final answer's output length. It's an estimate, and it can be negative per decision (routing can cost more than the baseline) — aggregated numbers include those.
- **Local-first storage.** Decisions are an append-only JSONL file; no database or hosted service.
- **LLM behind a one-method protocol** (`LLMClient.complete`), so the whole suite runs offline against `FakeClient`.
- **Out of scope:** streaming, tool-use loops, caching, async/concurrency, a web UI, deployment.

## How the tests prove each capability

All 44 tests are offline (`FakeClient`; the Anthropic SDK is stubbed).

| Capability | Tests |
|---|---|
| **Complexity routing** | `test_complexity.py` (easy→FAST, design/debug→PREMIUM, medium→BALANCED, monotonic score); `test_router.py::test_routes_easy_task_to_cheapest_model`, `test_routes_hard_task_straight_to_premium` assert exactly which model the client was called with |
| **Early exit on confidence** | `test_early_exit_when_confident_skips_higher_tiers`, `test_low_confidence_escalates_one_tier_at_a_time`, `test_top_tier_answer_is_accepted_even_if_unconfident`, `test_threshold_is_configurable`, `test_unparseable_reply_counts_as_not_confident`; parser behaviour in `test_confidence.py` |
| **Token budgeting** | `test_budget.py` (allowance math, USD and token caps, overshoot clamp); `test_router.py` budget tests: `max_tokens` capped on the wire, no call when nothing fits, strict mode raises, USD cap downgrades premium to a cheaper model, escalation stops (and never re-calls a lower tier) when the next tier is unaffordable |
| **Cost-per-decision analytics** | `test_analytics.py`: JSONL round-trip, totals equal sum of decisions, p50/p95, per-model/tier breakdown, escalation rate, ≥40% savings vs always-premium on easy-heavy traffic, and negative savings reported honestly when the baseline is cheaper |
| **Config, CLI, client** | `test_config_cli.py` (env overrides, `.env` precedence, demo end-to-end, demo savings in the 40–70% band, clean failure without an API key); `test_anthropic_client.py` (request/usage mapping) |
