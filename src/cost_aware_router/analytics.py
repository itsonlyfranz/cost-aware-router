"""Cost-per-decision analytics over a local JSONL decision log."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from cost_aware_router.models import Decision


class DecisionStore:
    """Append-only JSONL file; one Decision per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, decision: Decision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(decision.model_dump_json() + "\n")

    def load(self) -> list[Decision]:
        if not self.path.is_file():
            return []
        return [Decision.model_validate_json(line) for line in self.path.read_text().splitlines() if line.strip()]


class Group(BaseModel):
    decisions: int = 0
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.cost_usd / self.decisions if self.decisions else 0.0


class Summary(BaseModel):
    decisions: int
    baseline_model: str
    cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    savings_pct: float
    avg_cost_usd: float
    p50_cost_usd: float
    p95_cost_usd: float
    total_tokens: int
    by_model: dict[str, Group]
    by_complexity: dict[str, Group]
    by_exit_reason: dict[str, int]
    escalation_rate: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(pct * (len(s) - 1))))]


def summarize(decisions: Iterable[Decision]) -> Summary:
    ds = list(decisions)
    by_model: dict[str, Group] = defaultdict(Group)
    by_cx: dict[str, Group] = defaultdict(Group)
    exits: dict[str, int] = defaultdict(int)
    for d in ds:
        for key, groups in ((d.final_model or "none", by_model), (d.complexity_tier.name, by_cx)):
            g = groups[key]
            g.decisions += 1
            g.cost_usd += d.cost_usd
            g.baseline_cost_usd += d.baseline_cost_usd
        exits[d.exit_reason] += 1
    cost = sum(d.cost_usd for d in ds)
    base = sum(d.baseline_cost_usd for d in ds)
    costs = [d.cost_usd for d in ds]
    return Summary(
        decisions=len(ds),
        baseline_model=ds[0].baseline_model if ds else "",
        cost_usd=cost,
        baseline_cost_usd=base,
        savings_usd=base - cost,
        savings_pct=(base - cost) / base * 100 if base else 0.0,
        avg_cost_usd=statistics.fmean(costs) if costs else 0.0,
        p50_cost_usd=_percentile(costs, 0.5),
        p95_cost_usd=_percentile(costs, 0.95),
        total_tokens=sum(d.total_tokens for d in ds),
        by_model=dict(by_model),
        by_complexity=dict(by_cx),
        by_exit_reason=dict(exits),
        escalation_rate=sum(len(d.attempts) > 1 for d in ds) / len(ds) if ds else 0.0,
    )


def render(summary: Summary) -> str:
    s = summary
    lines = [
        f"decisions            {s.decisions}",
        f"total tokens         {s.total_tokens:,}",
        f"routed cost          ${s.cost_usd:.4f}",
        f"baseline cost        ${s.baseline_cost_usd:.4f}  (every task on {s.baseline_model})",
        f"savings              ${s.savings_usd:.4f}  ({s.savings_pct:.1f}%)",
        f"cost / decision      avg ${s.avg_cost_usd:.5f}  p50 ${s.p50_cost_usd:.5f}  p95 ${s.p95_cost_usd:.5f}",
        f"escalation rate      {s.escalation_rate:.0%}",
        "",
        "by final model            n   cost      avg/decision",
    ]
    for name, g in sorted(s.by_model.items()):
        lines.append(f"  {name:<22}{g.decisions:>3}   ${g.cost_usd:<8.4f} ${g.avg_cost_usd:.5f}")
    lines += ["", "by complexity tier        n   cost      baseline  "]
    for name, g in sorted(s.by_complexity.items()):
        lines.append(f"  {name:<22}{g.decisions:>3}   ${g.cost_usd:<8.4f} ${g.baseline_cost_usd:.4f}")
    lines += ["", "exit reasons"]
    for reason, n in sorted(s.by_exit_reason.items()):
        lines.append(f"  {reason:<28}{n}")
    return "\n".join(lines)
