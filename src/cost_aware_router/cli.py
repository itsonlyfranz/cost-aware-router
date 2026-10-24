"""`car` command line: demo, run, report."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from cost_aware_router.agent import CostAwareRouter
from cost_aware_router.analytics import DecisionStore, render, summarize
from cost_aware_router.client import AnthropicClient
from cost_aware_router.config import Settings, load_dotenv
from cost_aware_router.models import Task
from cost_aware_router.simulation import demo_workload, simulated_client


def _cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    workload = demo_workload()
    difficulty = {t.prompt: d for t, d in workload}
    store_path = Path(args.store) if args.store else Path(tempfile.mkdtemp()) / "decisions.jsonl"
    store = DecisionStore(store_path)
    router = CostAwareRouter(simulated_client(difficulty, settings), settings, store)

    print(f"Running {len(workload)} tasks through the router (simulated models, offline)\n")
    for task, _ in workload:
        d = router.run(task)
        path = " -> ".join(f"{a.model.split('-')[1]}({a.confidence:.2f})" if a.confidence is not None else a.model
                           for a in d.attempts)
        print(f"  {d.complexity_tier.name:<9}{path:<42}${d.cost_usd:.5f}  {d.exit_reason}")

    print(render(summarize(store.load())))

    hard = workload[-2][0]
    print("\n\nBudget enforcement on the same hard task")
    for label, caps in (("0.001 USD cap", {"budget_usd": 0.001}), ("120 token cap", {"budget_tokens": 120})):
        d = router.run(hard.model_copy(update={"id": label.replace(" ", "-"), **caps}))
        print(f"  {label}: {d.exit_reason}, model={d.final_model}, {d.total_tokens} tokens, ${d.cost_usd:.5f}"
              f"  [{d.route_reason}]")
    print(f"\nDecision log: {store_path}")
    return 0


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.api_key:
        print("ANTHROPIC_API_KEY is not set (env or .env). Use `car demo` for the offline demo.", file=sys.stderr)
        return 2
    store = DecisionStore(settings.decisions_path)
    router = CostAwareRouter(AnthropicClient(settings.api_key), settings, store)
    d = router.run(Task(prompt=args.prompt, budget_tokens=args.budget_tokens, budget_usd=args.budget_usd))
    print(d.answer)
    print(f"\n[{d.route_reason}] model={d.final_model} exit={d.exit_reason} tokens={d.total_tokens} cost=${d.cost_usd:.5f}",
          file=sys.stderr)
    return 0


def _cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    path = Path(args.store) if args.store else settings.decisions_path
    decisions = DecisionStore(path).load()
    if not decisions:
        print(f"No decisions found at {path}", file=sys.stderr)
        return 1
    print(render(summarize(decisions)))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="car", description="Cost-aware agent router")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="offline demo with simulated models")
    p.add_argument("--store", help="JSONL decision log path (default: temp file)")
    p.set_defaults(fn=_cmd_demo)

    p = sub.add_parser("run", help="route one prompt to the real Anthropic API")
    p.add_argument("prompt")
    p.add_argument("--budget-tokens", type=int)
    p.add_argument("--budget-usd", type=float)
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("report", help="cost-per-decision analytics from the decision log")
    p.add_argument("--store")
    p.set_defaults(fn=_cmd_report)

    args = parser.parse_args(argv)
    return args.fn(args, Settings.from_env())


if __name__ == "__main__":
    raise SystemExit(main())
