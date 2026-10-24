import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)

EDGE = re.compile(r"^\s*(\w+)\s+(-->|-\.->)\s+(\w+);?\s*$")


def diagram_edges(text: str) -> set[tuple[str, str, bool]]:
    """(source, target, conditional) edges of the Mermaid block between the flow-diagram markers."""
    _, _, rest = text.partition(render_graph.START)
    block, _, _ = rest.partition(render_graph.END)
    edges = set()
    for line in block.splitlines():
        match = EDGE.match(line)
        if match:
            source, arrow, target = match.groups()
            edges.add((source, target, arrow == "-.->"))
    return edges


def graph_edges() -> set[tuple[str, str, bool]]:
    return {(e.source, e.target, bool(e.conditional)) for e in render_graph.compiled_graph().edges}


def test_readme_flow_diagram_matches_compiled_graph():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    edges = diagram_edges(readme)
    assert edges == graph_edges(), "README flow diagram drifted; run: python scripts/render_graph.py"
    assert {n for e in edges for n in e[:2]} == set(render_graph.compiled_graph().nodes)


def test_flow_diagram_contains_every_node():
    diagram = render_graph.render_flow()
    for node in ("classify", "plan_call", "call_model", "evaluate", "finalize"):
        assert node in diagram


def test_splice_replaces_stale_block_and_keeps_surroundings():
    stale = f"intro\n{render_graph.START}\nold\n{render_graph.END}\noutro\n"
    spliced = render_graph.splice(stale)
    assert spliced.startswith(f"intro\n{render_graph.START}\n")
    assert spliced.endswith(f"{render_graph.END}\noutro\n")
    assert diagram_edges(spliced) == graph_edges()


def test_splice_requires_markers():
    with pytest.raises(ValueError):
        render_graph.splice("no markers here")
