import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def test_readme_flow_diagram_matches_compiled_graph():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert render_graph.flow_block() in readme, "README flow diagram drifted; run: python scripts/render_graph.py"


def test_flow_diagram_contains_every_node():
    diagram = render_graph.render_flow()
    for node in ("classify", "plan_call", "call_model", "evaluate", "finalize"):
        assert node in diagram


def test_splice_is_idempotent():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert render_graph.splice(readme) == readme
