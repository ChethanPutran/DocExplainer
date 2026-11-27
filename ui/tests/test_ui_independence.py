"""Architecture guard: the UI talks to the backend only through the SDK.

It must not import the backend package (``doc_explainer``) and must not start or configure the
backend. Allowed: its own package, ``doc_explainer_sdk``, third-party libraries and the stdlib.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "doc_explainer_ui"


def imported_top_levels(py: Path):
    for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


def test_ui_never_imports_the_backend():
    bad = [f"{py.relative_to(SRC)}:{line} imports {name}"
           for py in SRC.rglob("*.py") for name, line in imported_top_levels(py)
           if name == "doc_explainer"]
    assert not bad, "UI must use doc_explainer_sdk, not the backend:\n  " + "\n  ".join(bad)


def test_ui_does_not_launch_or_configure_the_backend():
    forbidden = ("doc-explainer-server", "doc_explainer.api", "uvicorn", "LLMConfig", "BackendConfig",
                 "OrchestratorConfig")
    hits = [f"{py.relative_to(SRC)}: {word}" for py in SRC.rglob("*.py")
            for word in forbidden if word in py.read_text(encoding="utf-8")]
    assert not hits, "\n".join(hits)
