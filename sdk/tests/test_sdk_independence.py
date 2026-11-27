"""Architecture guard: the SDK must stand alone (no backend, no UI, no Qt)."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "doc_explainer_sdk"
ALLOWED_THIRD_PARTY = {"httpx", "pydantic"}
FORBIDDEN = {"doc_explainer", "doc_explainer_ui", "PySide6", "fastapi", "uvicorn"}


def test_sdk_imports_nothing_from_backend_or_ui():
    bad = []
    for py in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                     else [])
            bad += [f"{py.name}:{node.lineno} {n}" for n in names if n.split(".")[0] in FORBIDDEN]
    assert not bad, bad


def test_sdk_runtime_dependencies_are_only_httpx_and_pydantic():
    import tomllib
    meta = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text())
    deps = {d.split(">")[0].split("=")[0].strip() for d in meta["project"]["dependencies"]}
    assert deps == ALLOWED_THIRD_PARTY
