"""Architecture guard: the packages under ``doc_explainer.core`` must not depend on each other.

Each ``core/<pkg>`` may import only itself and ``core.common`` (the shared kernel: shared data
models and interfaces). It must never import another core package, ``store`` or ``orchestrator``.
Anything it needs from a sibling is declared as a Protocol in the consuming package and injected
by the orchestrator, the only place where the packages are combined.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
CORE = SRC / "doc_explainer" / "core"
FORBIDDEN_ROOTS = {"store", "orchestrator"}


def _packages():
    return sorted(p.name for p in CORE.iterdir() if p.is_dir() and p.name != "__pycache__")


def _targets(py: Path, node):
    """Absolute dotted module names an import statement refers to."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    mod = node.module or ""
    if node.level:
        here = list(py.relative_to(SRC).with_suffix("").parts)
        if here[-1] == "__init__":
            here.pop()
        else:
            here = here[:-1]
        base = here[: len(here) - (node.level - 1)]
        mod = ".".join(base + ([node.module] if node.module else []))
    # `from pkg import sub` may import a submodule: include those candidates
    return [mod] + [f"{mod}.{a.name}" for a in node.names]


def violations():
    out = []
    for py in CORE.rglob("*.py"):
        rel = py.relative_to(CORE).parts
        if len(rel) < 2:
            continue
        me = rel[0]
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            seen = set()
            for t in _targets(py, node):
                parts = t.split(".")
                if parts[0] == "src":
                    parts[0] = "doc_explainer"
                if parts[:1] != ["doc_explainer"] or len(parts) < 2:
                    continue
                if parts[1] == "core" and len(parts) > 2 and parts[2] not in (me, "common"):
                    seen.add(f"core.{parts[2]}")
                elif parts[1] in FORBIDDEN_ROOTS:
                    seen.add(parts[1])
            for s in sorted(seen):
                out.append((me, s, f"{'/'.join(rel)}:{node.lineno}"))
    return out


@pytest.mark.parametrize("pkg", _packages())
def test_core_package_is_independent(pkg):
    bad = [f"{loc} -> {tgt}" for me, tgt, loc in violations() if me == pkg]
    assert not bad, f"core/{pkg} depends on other layers:\n  " + "\n  ".join(bad)
