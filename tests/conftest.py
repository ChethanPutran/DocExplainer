import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(autouse=True)
def _fresh_shared_llm_clients():
    """LLM wrappers with identical settings share one client; never let that leak between tests."""
    try:
        from doc_explainer.core.llm.factories.llm_factory import LLMFactory
    except Exception:  # backend not importable in this environment
        yield
        return
    LLMFactory.clear_shared()
    yield
    LLMFactory.clear_shared()
