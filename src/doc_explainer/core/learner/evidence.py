"""How each kind of evidence changes beliefs: the weights, and how a question is classified."""
from __future__ import annotations

import re

# How much each indirect signal counts, relative to a graded answer (1.0).
WEIGHT_ASKED_DEFINITION = 0.30    # "what is X?"  -> probably does not know X
WEIGHT_ASKED_DEEPER = 0.12        # "why does X ...?" -> probably knows X and what it rests on
WEIGHT_TOO_EASY = 0.20
WEIGHT_TOO_HARD = 0.25
WEIGHT_PROPAGATE_UP = 0.25        # a correct answer on X is weak evidence for X's prerequisites
WEIGHT_PROPAGATE_DOWN = 0.20      # a wrong answer on X is weak evidence against concepts built on X
EXPOSURE_AMOUNT = 0.03

_DEFINITION = re.compile(r"^\s*(what\s+(is|are|does|do)\b.*\b(mean|stand for)|what\s+(is|are)\s+(an?\s+|the\s+)?\w+|define\b|meaning of\b|"
                         r"what'?s\b|who\s+(is|was)\b|explain\s+what\b)", re.I)
_DEEPER = re.compile(r"\b(why|how\s+(does|do|can|would|is|are)|compare|difference|differ|versus|vs\.?|relate|relationship|"
                     r"what\s+if|what\s+would|apply|derive|prove|implication|trade-?off|assumption|limitation)\b", re.I)


def classify_question(text: str) -> str:
    """"definition" (asking what something is), "deeper" (why/how/compare...), or "" if unclear."""
    t = text.strip()
    if _DEEPER.search(t):
        return "deeper"
    if _DEFINITION.match(t):
        return "definition"
    return ""
