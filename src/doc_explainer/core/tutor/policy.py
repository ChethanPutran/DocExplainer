"""The adaptation policy: how an answer should be pitched for THIS reader on THIS passage.

Pure functions, no model and no I/O. Given the concepts in play and what we believe the reader knows
about each, decide which terms to take for granted, which to define, which to refresh in a phrase,
how many new terms one answer may introduce, and how long and how technical it should be. The aim:
never lose a beginner under jargon, never bore an expert with definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from ..common.knowledge import Concept
from ..common.learner import level_for

KNOWN = 0.70          # at or above: take it for granted
GAP = 0.40            # below: the reader needs it explained

MAX_NEW_TERMS = {"beginner": 2, "intermediate": 3, "advanced": 4, "expert": 5}
WORDS = {"beginner": (120, 200), "intermediate": (100, 170), "advanced": (80, 140), "expert": (60, 110)}
STYLE = {
    "beginner": "Use plain everyday words and one concrete analogy. Short sentences. Define every technical term the first time you use it.",
    "intermediate": "Be clear and direct. Define the terms listed below; use standard terminology for the rest.",
    "advanced": "Be concise and precise. Use technical terms freely; spend words on reasoning, not basics.",
    "expert": "Be terse and rigorous. Skip anything the reader already knows; point out subtleties and edge cases.",
}


@dataclass
class FocusConcept:
    concept: Concept
    p_known: float
    evidence: int = 0


@dataclass
class AdaptationPlan:
    level: str
    depth: str                                    # "adaptive" while we are still unsure of the reader, else "fixed"
    mastery: float                                # importance-weighted belief over the concepts in play
    assume: List[Concept] = field(default_factory=list)
    define: List[Concept] = field(default_factory=list)
    refresh: List[Concept] = field(default_factory=list)
    max_new_terms: int = 3
    words: tuple = (100, 170)
    style: str = ""
    gaps: List[Concept] = field(default_factory=list)      # every concept below GAP (define may be a subset)


def with_level(plan: AdaptationPlan, level: str) -> AdaptationPlan:
    """Honour an explicit request for a level ("explain it like I'm a beginner") over what we infer."""
    if level == plan.level:
        return plan
    lo, hi = WORDS[level]
    summarising = plan.max_new_terms == 0 and not plan.define
    plan.level, plan.style = level, STYLE[level]
    plan.words = (lo // 2, hi // 2) if summarising else (lo, hi)
    if not summarising:
        plan.max_new_terms = MAX_NEW_TERMS[level]
        plan.define = plan.gaps[: plan.max_new_terms]
    return plan


def plan_adaptation(focus: Sequence[FocusConcept], mode: str = "explain", default_prior: float = 0.35) -> AdaptationPlan:
    """``mode``: "explain" and "ask" teach; "summarize" condenses and defines nothing."""
    if not focus:
        level, mastery = level_for(default_prior), default_prior
    else:
        weight = sum(max(f.concept.importance, 1) for f in focus)
        mastery = sum(max(f.concept.importance, 1) * f.p_known for f in focus) / weight
        level = level_for(mastery)
    unsure = bool(focus) and sum(f.evidence for f in focus) / len(focus) < 2
    lo, hi = WORDS[level]
    plan = AdaptationPlan(level=level, depth="adaptive" if (unsure or not focus) else "fixed", mastery=round(mastery, 3),
                          max_new_terms=0 if mode == "summarize" else MAX_NEW_TERMS[level],
                          words=(lo // 2, hi // 2) if mode == "summarize" else (lo, hi), style=STYLE[level])
    by_need = sorted(focus, key=lambda f: (-(max(f.concept.importance, 1) * (1 - f.p_known)), f.concept.introduced_ordinal))
    plan.gaps = [f.concept for f in by_need if f.p_known < GAP]
    plan.define = [] if mode == "summarize" else plan.gaps[: plan.max_new_terms]
    plan.refresh = [f.concept for f in by_need if GAP <= f.p_known < KNOWN][:3]
    plan.assume = [f.concept for f in sorted(focus, key=lambda f: -max(f.concept.importance, 1)) if f.p_known >= KNOWN][:8]
    return plan
