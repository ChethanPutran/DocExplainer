"""Shared learner-model data types (data only; the logic lives in ``core.learner``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Belief:
    """What we currently believe about one concept for one learner."""
    p_known: float          # probability the learner knows it, in [0, 1]
    evidence: int = 0       # how many observations support the estimate (0 = just the prior)
    updated_at: float = 0.0


class EvidenceKind(str, Enum):
    QUIZ_ANSWER = "quiz_answer"                 # strong: a graded answer
    QUESTION_ASKED = "question_asked"           # weak: what kind of question they ask says what they know
    EXPLANATION_FEEDBACK = "explanation_feedback"   # weak: "too easy" / "too hard"
    READ_SECTION = "read_section"               # very weak: they were exposed to it
    SELF_REPORT = "self_report"                 # medium: "I know this"


@dataclass
class Evidence:
    kind: EvidenceKind
    concept_ids: List[str]
    correct: Optional[bool] = None              # quiz answers
    difficulty: float = 0.5                     # 0 easy .. 1 hard (quiz answers)
    question_type: str = ""                     # definition | deeper (questions)
    verdict: str = ""                           # too_easy | too_hard | just_right (feedback)
    value: Optional[float] = None               # self reports: 0..1
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefChange:
    concept_id: str
    before: float
    after: float


@dataclass
class LearnerProfile:
    user_id: str
    doc_id: str
    prior: float                                # belief for any concept with no evidence yet
    level: str                                  # beginner | intermediate | advanced | expert
    onboarding: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


def level_for(p: float) -> str:
    return "beginner" if p < 0.3 else "intermediate" if p < 0.55 else "advanced" if p < 0.78 else "expert"
