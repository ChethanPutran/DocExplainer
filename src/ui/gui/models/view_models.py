from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ExplanationViewModel:
    """View model for explanations"""
    explanation: str
    known_concepts: List[str] = field(default_factory=list)
    unknown_concepts: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    section_id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_explanation(cls, explanation, section_id: int = 0) -> 'ExplanationViewModel':
        """Create view model from explanation object"""
        return cls(
            explanation=explanation.explanation,
            known_concepts=explanation.known_concepts_used,
            unknown_concepts=explanation.unknown_concepts_explained,
            follow_up_questions=explanation.follow_up_questions,
            resources=[r.to_dict() if hasattr(r, 'to_dict') else r 
                      for r in explanation.resources],
            section_id=section_id
        )


@dataclass
class DocumentInfo:
    """Information about a loaded document"""
    doc_id: str
    path: str
    title: str
    viewer_type: str
    loaded_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class UISession:
    """UI session state"""
    current_doc_id: Optional[str] = None
    current_section_id: int = 0
    theme: str = "light"
    sidebar_visible: bool = True
    voice_enabled: bool = True