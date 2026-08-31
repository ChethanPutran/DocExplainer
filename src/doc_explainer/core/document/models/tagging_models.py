"""Data models for paragraph-concept tagging."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
import uuid


@dataclass
class ParagraphTag:
    """Represents a single paragraph-concept tag with confidence."""
    
    paragraph_id: str
    concept_id: str
    concept_name: str
    confidence: float
    ner_confidence: Optional[float] = None
    llm_confidence: Optional[float] = None
    tagged_by: Literal['auto', 'manual'] = 'auto'
    method: str = 'hybrid'  # 'ner', 'llm', 'hybrid'
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tag_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.ner_confidence and not 0.0 <= self.ner_confidence <= 1.0:
            raise ValueError(f"NER confidence must be between 0.0 and 1.0, got {self.ner_confidence}")
        if self.llm_confidence and not 0.0 <= self.llm_confidence <= 1.0:
            raise ValueError(f"LLM confidence must be between 0.0 and 1.0, got {self.llm_confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tag_id': self.tag_id,
            'paragraph_id': self.paragraph_id,
            'concept_id': self.concept_id,
            'concept_name': self.concept_name,
            'confidence': self.confidence,
            'ner_confidence': self.ner_confidence,
            'llm_confidence': self.llm_confidence,
            'tagged_by': self.tagged_by,
            'method': self.method,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'attributes': self.attributes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParagraphTag':
        """Create from dictionary."""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class TaggingResult:
    """Result of tagging a single paragraph."""
    
    paragraph_id: str
    paragraph_text: str
    tags: List[ParagraphTag] = field(default_factory=list)
    auto_tags: List[ParagraphTag] = field(default_factory=list)
    manual_tags: List[ParagraphTag] = field(default_factory=list)
    processing_time: float = 0.0
    ner_entities: List[Dict[str, Any]] = field(default_factory=list)
    llm_extracted_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Populate auto and manual tags from all tags."""
        if self.tags:
            self.auto_tags = [t for t in self.tags if t.tagged_by == 'auto']
            self.manual_tags = [t for t in self.tags if t.tagged_by == 'manual']
    
    def add_tag(self, tag: ParagraphTag) -> None:
        """Add a tag to the result."""
        self.tags.append(tag)
        if tag.tagged_by == 'auto':
            self.auto_tags.append(tag)
        else:
            self.manual_tags.append(tag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'paragraph_id': self.paragraph_id,
            'paragraph_text': self.paragraph_text,
            'tags': [t.to_dict() for t in self.tags],
            'auto_tags_count': len(self.auto_tags),
            'manual_tags_count': len(self.manual_tags),
            'processing_time': self.processing_time,
            'ner_entities': self.ner_entities,
            'llm_extracted_concepts': self.llm_extracted_concepts,
            'metadata': self.metadata,
        }


@dataclass
class ConceptMention:
    """Represents a mention of a concept within a paragraph."""
    
    concept_name: str
    entity_type: str  # From NER: PERSON, ORG, PRODUCT, etc.
    start_char: int
    end_char: int
    mention_text: str
    confidence: float = 1.0
    source: Literal['ner', 'llm', 'both'] = 'ner'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'concept_name': self.concept_name,
            'entity_type': self.entity_type,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'mention_text': self.mention_text,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata,
        }


@dataclass
class TaggingConfig:
    """Configuration for the tagging service."""
    
    # NER configuration
    use_ner: bool = True
    ner_confidence_threshold: float = 0.5
    
    # LLM configuration
    use_llm: bool = True
    llm_confidence_threshold: float = 0.6
    llm_model: str = 'gemini-pro'
    
    # Combined scoring
    ner_weight: float = 0.4
    llm_weight: float = 0.6
    combined_confidence_threshold: float = 0.5
    
    # Processing
    max_concepts_per_paragraph: int = 10
    min_concept_length: int = 2
    max_concept_length: int = 50
    
    # Feature flags
    track_confidence_sources: bool = True
    enable_manual_override: bool = True
    learn_from_corrections: bool = True
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not (0.0 <= self.ner_confidence_threshold <= 1.0):
            raise ValueError("ner_confidence_threshold must be between 0 and 1")
        if not (0.0 <= self.llm_confidence_threshold <= 1.0):
            raise ValueError("llm_confidence_threshold must be between 0 and 1")
        if not (0.0 <= self.combined_confidence_threshold <= 1.0):
            raise ValueError("combined_confidence_threshold must be between 0 and 1")
        if not (0.0 <= self.ner_weight <= 1.0):
            raise ValueError("ner_weight must be between 0 and 1")
        if not (0.0 <= self.llm_weight <= 1.0):
            raise ValueError("llm_weight must be between 0 and 1")
        # Weights don't need to sum to 1, but should be reasonable
        if self.ner_weight + self.llm_weight == 0:
            raise ValueError("At least one weight must be positive")
        if self.max_concepts_per_paragraph < 1:
            raise ValueError("max_concepts_per_paragraph must be at least 1")
        if self.min_concept_length < 1:
            raise ValueError("min_concept_length must be at least 1")
        return True
