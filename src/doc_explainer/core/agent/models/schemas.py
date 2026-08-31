from typing import List, Dict
from pydantic import BaseModel, Field 
from ....core.common.enums import ResourceType, ExplanationLevel
from ....core.common.dataclasses import ExplanationStyle

class ExplanationMetadata(BaseModel):
    """Metadata for explanations"""
    estimated_complexity: float = Field(description="Score from 0.0 to 1.0")
    user_knowledge_matched: bool = Field(description="Whether known concepts were used")
    gap_bridging: bool = Field(description="Whether unknown concepts were successfully linked")
    processing_time_ms: float = Field(default=0.0, description="Time taken to generate")


class ResourceSuggestion(BaseModel):
    """Suggestion for learning resources"""
    concept: str = Field(description="The specific concept requiring extra resources")
    resource_type: ResourceType = Field(description="Type of resource")
    difficulty: ExplanationLevel = Field(description="Difficulty level")


class ExplanationPydantic(BaseModel):
    """LLM output schema for explanations"""
    explanation: str = Field(description="The main text of the response")
    style: ExplanationStyle = Field(description="Tone, depth, and structural choices")
    context_used: Dict = Field(description="Context bits from document/session used")
    known_concepts_used: List[str] = Field(description="Concepts bridged from user profile")
    unknown_concepts_explained: List[str] = Field(description="New concepts defined")
    suggested_resources: List[ResourceSuggestion] = Field(description="Resources to fetch")
    follow_up_questions: List[str] = Field(description="3 context-aware questions")
    metadata: ExplanationMetadata


class Explanation(BaseModel):
    """Final unified explanation model"""
    explanation: str
    style: ExplanationStyle
    context_used: Dict
    known_concepts_used: List[str]
    unknown_concepts_explained: List[str]
    follow_up_questions: List[str]
    suggested_resources: List[ResourceSuggestion] = Field(description="Resources to fetch")
    metadata: ExplanationMetadata
    resources: List['Resource'] = []  # Forward reference

