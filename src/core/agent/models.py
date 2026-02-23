from typing import List, Dict
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass

# --- Shared Enums ---
class ExplanationDepth(Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"

class ExplanationStyleEnum(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

# --- Common Metadata ---
class ExplanationMetadata(BaseModel):
    estimated_complexity: float = Field(description="Score from 0.0 to 1.0")
    user_knowledge_matched: bool = Field(description="Whether known concepts were used")
    gap_bridging: bool = Field(description="Whether unknown concepts were successfully linked")

# --- Resource Objects ---
class ResourceSuggestion(BaseModel):
    concept: str = Field(description="The specific concept requiring extra resources")
    resource_type: str = Field(description="video, article, or exercise")
    difficulty: str = Field(description="beginner, intermediate, or advanced")

@dataclass
class Resource:
    title: str
    url: str
    type: str
    description: str
    difficulty: str

# --- The Final Unified Explanation Model ---
# This is what the rest of your app (Sidebar, Pipeline) will use.
class Explanation(BaseModel):
    explanation: str
    style: Dict
    context_used: Dict
    known_concepts_used: List[str]
    unknown_concepts_explained: List[str]
    follow_up_questions: List[str]
    metadata: ExplanationMetadata
    # We add the "Enriched" resources here after the Recommender runs
    resources: List[Resource] = [] 

# --- The LLM Output Parser Schema ---
# This is used ONLY by the PydanticOutputParser to guide the LLM.
class ExplanationPydantic(BaseModel):
    explanation: str = Field(description="The main text of the response")
    style: Dict = Field(description="Tone, depth, and structural choices")
    context_used: Dict = Field(description="Context bits from document/session used")
    known_concepts_used: List[str] = Field(description="Concepts bridged from user profile")
    unknown_concepts_explained: List[str] = Field(description="New concepts defined in this response")
    suggested_resources: List[ResourceSuggestion] = Field(description="Resources to fetch")
    follow_up_questions: List[str] = Field(description="3 context-aware questions")
    metadata: ExplanationMetadata