from dataclasses import dataclass
from typing import Optional
from .enums import ExplanationDepth, ExplanationLevel

@dataclass
class ExplanationStyle:
    """Explanation style dataclass
      How to explain a concept, including the level of detail and depth of explanation."""
    level: ExplanationLevel
    depth: ExplanationDepth
    description: str
    examples: Optional[list] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'ExplanationStyle':
        """Create ExplanationStyle from dictionary"""
        return cls(
            level=ExplanationLevel(data['level']),
            depth=ExplanationDepth(data['depth']),
            description=data['description'],
            examples=data.get('examples')
        )

    
    def to_dict(self) -> dict:
        """Convert ExplanationStyle to dictionary"""
        return {
            'level': self.level.value,
            'depth': self.depth.value,
            'description': self.description,
            'examples': self.examples
        }
    
    @classmethod
    def get_default_style(cls) -> 'ExplanationStyle':
        """Return default explanation style"""
        return ExplanationStyle(
            level=ExplanationLevel.INTERMEDIATE,
            depth=ExplanationDepth.FIXED,
            description="Default explanation style with intermediate level and fixed depth."
        )

    def get_style(self) -> str:
        """Return a string representation of the style"""
        return f"Level: {self.level.value}, Depth: {self.depth.value}, Description: {self.description}"
    