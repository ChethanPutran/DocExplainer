from dataclasses import dataclass
from ....core.common.enums import ResourceType, ExplanationLevel


@dataclass
class Resource:
    """Learning resource"""
    title: str
    url: str
    type: ResourceType
    description: str
    difficulty: ExplanationLevel
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'url': self.url,
            'type': self.type.value,
            'description': self.description,
            'difficulty': self.difficulty
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Resource':
        """Create from dictionary"""
        return cls(
            title=data['title'],
            url=data['url'],
            type=ResourceType(data['type']),
            description=data['description'],
            difficulty=ExplanationLevel(data['difficulty'])
        )