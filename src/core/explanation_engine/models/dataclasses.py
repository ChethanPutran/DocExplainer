from dataclasses import dataclass
from typing import Optional
from src.core.common.enums import ResourceType, ExplanationLevel


@dataclass
class Resource:
    """Learning resource"""
    title: str
    url: str
    type: ResourceType
    description: str
    difficulty: ExplanationLevel
    source: Optional[str] = None
    duration_minutes: Optional[int] = None
    rating: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'url': self.url,
            'type': self.type.value,
            'description': self.description,
            'difficulty': self.difficulty.value,
            'source': self.source,
            'duration_minutes': self.duration_minutes,
            'rating': self.rating
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Resource':
        """Create from dictionary"""
        return cls(
            title=data['title'],
            url=data['url'],
            type=ResourceType(data['type']),
            description=data['description'],
            difficulty=ExplanationLevel(data['difficulty']),
            source=data.get('source'),
            duration_minutes=data.get('duration_minutes'),
            rating=data.get('rating')
        )
    
    @classmethod
    def create_search_link(cls, concept: str, level: str, 
                          platform: str = "google") -> str:
        """Create a search link for resources"""
        import urllib.parse
        query = f"{concept} {level} tutorial"
        
        if platform == "youtube":
            base_url = "https://www.youtube.com/results?search_query="
        elif platform == "scholar":
            base_url = "https://scholar.google.com/scholar?q="
        elif platform == "google":
            base_url = "https://www.google.com/search?q="
        else:
            base_url = "https://www.google.com/search?q="
        
        return f"{base_url}{urllib.parse.quote(query)}"