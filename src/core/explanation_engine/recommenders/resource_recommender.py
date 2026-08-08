import urllib.parse
from typing import List, Optional

from ..base.interfaces import RecommendationStrategy
from ..models.dataclasses import Resource
from src.core.common.enums import ResourceType, ExplanationLevel
from .base import BaseResourceRecommender
from .strategies.video_strategy import VideoRecommendationStrategy
from .strategies.article_strategy import ArticleRecommendationStrategy
from .strategies.exercise_strategy import ExerciseRecommendationStrategy


class ResourceRecommender(BaseResourceRecommender):
    """
    Resource recommender that uses different strategies for different resource types.
    Generates search links for educational resources.
    """
    
    def __init__(self):
        # Base URLs for different platforms
        self.base_youtube_url = "https://www.youtube.com/results?search_query="
        self.base_scholar_url = "https://scholar.google.com/scholar?q="
        self.base_google_url = "https://www.google.com/search?q="
        self.base_khan_academy = "https://www.khanacademy.org/search?search_again=1&page_search_query="
        
        super().__init__()
    
    def _register_default_strategies(self):
        """Register default recommendation strategies"""
        self.register_strategy(VideoRecommendationStrategy(self))
        self.register_strategy(ArticleRecommendationStrategy(self))
        self.register_strategy(ExerciseRecommendationStrategy(self))
    
    # Public API methods
    
    def recommend_videos(self, concept: str, level: str) -> Resource:
        """Recommend educational videos"""
        query = self._build_search_query(concept, level)
        
        return Resource(
            title=f"Video Tutorial: {concept} ({level})",
            url=f"{self.base_youtube_url}{urllib.parse.quote(query)}",
            type=ResourceType.VIDEO,
            description=f"Watch video tutorials to understand {concept} at {level} level.",
            difficulty=self._parse_level(level),
            source="YouTube"
        )
    
    def recommend_articles(self, concept: str, level: str) -> Resource:
        """Recommend articles and papers"""
        query = self._build_search_query(concept, level)
        
        return Resource(
            title=f"Reading: {concept} Explained",
            url=f"{self.base_scholar_url}{urllib.parse.quote(query)}",
            type=ResourceType.ARTICLE,
            description=f"Read articles and papers about {concept} at {level} level.",
            difficulty=self._parse_level(level),
            source="Google Scholar"
        )
    
    def recommend_exercises(self, concept: str, level: str) -> Resource:
        """Recommend practice exercises"""
        query = self._build_search_query(concept, level, include_terms=["practice", "exercises", "quiz"])
        
        return Resource(
            title=f"Practice: {concept} Exercises",
            url=f"{self.base_google_url}{urllib.parse.quote(query)}",
            type=ResourceType.EXERCISE,
            description=f"Test your knowledge of {concept} with interactive exercises.",
            difficulty=self._parse_level(level),
            source="Google Search"
        )
    
    def recommend_khan_academy(self, concept: str, level: str) -> Optional[Resource]:
        """Recommend Khan Academy resources"""
        query = self._build_search_query(concept, level)
        
        return Resource(
            title=f"Khan Academy: {concept}",
            url=f"{self.base_khan_academy}{urllib.parse.quote(query)}",
            type=ResourceType.COURSE,
            description=f"Structured learning for {concept} on Khan Academy.",
            difficulty=self._parse_level(level),
            source="Khan Academy"
        )
    
    def recommend_all(self, concept: str, level: str = "intermediate") -> List[Resource]:
        """Recommend all resource types"""
        resources = [
            self.recommend_videos(concept, level),
            self.recommend_articles(concept, level),
            self.recommend_exercises(concept, level)
        ]
        
        # Add Khan Academy if available
        khan = self.recommend_khan_academy(concept, level)
        if khan:
            resources.append(khan)
        
        return resources
    
    def recommend_by_type(self, concept: str, level: str, 
                          resource_type: ResourceType) -> Resource:
        """Recommend a specific resource type"""
        if resource_type == ResourceType.VIDEO:
            return self.recommend_videos(concept, level)
        elif resource_type == ResourceType.ARTICLE:
            return self.recommend_articles(concept, level)
        elif resource_type == ResourceType.EXERCISE:
            return self.recommend_exercises(concept, level)
        elif resource_type == ResourceType.COURSE:
            result = self.recommend_khan_academy(concept, level)
            if result:
                return result
        
        # Fallback
        return Resource(
            title=f"Resources for {concept}",
            url=Resource.create_search_link(concept, level),
            type=resource_type,
            description=f"Find resources for {concept} at {level} level.",
            difficulty=self._parse_level(level)
        )
    
    # Helper methods
    
    def _build_search_query(self, concept: str, level: str, 
                           include_terms: List[str] = None) -> str: # type: ignore
        """Build a search query string"""
        base_query = f"{concept} tutorial {level}"
        
        if include_terms:
            base_query += " " + " ".join(include_terms)
        
        return base_query
    
    def _parse_level(self, level: str) -> ExplanationLevel:
        """Parse level string to enum"""
        try:
            return ExplanationLevel(level.lower())
        except ValueError:
            # Default to intermediate
            return ExplanationLevel.INTERMEDIATE
    
    def create_custom_search(self, concept: str, level: str, 
                            platform: str = "google") -> str:
        """Create a custom search link"""
        return Resource.create_search_link(concept, level, platform)