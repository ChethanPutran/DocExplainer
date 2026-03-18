from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.core.agent.models.enums import ExplanationStyleEnum


@dataclass
class ExplanationEngineConfig:
    """Configuration for explanation engine"""
    
    # Default settings
    default_level: ExplanationStyleEnum = ExplanationStyleEnum.INTERMEDIATE
    
    # Resource recommendation settings
    enable_video_recommendations: bool = True
    enable_article_recommendations: bool = True
    enable_exercise_recommendations: bool = True
    
    # Search settings
    youtube_enabled: bool = True
    scholar_enabled: bool = True
    khan_academy_enabled: bool = False
    
    # Custom URLs (can be overridden)
    youtube_base_url: str = "https://www.youtube.com/results?search_query="
    scholar_base_url: str = "https://scholar.google.com/scholar?q="
    google_base_url: str = "https://www.google.com/search?q="
    khan_academy_base_url: str = "https://www.khanacademy.org/search?search_again=1&page_search_query="
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExplanationEngineConfig':
        """Create config from dictionary"""
        config = cls()
        
        if 'default_level' in config_dict:
            level_val = config_dict['default_level']
            if isinstance(level_val, str):
                config.default_level = ExplanationStyleEnum(level_val)
        
        if 'enable_video_recommendations' in config_dict:
            config.enable_video_recommendations = config_dict['enable_video_recommendations']
        
        if 'enable_article_recommendations' in config_dict:
            config.enable_article_recommendations = config_dict['enable_article_recommendations']
        
        if 'enable_exercise_recommendations' in config_dict:
            config.enable_exercise_recommendations = config_dict['enable_exercise_recommendations']
        
        if 'youtube_enabled' in config_dict:
            config.youtube_enabled = config_dict['youtube_enabled']
        
        if 'scholar_enabled' in config_dict:
            config.scholar_enabled = config_dict['scholar_enabled']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'default_level': self.default_level.value,
            'enable_video_recommendations': self.enable_video_recommendations,
            'enable_article_recommendations': self.enable_article_recommendations,
            'enable_exercise_recommendations': self.enable_exercise_recommendations,
            'youtube_enabled': self.youtube_enabled,
            'scholar_enabled': self.scholar_enabled,
            'khan_academy_enabled': self.khan_academy_enabled
        }