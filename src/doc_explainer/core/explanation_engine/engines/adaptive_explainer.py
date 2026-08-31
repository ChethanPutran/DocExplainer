from typing import Optional, Any, List
import logging

from doc_explainer.core.common.enums import ExplanationLevel

from ...agent import Agent
from ...agent.models.schemas import Explanation, ResourceSuggestion
from ...common.dataclasses import ExplanationStyle
from ..base.exceptions import RecommendationError
from ..recommenders.resource_recommender import ResourceRecommender
from ..models.dataclasses import Resource
from .base import BaseExplanationEngine

logger = logging.getLogger(__name__)


class AdaptiveExplainer(BaseExplanationEngine):
    """
    Adaptive explainer that adjusts explanations based on user knowledge
    and enriches them with recommended resources.
    """
    
    def __init__(self, 
                 agent: Agent,
                 recommender: Optional[ResourceRecommender] = None,
                 default_level: ExplanationStyle=ExplanationStyle.get_default_style()):
        super().__init__(agent, default_level)
        self.recommender = recommender or ResourceRecommender()
    
    def summarize(self, text: str, context: Any) -> Explanation:
        """Generate summary and enrich with resources"""
        explanation = super().summarize(text, context)
        return self.enrich_with_resources(explanation)
    
    def explain(self, text: str, context: Any) -> Explanation:
        """Generate explanation and enrich with resources"""
        explanation = super().explain(text, context)
        return self.enrich_with_resources(explanation)
    
    def ask(self, question: str, context: Any) -> Explanation:
        """Answer question and enrich with resources"""
        explanation = super().ask(question, context)
        return self.enrich_with_resources(explanation)
    
    def enrich_with_resources(self, explanation: Explanation) -> Explanation:
        """
        Enrich explanation with recommended resources based on suggestions
        
        This converts ResourceSuggestion objects to actual Resource objects
        using the recommender.
        """
        try:
            enriched_resources = []
            
            # Check if explanation has suggested_resources attribute
            if hasattr(explanation, 'suggested_resources') and explanation.suggested_resources:
                for suggestion in explanation.suggested_resources:
                    resource = self._get_resource_from_suggestion(suggestion)
                    if resource:
                        enriched_resources.append(resource)
            
            # Also check if there are unknown concepts that might need resources
            if not enriched_resources and explanation.unknown_concepts_explained:
                for concept in explanation.unknown_concepts_explained[:3]:  # Limit to 3
                    # Get level from explanation style
                    level = self.current_level
                    
                    # Get all resource types for this concept
                    resources = self.recommender.recommend_all(concept, level)
                    enriched_resources.extend(resources)
            
            # Add resources to explanation
            explanation.resources = enriched_resources
            
        except Exception as e:
            logger.error(f"Failed to enrich with resources: {e}")
            # Don't fail the whole explanation if resource enrichment fails
            explanation.resources = []
        
        return explanation
    
    def _get_resource_from_suggestion(self, suggestion: 'ResourceSuggestion') -> Optional[Resource]:
        """
        Convert a ResourceSuggestion to an actual Resource
        """
        try:
            concept = suggestion.concept
            difficulty = suggestion.difficulty 
            
            if suggestion.resource_type == "video" or suggestion.resource_type.value == "video":
                return self.recommender.recommend_videos(concept, difficulty)
            elif suggestion.resource_type == "article" or suggestion.resource_type.value == "article":
                return self.recommender.recommend_articles(concept, difficulty)
            elif suggestion.resource_type == "exercise" or suggestion.resource_type.value == "exercise":
                return self.recommender.recommend_exercises(concept, difficulty)
            else:
                logger.warning(f"Unknown resource type: {suggestion.resource_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to get resource from suggestion: {e}")
            return None
    
    def enrich_specific_concept(self, concept: str, level: Optional[ExplanationLevel] = None) -> List[Resource]:
        """
        Enrich a specific concept with resources
        
        Args:
            concept: The concept to enrich
            level: Optional difficulty level (uses current level if not provided)
        
        Returns:
            List of recommended resources
        """
        if level is None:
            level = self.current_level
        
        return self.recommender.recommend_all(concept, level)
    
    def set_recommender(self, recommender: ResourceRecommender):
        """Set a new resource recommender"""
        self.recommender = recommender
        logger.info("Resource recommender updated")