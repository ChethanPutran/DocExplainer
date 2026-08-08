from enum import Enum



class RecommendationStrategy(str, Enum):
    """Strategies for recommendation"""
    SIMPLE_SEARCH = "simple_search"
    SEMANTIC_SEARCH = "semantic_search"
    KNOWLEDGE_BASED = "knowledge_based"
    POPULARITY_BASED = "popularity_based"



class QueryType(Enum):
    """Type of user query"""
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    EXAMPLE = "example"
    SIMPLIFICATION = "simplification"
    REPETITION = "repetition"
    GENERAL = "general"