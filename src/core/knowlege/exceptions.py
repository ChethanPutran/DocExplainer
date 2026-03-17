class KnowledgeBaseError(Exception):
    """Base exception for knowledge module"""
    pass

class ConceptNotFoundError(KnowledgeBaseError):
    """Raised when a concept is not found"""
    pass

class RelationshipNotFoundError(KnowledgeBaseError):
    """Raised when a relationship is not found"""
    pass

class ExtractionError(KnowledgeBaseError):
    """Raised when concept extraction fails"""
    pass

class CanonicalizationError(KnowledgeBaseError):
    """Raised when concept canonicalization fails"""
    pass

class GraphError(KnowledgeBaseError):
    """Raised when graph operations fail"""
    pass

class CycleDetectedError(GraphError):
    """Raised when a cycle is detected in the graph"""
    pass

class InvalidConfigurationError(KnowledgeBaseError):
    """Raised when configuration is invalid"""
    pass