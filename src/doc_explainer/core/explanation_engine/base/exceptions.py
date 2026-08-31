class ExplanationEngineError(Exception):
    """Base exception for explanation engine"""
    pass


class GenerationError(ExplanationEngineError):
    """Raised when explanation generation fails"""
    pass


class RecommendationError(ExplanationEngineError):
    """Raised when resource recommendation fails"""
    pass


class ContextError(ExplanationEngineError):
    """Raised when context is invalid"""
    pass


class ConfigurationError(ExplanationEngineError):
    """Raised when configuration is invalid"""
    pass