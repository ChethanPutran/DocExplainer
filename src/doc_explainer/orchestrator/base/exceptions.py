class OrchestratorError(Exception):
    """Base exception for orchestrator module"""
    pass


class PipelineError(OrchestratorError):
    """Raised when pipeline processing fails"""
    pass


class ValidationError(OrchestratorError):
    """Raised when request validation fails"""
    pass


class DocumentNotFoundError(OrchestratorError):
    """Raised when document is not found"""
    pass


class ContextBuildError(OrchestratorError):
    """Raised when context building fails"""
    pass


class ServiceInitializationError(OrchestratorError):
    """Raised when service initialization fails"""
    pass


class ConfigurationError(OrchestratorError):
    """Raised when configuration is invalid"""
    pass