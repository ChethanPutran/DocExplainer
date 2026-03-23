class AgentError(Exception):
    """Base exception for agent module"""
    pass


class LLMError(AgentError):
    """Raised when LLM operations fail"""
    pass


class ParserError(AgentError):
    """Raised when output parsing fails"""
    pass


class PromptError(AgentError):
    """Raised when prompt operations fail"""
    pass


class ChainError(AgentError):
    """Raised when chain operations fail"""
    pass


class ConfigurationError(AgentError):
    """Raised when configuration is invalid"""
    pass