class EvaluationError(Exception):
    """Base exception for evaluation module"""
    pass


class QuizGenerationError(EvaluationError):
    """Raised when quiz generation fails"""
    pass


class ResponseEvaluationError(EvaluationError):
    """Raised when response evaluation fails"""
    pass


class LearningGainError(EvaluationError):
    """Raised when learning gain calculation fails"""
    pass


class InvalidQuestionError(EvaluationError):
    """Raised when question is invalid"""
    pass


class ConfigurationError(EvaluationError):
    """Raised when configuration is invalid"""
    pass