from enum import Enum


class QuestionType(str, Enum):
    """Types of questions"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"
    MATCHING = "matching"
    ESSAY = "essay"


class DifficultyLevel(str, Enum):
    """Difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    ADAPTIVE = "adaptive"


class EvaluationMetric(str, Enum):
    """Metrics for evaluation"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LEARNING_GAIN = "learning_gain"
    NORMALIZED_GAIN = "normalized_gain"
    MASTERY_LEVEL = "mastery_level"


class ResponseCorrectness(str, Enum):
    """Correctness of response"""
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    NOT_ATTEMPTED = "not_attempted"