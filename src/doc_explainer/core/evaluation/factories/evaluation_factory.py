from typing import Optional, Dict, Any

from ..generators.quiz_generator import QuizGenerator
from ..evaluators.response_evaluator import ResponseEvaluator
from ..analytics.learning_gain import LearningGainCalculator
from ..analytics.difficulty_analyzer import DifficultyAnalyzer
from ..analytics.mastery_tracker import MasteryTracker
from ..evaluators.knowledge_evaluator import KnowledgeEvaluator
from ..config import EvaluationConfig


class EvaluationFactory:
    """Factory for creating evaluation components"""
    
    @classmethod
    def create_quiz_generator(cls, use_llm: bool = False, 
                             llm_wrapper=None) -> QuizGenerator:
        """Create a quiz generator"""
        return QuizGenerator(use_llm=use_llm, llm_wrapper=llm_wrapper)
    
    @classmethod
    def create_response_evaluator(cls, similarity_threshold: float = 0.8,
                                 enable_partial_credit: bool = True) -> ResponseEvaluator:
        """Create a response evaluator"""
        return ResponseEvaluator(
            similarity_threshold=similarity_threshold,
            enable_partial_credit=enable_partial_credit
        )
    
    @classmethod
    def create_learning_gain_calculator(cls) -> LearningGainCalculator:
        """Create a learning gain calculator"""
        return LearningGainCalculator()
    
    @classmethod
    def create_difficulty_analyzer(cls) -> DifficultyAnalyzer:
        """Create a difficulty analyzer"""
        return DifficultyAnalyzer()
    
    @classmethod
    def create_mastery_tracker(cls, decay_rate: float = 0.1) -> MasteryTracker:
        """Create a mastery tracker"""
        return MasteryTracker(decay_rate=decay_rate)
    
    @classmethod
    def create_knowledge_evaluator(cls, 
                                  quiz_generator: Optional[QuizGenerator] = None,
                                  response_evaluator: Optional[ResponseEvaluator] = None,
                                  mastery_tracker: Optional[MasteryTracker] = None,
                                  config: Optional[EvaluationConfig] = None) -> KnowledgeEvaluator:
        """Create a knowledge evaluator"""
        if config is None:
            config = EvaluationConfig()
        
        return KnowledgeEvaluator(
            quiz_generator=quiz_generator or cls.create_quiz_generator(),
            response_evaluator=response_evaluator or cls.create_response_evaluator(),
            mastery_tracker=mastery_tracker or cls.create_mastery_tracker(),
            config=config
        )
    
    @classmethod
    def create_default(cls) -> KnowledgeEvaluator:
        """Create default knowledge evaluator with all components"""
        return cls.create_knowledge_evaluator()
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> KnowledgeEvaluator:
        """Create knowledge evaluator from configuration"""
        config = EvaluationConfig.from_dict(config_dict)
        return cls.create_knowledge_evaluator(config=config)