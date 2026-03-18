from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .models.enums import DifficultyLevel


@dataclass
class EvaluationConfig:
    """Configuration for evaluation module"""
    
    # Quiz generation settings
    default_num_questions: int = 5
    default_difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    use_llm: bool = False
    
    # Response evaluation settings
    similarity_threshold: float = 0.8
    enable_partial_credit: bool = True
    
    # Mastery tracking settings
    decay_rate: float = 0.1
    mastery_threshold: float = 0.7
    
    # Quiz settings
    remediation_quiz_size: int = 3
    max_question_attempts: int = 3
    
    # Question generation weights
    question_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "multiple_choice": 0.5,
        "true_false": 0.3,
        "fill_blank": 0.2
    })
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create config from dictionary"""
        config = cls()
        
        if 'default_num_questions' in config_dict:
            config.default_num_questions = config_dict['default_num_questions']
        
        if 'default_difficulty' in config_dict:
            diff = config_dict['default_difficulty']
            if isinstance(diff, str):
                config.default_difficulty = DifficultyLevel(diff)
        
        if 'use_llm' in config_dict:
            config.use_llm = config_dict['use_llm']
        
        if 'similarity_threshold' in config_dict:
            config.similarity_threshold = config_dict['similarity_threshold']
        
        if 'enable_partial_credit' in config_dict:
            config.enable_partial_credit = config_dict['enable_partial_credit']
        
        if 'decay_rate' in config_dict:
            config.decay_rate = config_dict['decay_rate']
        
        if 'mastery_threshold' in config_dict:
            config.mastery_threshold = config_dict['mastery_threshold']
        
        if 'remediation_quiz_size' in config_dict:
            config.remediation_quiz_size = config_dict['remediation_quiz_size']
        
        if 'question_type_weights' in config_dict:
            config.question_type_weights.update(config_dict['question_type_weights'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'default_num_questions': self.default_num_questions,
            'default_difficulty': self.default_difficulty.value,
            'use_llm': self.use_llm,
            'similarity_threshold': self.similarity_threshold,
            'enable_partial_credit': self.enable_partial_credit,
            'decay_rate': self.decay_rate,
            'mastery_threshold': self.mastery_threshold,
            'remediation_quiz_size': self.remediation_quiz_size,
            'question_type_weights': self.question_type_weights
        }