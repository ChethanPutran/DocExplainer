from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LongTermMemoryConfig:
    """Configuration for long-term memory"""
    file_path: str = "data/memory/user_memory.json"
    auto_save: bool = True
    max_concept_traces: int = 100
    enable_forgetting_curve: bool = True


@dataclass
class SessionMemoryConfig:
    """Configuration for session memory"""
    max_session_interactions: int = 1000
    auto_clear_on_exit: bool = False


@dataclass
class ForgettingCurveConfig:
    """Configuration for forgetting curve"""
    strategy: str = "ebbinghaus"  # ebbinghaus or power_law
    decay_factor: float = 0.1
    alpha: float = 0.5
    beta: float = 1.0


@dataclass
class ReviewSchedulerConfig:
    """Configuration for review scheduler"""
    strategy: str = "spaced_repetition"  # spaced_repetition or leitner
    initial_interval: float = 1.0  # hours
    multiplier: float = 2.0


@dataclass
class MemoryModuleConfig:
    """Main configuration for memory module"""
    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)
    session: SessionMemoryConfig = field(default_factory=SessionMemoryConfig)
    forgetting_curve: ForgettingCurveConfig = field(default_factory=ForgettingCurveConfig)
    review_scheduler: ReviewSchedulerConfig = field(default_factory=ReviewSchedulerConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MemoryModuleConfig':
        """Create config from dictionary"""
        config = cls()
        
        if 'long_term' in config_dict:
            lt = config_dict['long_term']
            config.long_term.file_path = lt.get('file_path', config.long_term.file_path)
            config.long_term.auto_save = lt.get('auto_save', config.long_term.auto_save)
            config.long_term.max_concept_traces = lt.get('max_concept_traces', config.long_term.max_concept_traces)
            config.long_term.enable_forgetting_curve = lt.get('enable_forgetting_curve', config.long_term.enable_forgetting_curve)
        
        if 'session' in config_dict:
            sess = config_dict['session']
            config.session.max_session_interactions = sess.get('max_session_interactions', config.session.max_session_interactions)
            config.session.auto_clear_on_exit = sess.get('auto_clear_on_exit', config.session.auto_clear_on_exit)
        
        if 'forgetting_curve' in config_dict:
            fc = config_dict['forgetting_curve']
            config.forgetting_curve.strategy = fc.get('strategy', config.forgetting_curve.strategy)
            config.forgetting_curve.decay_factor = fc.get('decay_factor', config.forgetting_curve.decay_factor)
            config.forgetting_curve.alpha = fc.get('alpha', config.forgetting_curve.alpha)
            config.forgetting_curve.beta = fc.get('beta', config.forgetting_curve.beta)
        
        if 'review_scheduler' in config_dict:
            rs = config_dict['review_scheduler']
            config.review_scheduler.strategy = rs.get('strategy', config.review_scheduler.strategy)
            config.review_scheduler.initial_interval = rs.get('initial_interval', config.review_scheduler.initial_interval)
            config.review_scheduler.multiplier = rs.get('multiplier', config.review_scheduler.multiplier)
        
        return config