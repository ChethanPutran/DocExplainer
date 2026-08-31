from abc import ABC, abstractmethod
from typing import Dict
from datetime import datetime


class ForgettingCurveStrategy(ABC):
    """Strategy for calculating forgetting curves"""
    
    @abstractmethod
    def calculate_retention(self, last_access: datetime, **kwargs) -> float:
        """Calculate retention probability"""
        pass


class EbbinghausForgettingCurve(ForgettingCurveStrategy):
    """Ebbinghaus forgetting curve implementation"""
    
    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor
    
    def calculate_retention(self, last_access: datetime, **kwargs) -> float:
        """Calculate retention using Ebbinghaus formula"""
        hours_since = (datetime.now() - last_access).total_seconds() / 3600
        return 1.0 / (1.0 + self.decay_factor * hours_since)


class PowerLawForgettingCurve(ForgettingCurveStrategy):
    """Power law forgetting curve"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
    
    def calculate_retention(self, last_access: datetime, **kwargs) -> float:
        """Calculate retention using power law"""
        hours_since = (datetime.now() - last_access).total_seconds() / 3600
        return 1.0 / (1.0 + self.alpha * (hours_since ** self.beta))