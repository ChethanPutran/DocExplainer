from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime, timedelta


class ReviewScheduler(ABC):
    """Strategy for scheduling reviews"""
    
    @abstractmethod
    def schedule_next_review(self, retention: float, review_count: int) -> timedelta:
        """Schedule next review based on retention"""
        pass


class SpacedRepetitionScheduler(ReviewScheduler):
    """Spaced repetition scheduler"""
    
    def __init__(self, initial_interval: float = 1.0, multiplier: float = 2.0):
        self.initial_interval = initial_interval  # hours
        self.multiplier = multiplier
    
    def schedule_next_review(self, retention: float, review_count: int) -> timedelta:
        """Schedule next review using spaced repetition"""
        if retention < 0.4:
            # Need immediate review
            interval_hours = self.initial_interval
        elif retention < 0.6:
            # Review soon
            interval_hours = self.initial_interval * self.multiplier
        elif retention < 0.8:
            # Review later
            interval_hours = self.initial_interval * (self.multiplier ** 2)
        else:
            # Review much later
            interval_hours = self.initial_interval * (self.multiplier ** 3)
        
        # Adjust for review count
        interval_hours *= (review_count + 1)
        
        return timedelta(hours=interval_hours)


class LeitnerSystemScheduler(ReviewScheduler):
    """Leitner system scheduler"""
    
    def __init__(self, boxes: Dict[int, float] = None):
        self.boxes = boxes or {
            1: 1.0,   # Box 1: 1 hour
            2: 24.0,  # Box 2: 1 day
            3: 168.0, # Box 3: 1 week
            4: 720.0, # Box 4: 1 month
            5: 2160.0 # Box 5: 3 months
        }
    
    def schedule_next_review(self, retention: float, review_count: int) -> timedelta:
        """Schedule next review using Leitner system"""
        # Determine box based on retention
        if retention < 0.4:
            box = 1
        elif retention < 0.6:
            box = 2
        elif retention < 0.8:
            box = 3
        elif retention < 0.9:
            box = 4
        else:
            box = 5
        
        interval_hours = self.boxes.get(box, 24.0)
        return timedelta(hours=interval_hours)