# src/core/rl/adaptive_learning.py
import numpy as np
from collections import defaultdict
import random

class QLearningAdaptivePath:
    """Reinforcement Learning for adaptive learning path optimization"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
    def get_state(self, user_knowledge_state: Dict) -> str:
        """Convert user knowledge state to state string"""
        state_parts = []
        for concept, mastery in user_knowledge_state.items():
            level = 'beginner' if mastery < 0.3 else 'intermediate' if mastery < 0.7 else 'advanced'
            state_parts.append(f"{concept}:{level}")
        return "|".join(state_parts)
    
    def get_actions(self, available_content: List[str]) -> List[str]:
        """Get possible actions (content to recommend)"""
        return available_content
    
    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(available_actions)
        else:
            # Exploit: choose best action
            if not self.q_table[state]:
                return random.choice(available_actions)
            return max(available_actions, key=lambda a: self.q_table[state][a])
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-values based on reward"""
        if not self.q_table[next_state]:
            best_next_value = 0
        else:
            best_next_value = max(self.q_table[next_state].values())
        
        current_q = self.q_table[state][action]
        new_q = current_q + self.lr * (reward + self.gamma * best_next_value - current_q)
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, user_performance: float, time_spent: float, difficulty: float) -> float:
        """Calculate reward based on user interaction"""
        # Positive reward for good performance
        performance_reward = user_performance * 10
        
        # Penalty for too much time
        time_penalty = -min(5, time_spent / 60)  # Penalize after 60 seconds
        
        # Reward for appropriate difficulty
        difficulty_reward = 5 if 0.3 <= difficulty <= 0.7 else -2
        
        return performance_reward + time_penalty + difficulty_reward