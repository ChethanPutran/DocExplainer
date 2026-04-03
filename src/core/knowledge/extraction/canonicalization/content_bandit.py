# src/core/bandit/content_bandit.py
import numpy as np
from scipy.stats import beta

class ThompsonSamplingBandit:
    """Multi-armed bandit for optimal content selection"""
    
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Success counts
        self.beta = np.ones(n_arms)   # Failure counts
        
    def select_arm(self) -> int:
        """Select arm using Thompson Sampling"""
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """Update beta distribution with observed reward"""
        # Convert reward to binary (success/failure)
        if reward > 0.5:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += (1 - reward)
    
    def get_arm_probabilities(self) -> np.ndarray:
        """Get probability of each arm being optimal"""
        probs = []
        for i in range(self.n_arms):
            # Expected value of beta distribution
            prob = self.alpha[i] / (self.alpha[i] + self.beta[i])
            probs.append(prob)
        return np.array(probs)

class ContextualBandit:
    """Contextual bandit for personalized content selection"""
    
    def __init__(self, n_features=20, n_arms=10, alpha=0.1):
        self.n_features = n_features
        self.n_arms = n_arms
        self.alpha = alpha
        self.theta = np.zeros((n_arms, n_features))  # Weights for each arm
        self.covariance = [np.eye(n_features) for _ in range(n_arms)]
        
    def select_arm(self, context: np.ndarray) -> int:
        """Select arm using LinUCB"""
        ucb_values = []
        
        for arm in range(self.n_arms):
            # Calculate mean reward
            mean_reward = np.dot(self.theta[arm], context)
            
            # Calculate confidence bound
            confidence = self.alpha * np.sqrt(np.dot(context.T, np.dot(self.covariance[arm], context)))
            
            ucb_values.append(mean_reward + confidence)
        
        return np.argmax(ucb_values)
    
    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update model with observed reward"""
        # Update covariance matrix
        self.covariance[arm] += np.outer(context, context)
        
        # Update weights using ridge regression
        inv_cov = np.linalg.inv(self.covariance[arm])
        self.theta[arm] = np.dot(inv_cov, np.dot(self.covariance[arm] - np.eye(self.n_features), self.theta[arm]) 
                                 + reward * context)