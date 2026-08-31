# src/core/user/learning_patterns.py
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from prophet import Prophet
import pandas as pd
from typing import List, Dict, Any

class LearningPatternAnalyzer:
    """Analyze user learning patterns using time series"""
    
    def __init__(self):
        self.regression_model = LinearRegression()
        self.svr_model = SVR(kernel='rbf')
        
    def analyze_learning_trend(self, user_history: List[Dict]) -> Dict[str, Any]:
        """Analyze learning trends over time"""
        # Prepare time series data
        df = pd.DataFrame(user_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate learning rate
        df['cumulative_knowledge'] = df['knowledge_gain'].cumsum()
        
        # Fit regression model
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['cumulative_knowledge'].values
        
        self.regression_model.fit(X, y)
        learning_rate = self.regression_model.coef_[0]
        
        # Predict future performance
        future_steps = 10
        future_X = np.arange(len(df), len(df) + future_steps).reshape(-1, 1)
        future_predictions = self.regression_model.predict(future_X)
        
        return {
            'learning_rate': learning_rate,
            'trend_direction': 'increasing' if learning_rate > 0 else 'decreasing',
            'consistency_score': self._calculate_consistency(df['knowledge_gain'].values),
            'predictions': future_predictions.tolist()
        }
    
    def detect_learning_plateaus(self, performance_history: List[float]) -> List[Dict]:
        """Detect when user hits learning plateaus"""
        plateaus = []
        
        # Use rolling window to detect stagnation
        window_size = 5
        for i in range(len(performance_history) - window_size):
            window = performance_history[i:i+window_size]
            if np.std(window) < 0.05:  # Very low variance indicates plateau
                plateaus.append({
                    'start_index': i,
                    'duration': window_size,
                    'avg_performance': np.mean(window)
                })
        
        return plateaus