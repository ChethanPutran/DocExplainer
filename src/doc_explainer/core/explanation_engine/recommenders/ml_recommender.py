# src/core/explanation_engine/recommenders/ml_recommender.py
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import List, Dict, Any
import pandas as pd

class MLRecommendationSystem:
    """Advanced recommendation system using ML algorithms"""
    
    def __init__(self):
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        
    def build_user_profiles(self, user_interactions: List[Dict]) -> np.ndarray:
        """Build user profiles using collaborative filtering"""
        # Create user-item interaction matrix
        user_item_matrix = self._create_interaction_matrix(user_interactions)
        
        # Apply SVD for dimensionality reduction
        user_profiles = self.svd.fit_transform(user_item_matrix)
        
        return user_profiles


    def _create_interaction_matrix(self, user_interactions: List[Dict]) -> np.ndarray:
        """Create a user-item interaction matrix from interactions"""
        df = pd.DataFrame(user_interactions)
        user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='interaction', fill_value=0)
        return user_item_matrix.values
    
    def cluster_users(self, user_features: np.ndarray) -> Dict[int, List[str]]:
        """Group similar users for collaborative recommendations"""
        user_clusters = self.kmeans.fit_predict(user_features)
        
        clusters = {}
        for user_id, cluster_id in enumerate(user_clusters):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(user_id)
        
        return clusters
    
    def recommend_content(self, 
                         user_id: str, 
                         user_embeddings: np.ndarray,
                         content_embeddings: np.ndarray,
                         top_k: int = 5) -> List[Dict]:
        """Generate personalized content recommendations"""
        # Find similar users
        similar_users = self._find_similar_users(user_id, user_embeddings)
        
        # Get content liked by similar users
        collaborative_recs = self._collaborative_filtering(similar_users)
        
        # Get content similar to user's history
        content_based_recs = self._content_based_filtering(user_embeddings, content_embeddings)
        
        # Hybrid recommendations
        hybrid_recs = self._hybrid_recommendations(collaborative_recs, content_based_recs)
        
        return hybrid_recs[:top_k]


    def _find_similar_users(self, user_id: str, user_embeddings: np.ndarray) -> List[str]:
        """Find similar users based on embeddings"""
        distances, indices = self.knn.fit(user_embeddings).kneighbors(user_embeddings[user_id].reshape(1, -1))
        similar_users = [i for i in indices.flatten() if i != user_id]
        return similar_users
    
    def _hybrid_recommendations(self, collaborative_recs: List[Dict], content_based_recs: List[Dict]) -> List[Dict]:
        """Combine collaborative and content-based recommendations"""
        # Simple merging strategy
        combined = {rec['id']: rec for rec in collaborative_recs}
        for rec in content_based_recs:
            if rec['id'] not in combined:
                combined[rec['id']] = rec
        
        return list(combined.values())
    
    def _collaborative_filtering(self, similar_users: List[str]) -> List[Dict]:
        """Collaborative filtering based on similar users"""
        # Placeholder for actual collaborative filtering logic
        recommendations = []
        for user in similar_users:
            # Fetch content liked by similar users (mocked)
            recommendations.append({'id': f'content_{user}', 'title': f'Content liked by user {user}'})
        
        return recommendations
    
    def _content_based_filtering(self, user_embeddings, content_embeddings):
        """Content-based filtering using cosine similarity"""
        similarities = cosine_similarity(user_embeddings.reshape(1, -1), content_embeddings)
        return similarities[0]