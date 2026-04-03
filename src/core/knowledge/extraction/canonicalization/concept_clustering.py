# src/core/knowledge/concept_clustering.py
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import networkx as nx
from typing import List, Dict, Any

class ConceptClusterer:
    """Cluster related concepts for better knowledge organization"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
        self.dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        
    def cluster_concepts(self, concepts: List['Concept']) -> Dict[int, List[str]]:
        """Group related concepts using multiple clustering algorithms"""
        # Extract concept embeddings
        concept_texts = [c.name + " " + " ".join(c.definitions) for c in concepts]
        
        # Create feature matrix
        X = self.tfidf.fit_transform(concept_texts)
        
        # Hierarchical clustering
        hierarchical_labels = self.clusterer.fit_predict(X.toarray())
        
        # DBSCAN for density-based clustering
        dbscan_labels = self.dbscan.fit_predict(X.toarray())
        
        # Combine results
        clusters = self._combine_clustering_results(concepts, hierarchical_labels, dbscan_labels)
        
        return clusters
    
    def find_concept_hierarchy(self, concepts: List['Concept']) -> nx.DiGraph:
        """Build hierarchical relationships between concepts"""
        G = nx.DiGraph()
        
        for concept in concepts:
            G.add_node(concept.id, name=concept.name, level=concept.difficulty_level)
        
        # Add edges based on relationships
        for concept in concepts:
            for related in concept.related_concepts:
                if related.weight > 0.7:  # Strong relationship
                    G.add_edge(concept.id, related.id, weight=related.weight)
        
        return G
    
    def visualize_clusters_tsne(self, concepts: List['Concept'], output_path: str):
        """Visualize concept clusters using t-SNE"""
        concept_texts = [c.name for c in concepts]
        X = self.tfidf.fit_transform(concept_texts)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X.toarray())
        
        # Save visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        
        for i, concept in enumerate(concepts):
            plt.annotate(concept.name, (X_tsne[i, 0], X_tsne[i, 1]))
        
        plt.savefig(output_path)
        plt.close()