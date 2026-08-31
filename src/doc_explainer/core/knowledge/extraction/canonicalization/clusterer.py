from typing import List, Dict, DefaultDict
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ConceptClusterer:
    """Cluster similar concepts using embeddings"""
    
    def __init__(self, embedder, threshold: float = 0.85):
        self.embedder = embedder
        self.threshold = threshold
        self.embedding_cache = {}

    def _encode_batch(self, concepts: List[str]) -> np.ndarray:
        """Encode concepts as a 2D matrix: one row per concept."""
        embeddings = np.asarray(self.embedder.encode(concepts), dtype=np.float32)

        if embeddings.ndim == 1:
            if len(concepts) == 1:
                return embeddings.reshape(1, -1)
            embeddings = np.asarray(
                [self.embedder.encode(concept) for concept in concepts],
                dtype=np.float32,
            )

        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        return embeddings
    
    def _get_embeddings(self, concepts: List[str]) -> np.ndarray:
        """Get embeddings for concepts, using cache when possible"""
        to_encode = []
        indices = []
        
        for i, concept in enumerate(concepts):
            if concept in self.embedding_cache:
                continue
            to_encode.append(concept)
            indices.append(i)
        
        if to_encode:
            new_embeddings = self._encode_batch(to_encode)
            for idx, concept, emb in zip(indices, to_encode, new_embeddings):
                self.embedding_cache[concept] = emb
        
        # Build full embeddings array
        embeddings = []
        for concept in concepts:
            embeddings.append(self.embedding_cache[concept])
        
        return np.asarray(embeddings, dtype=np.float32).reshape(len(concepts), -1)
    
    def cluster(self, concepts: List[str]) -> Dict[int, List[str]]:
        """Cluster similar concepts together"""
        if len(concepts) <= 1:
            return {0: concepts}

        embeddings = self._get_embeddings(concepts)
        similarity_matrix = cosine_similarity(embeddings)

        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - self.threshold,
            n_clusters=None,
        )

        distance_matrix = 1 - similarity_matrix
        labels = clustering.fit_predict(distance_matrix)

        clusters = defaultdict(list)
        for concept, label in zip(concepts, labels):
            clusters[label].append(concept)

        return clusters
    
    def get_canonical_map(self, clusters: Dict[int, List[str]]) -> Dict[str, List[str]]:
        """Convert clusters to canonical mapping (shortest name as canonical)"""
        canonical_map = {}
        for cluster in clusters.values():
            canonical = min(cluster, key=len)
            canonical_map[canonical] = cluster
        return canonical_map
