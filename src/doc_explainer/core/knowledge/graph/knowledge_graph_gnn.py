# src/core/gnn/knowledge_graph_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import networkx as nx

class KnowledgeGraphGNN(nn.Module):
    """Graph Neural Network for knowledge graph reasoning"""
    
    def __init__(self, num_features=128, hidden_dim=64, num_classes=10):
        super().__init__()
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.sage = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index):
        """Forward pass through GNN layers"""
        # GCN layer
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # GAT layer (attention)
        x = self.gat(x, edge_index)
        x = F.relu(x)
        
        # GraphSAGE layer
        x = self.sage(x, edge_index)
        x = F.relu(x)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    def embed_nodes(self, x, edge_index):
        """Generate node embeddings for downstream tasks"""
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gat(x, edge_index)
        x = self.sage(x, edge_index)
        return x

class KnowledgeGraphReasoner:
    """Reason over knowledge graph using GNNs"""
    
    def __init__(self, embedding_dim=128):
        self.model = KnowledgeGraphGNN(num_features=embedding_dim)
        self.node_embeddings = {}
        
    def prepare_graph_data(self, knowledge_graph: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric format"""
        # Create node features
        node_list = list(knowledge_graph.nodes())
        node_mapping = {node: idx for idx, node in enumerate(node_list)}
        
        # Edge index
        edge_index = []
        for u, v in knowledge_graph.edges():
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_index.append([node_mapping[v], node_mapping[u]])  # undirected
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Node features (use pre-trained embeddings or random)
        x = torch.randn(len(node_list), 128)
        
        return Data(x=x, edge_index=edge_index)
    
    def predict_relationship(self, concept_a: str, concept_b: str) -> float:
        """Predict if two concepts are related and strength"""
        # Get embeddings for concepts
        emb_a = self.node_embeddings.get(concept_a)
        emb_b = self.node_embeddings.get(concept_b)
        
        if emb_a is None or emb_b is None:
            return 0.0
        
        # Cosine similarity
        similarity = torch.cosine_similarity(emb_a, emb_b, dim=0)
        return float(similarity)