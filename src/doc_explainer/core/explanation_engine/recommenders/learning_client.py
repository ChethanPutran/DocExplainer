# src/core/federated/learning_client.py
import torch
import torch.nn as nn
from typing import List, Dict
import copy

class FederatedLearningClient:
    """Client for federated learning of user preferences"""
    
    def __init__(self, model: nn.Module, learning_rate=0.01):
        self.model = model
        self.local_model = copy.deepcopy(model)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)
        
    def local_train(self, data_loader, epochs: int = 5) -> Dict:
        """Train locally on user data"""
        self.local_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                self.optimizer.zero_grad()
                output = self.local_model(batch['features'])
                loss = nn.CrossEntropyLoss()(output, batch['labels'])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        # Return model updates
        model_updates = {}
        for name, param in self.local_model.named_parameters():
            model_updates[name] = param.data - self.model.state_dict()[name]
        
        return {
            'updates': model_updates,
            'loss': total_loss / epochs,
            'n_samples': len(data_loader.dataset)
        }
    
    def apply_updates(self, global_updates: Dict):
        """Apply global model updates"""
        for name, param in self.model.named_parameters():
            param.data += global_updates[name]

class FederatedAggregator:
    """Aggregate updates from multiple clients"""
    
    def __init__(self):
        self.client_updates = []
        
    def aggregate_updates(self, client_responses: List[Dict]) -> Dict:
        """Federated averaging of client updates"""
        total_samples = sum(response['n_samples'] for response in client_responses)
        
        aggregated_updates = {}
        for response in client_responses:
            weight = response['n_samples'] / total_samples
            
            for name, update in response['updates'].items():
                if name not in aggregated_updates:
                    aggregated_updates[name] = update * weight
                else:
                    aggregated_updates[name] += update * weight
        
        return aggregated_updates