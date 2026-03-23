from src.core.user import UserManager, UserKnowledgeState
from ..repository import BaseKnowledgeStore
from ..models import ConceptGraph, GraphDelta, ConceptNodeRelationship

class GraphUpdater:
    """Updates the knowledge graph with new information"""
    
    def __init__(self, user_manager: UserManager, knowledge_store: BaseKnowledgeStore):
        self.user_manager = user_manager
        self.knowledge_store = knowledge_store
        self.user_state:UserKnowledgeState = None

    def apply_delta(self, delta: GraphDelta, target_graph: ConceptGraph):
        """Apply delta to target graph"""
        # Add new concepts
        for name, node in delta.new_concepts.items():
            if not self.knowledge_store.get_concept_by_name(name):
                self.knowledge_store.save_concept(node)
                target_graph.add_concept_node(node)

        # Add new relationships
        for edge in delta.new_edges:
            u = self.knowledge_store.get_concept_by_name(edge.concept1.primary_concept.name)
            v = self.knowledge_store.get_concept_by_name(edge.concept2.primary_concept.name)

            if not (u and v):
                print(f"Warning: Skipping edge - nodes not found in store.")
                continue

            # Re-assign to persistent instances
            edge.concept1, edge.concept2 = u, v
            
            # Update weight based on user knowledge
            edge.relationship.strength = self.compute_subjective_weight(edge)
            
            self.knowledge_store.save_relationship(edge)
            target_graph.add_relationship(u, v, edge)

        # Update existing relationships
        for (u_name, v_name), edge in delta.edge_updates.items():
            if target_graph.graph.has_edge(u_name, v_name):
                target_graph.update_relationship(
                    edge.concept1, edge.concept2, edge
                )

    def compute_subjective_weight(self, edge: ConceptNodeRelationship) -> float:
        """Compute weight adjusted by user knowledge"""
        base_strength = edge.relationship.strength
        
        conf_u = self.user_manager.user_confidence(edge.concept1.primary_concept.name)
        conf_v = self.user_manager.user_confidence(edge.concept2.primary_concept.name)
        
        # Penalize edges where user knowledge is low
        return base_strength * ((conf_u + conf_v) / 2)