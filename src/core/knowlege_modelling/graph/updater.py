from src.core.knowlege_modelling.base import ConceptGraph, ConceptNodeRelationship, GraphDelta
from src.core.knowlege_modelling.user_model import UserKnowledgeState


class GraphUpdater:
    def __init__(self, base_graph: ConceptGraph, user_state: UserKnowledgeState):
        self.graph = base_graph
        self.user = user_state

    def apply_delta(self, delta: GraphDelta):
        # 1. Merge new concepts into the persistent graph
        for name, node in delta.new_concepts.items():
            if not self.graph.has_concept(name):
                self.graph.add_concept(node)

        # 2. Apply edges (ensuring they link to graph-resident nodes)
        for edge in delta.new_edges:
            # Resolve nodes from the graph to ensure pointer consistency
            u = self.graph.get_concept(edge.concept1.primary_concept.name)
            v = self.graph.get_concept(edge.concept2.primary_concept.name)

            if u and v:
                edge.concept1, edge.concept2 = u, v
                edge.relationship.attributes["subjective_weight"] = self.compute_subjective_weight(
                    edge
                )
                self.graph.add_relationship(u, v, edge)

    def compute_subjective_weight(self, edge: ConceptNodeRelationship):
        """
        Adjusts edge strength based on user's current confidence.
        If a user is confused about 'A', the link 'A -> B' is considered 'weak' or 'noisy'.
        """
        # Get confidence scores (default to 0.1 for brand new concepts)
        name_u = edge.concept1.primary_concept.name
        name_v = edge.concept2.primary_concept.name

        conf_u = self.user.confidence.get(name_u, 0.1)
        conf_v = self.user.confidence.get(name_v, 0.1)

        # Base strength from text analysis
        base_strength = edge.relationship.attributes.get("weight", 0.5)

        # Subjective weight = Base * (Average Confidence)
        return base_strength * ((conf_u + conf_v) / 2)

    def compute_weight(self, edge: ConceptNodeRelationship):
        cu = self.user.confidence.get(edge.concept1.primary_concept.name, 0.5)
        cv = self.user.confidence.get(edge.concept2.primary_concept.name, 0.5)
        return edge.relationship.strength * min(cu, cv)
