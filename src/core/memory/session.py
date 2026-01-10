from typing import Dict

from core.knowlege_modelling.base import ConceptGraph
from core.knowlege_modelling.user_model import UserKnowledgeState

class SessionMemory:
    """Stores and retrieves long-term user knowledge"""
    def __init__(self) -> None:
        self.session_data = {
            "interactions": [],
            "concepts": {},
            "preferences": {}
        }

    def get_session_context(self):
        """Save concept to long-term memory"""
        return self.session_data

class SessionContext:
    """Holds session context information"""
    def __init__(self, interactions: list, concepts: Dict, preferences: Dict) -> None:
        self.interactions = interactions
        self.concepts = concepts
        self.preferences = preferences
    
class Context:
    """Holds context information for explanations"""
    def __init__(self, user_knowledge: UserKnowledgeState, session_context: SessionContext, document_context, concept_graph: ConceptGraph) -> None:
        self.user_knowledge = user_knowledge
        self.session_context = session_context
        self.document_context = document_context
        self.concept_graph = concept_graph


class SessionChain:
    """Represents a session graph for tracking user interactions"""
    def __init__(self):
        self.memory = SessionMemory()
        self.graph = {0: None}  # Placeholder for graph structure
        self.current_node = 0
        self.branch_heads = [0]
    
    def add_interaction(self,name, interaction,branch=None):
        """Add an interaction to the session graph"""
        self.memory.session_data["interactions"].append({"name": name, "interaction": interaction})
        self.update_graph(interaction,branch)
        

    def get_graph(self):
        """Retrieve the session graph"""
        return self.graph
    
    def clear_graph(self):
        """Clear the session graph"""
        self.memory.session_data["interactions"] = []
        self.graph = {0: None}
        self.current_node = 0
        self.branch_heads = [0]    

    def update_graph(self, interaction, branch=None):
        """Update the session graph with a new interaction"""
        if branch is not None:
            self.current_node = branch
    
        self.graph[self.current_node] = interaction
        self.current_node += 1      

    def get_session_context(self)->SessionContext:
        """Retrieve the current session context"""
        session_context = self.memory.get_session_context()
        return SessionContext(
            interactions=session_context["interactions"],
            concepts=session_context["concepts"],
            preferences=session_context["preferences"]
        )