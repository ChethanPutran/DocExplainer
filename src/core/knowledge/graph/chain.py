from typing import Dict, List, Optional
from src.core.knowledge.models.graph import ConceptGraph
from src.core.knowledge.models.delta import GraphDelta

class BaseDocumentChain:
    """Base interface for document chain"""
    
    def add(self, section_id: int, delta: GraphDelta):
        """Add a delta to the chain"""
        raise NotImplementedError
    
    def get_concept_graph_upto(self, section_id: int) -> List[GraphDelta]:
        """Get deltas up to section_id"""
        raise NotImplementedError
    
    def get_document_context(self, section_id: int) -> Dict:
        """Get document context up to section_id"""
        raise NotImplementedError

class DocumentChain(BaseDocumentChain):
    """Represents a chain of document sections or paragraphs"""
    
    def __init__(self):
        self.chain: List[GraphDelta] = []
        self.ids: Dict[int, int] = {}
        self.init_graph = ConceptGraph()

    def add(self, section_id: int, delta: GraphDelta):
        """Add a delta to the chain"""
        self.chain.append(delta)
        self.ids[section_id] = len(self.chain)

    def get_concept_graph_upto(self, section_id: int) -> List[GraphDelta]:
        """Get deltas up to section id"""
        if section_id == -1:
            idx = len(self.chain)
        else:
            idx = self.ids.get(section_id, 0)

        return self.chain[:idx]

    def get_document_context(self, section_id: int) -> Dict:
        """Get document context up to section"""
        context = {"text": "", "embeddings": []}
        idx = self.ids.get(section_id, 0)
        text = ""
        for delta in self.chain[:idx]:
            text += delta.data.text + "\n"
        context["text"] = text
        return context