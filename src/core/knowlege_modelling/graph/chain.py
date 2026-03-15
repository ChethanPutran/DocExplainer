from typing import Dict, List

from .base import ConceptGraph,GraphDelta


class DocumentChain:
    """Represents a chain of document sections or paragraphs."""

    def __init__(self):
        self.sections = None
        self.chain: List[GraphDelta] = []
        self.ids: Dict[int, int] = {}

        # Inital graph of the starting section
        self.init_graph = ConceptGraph()

    def _append(self, section_id, delta):
        # Add the delta to the chain
        self.chain.append(delta)

        # Section id to chain index mapping
        self.ids[section_id] = len(self.chain)

    def get_concept_graph_upto(self, section_id: int):
        # Get the deltas till section id 
        if section_id == -1:
            idx = len(self.chain)
        else:
            idx = self.ids[section_id]

        return self.chain[:idx]

    def get_document_context(self, section_id) -> Dict:
        # Get the documnet context till section
        context = {"text": "", "embeddings": []}
        idx = self.ids[section_id]
        text = ""
        for delta in self.chain[:idx]:
            text += delta.data.text + "\n"
        context["text"] = text
        return context
