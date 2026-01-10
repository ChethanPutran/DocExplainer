from src.core.document.document import Document
from src.core.knowlege_modelling.base import ConceptRelationship, GraphDelta
from typing import List, Tuple
from src.core.knowlege_modelling.base import Concept

class DocumentCache:
    """
    A placeholder class for caching documents.
    """
    def __init__(self):
        self.cache:dict[int, List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]] = {}

    def get(self, key: int) -> List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]] | None:
        return self.cache.get(key,None)

    def store(self, key : int, doc: List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]):
        self.cache[key] = doc


class DocumentCacher:
    """
    A class responsible for caching documents using DocumentCache.
    """
    def __init__(self):
        self.document_cache = DocumentCache()

    def cache_document(self, doc_id, document):
        self.document_cache.store(doc_id, document)

    def retrieve_document(self, doc_id) -> List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]] | None:
        return self.document_cache.get(doc_id)
    