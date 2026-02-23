from typing import Any

class DocumentCache:
    """
    A placeholder class for caching documents.
    """
    def __init__(self):
        self.cache:dict[Any, Any] = {}

    def get(self, key: Any):
        return self.cache.get(key,None)

    def store(self, key : Any, doc: Any):
        self.cache[key] = doc


class DocumentCacher:
    """
    A class responsible for caching documents using DocumentCache.
    """
    def __init__(self):
        self.document_cache = DocumentCache()

    def cache_document(self, doc_id, document):
        self.document_cache.store(doc_id, document)

    def retrieve_document(self, doc_id):
        return self.document_cache.get(doc_id)
    
