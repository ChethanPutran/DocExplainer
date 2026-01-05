from src.core.document.document import Document

class DocumentCache:
    """
    A placeholder class for caching documents.
    """
    def __init__(self):
        self.cache:dict[str, Document] = {}

    def get(self, key: str) -> Document:
        return self.cache.get(key)

    def store(self, doc: Document):
        self.cache[doc.doc_id] = doc

class DocumentCacher:
    """
    A class responsible for caching documents using DocumentCache.
    """
    def __init__(self):
        self.document_cache = DocumentCache()

    def cache_document(self, doc_id, document):
        self.document_cache.set(doc_id, document)

    def retrieve_document(self, doc_id):
        return self.document_cache.get(doc_id)
    