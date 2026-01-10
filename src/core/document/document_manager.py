import uuid
from src.core.document.document_processing import parse_document
from src.core.document.document_cacher import DocumentCache
from src.core.document.document import Document

class DocumentManager:
    def __init__(self):
        self.cache = DocumentCache()

    def get_document(self, doc_id: str) -> Document:
        return self.cache.get(doc_id)
    
    def load_document(self, path: str) -> int:
        text, sections = parse_document(path)

        doc = Document(
            doc_id=str(uuid.uuid4()),
            path=path,
            raw_text=text,
            sections=sections,
            embeddings={}
        )

        # self.cache.store(doc)
        return doc.doc_id
