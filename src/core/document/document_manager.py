from src.core.document.document_loader import PDFTreeParser, TreeToModelConverter
from src.core.document.document_cacher import DocumentCacher
from src.core.document.document import Document


class DocumentManager:
    def __init__(self):
        self.cacher = DocumentCacher()
        self.parser = PDFTreeParser()
        self.tree_to_model = TreeToModelConverter()
        self.documents = {}

    def get_document(self, doc_id: int) -> Document:
        document = self.documents.get(doc_id)
        if document is not None:
            return document
        document = self.cacher.retrieve_document(doc_id)
        if document is not None:
            return document
        raise ValueError("Documnet not found!")
    
    def load_document(self, path: str) -> int:
        tree_root =  self.parser.parse(path)
        doc = self.tree_to_model.convert(tree_root)
        self.documents[doc.doc_id] = doc
        self.cacher.cache_document(doc.doc_id,doc)
        return doc.doc_id

    def has_document(self, doc_id: int) -> bool:
        return doc_id in self.documents
    

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    import os

    PATH = "data/report.pdf" # Change to your PDF path

    if os.path.exists(PATH):

        manager = DocumentManager()

        doc_id = manager.load_document(PATH)
        doc = manager.get_document(doc_id)

        print(doc)


        # # 1. Parse Structure
        # parser = PDFTreeParser()
        # raw_tree = parser.parse(PATH)

        # # 2. Convert to Model
        # converter = TreeToModelConverter()
        
        # document = converter.convert(raw_tree)

        # # 4. Print Table of Contents
        # print(f"\nTABLE OF CONTENTS for {document.get_title()}")
        # for entry in get_toc(document.sections):
        #     print(f"{'  ' * entry['level']}- {entry['title']} (p. {entry['page']})")
