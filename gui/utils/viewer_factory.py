import os
from gui.widgets.pdf_viewer import PDFViewer
from gui.widgets.document_viewer import DocumentViewer

PDF_EXT = {".pdf"}
TEXT_EXT = {".txt", ".md", ".html", ".py", ".cpp", ".json"}

def create_viewer(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in PDF_EXT:
        return PDFViewer()
    elif ext in TEXT_EXT:
        return DocumentViewer()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
