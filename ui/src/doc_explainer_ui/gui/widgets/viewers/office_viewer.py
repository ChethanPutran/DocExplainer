from .html_viewer import HTMLViewer
from ...utils.office_html import to_html


class OfficeViewer(HTMLViewer):
    """Word (.docx) and PowerPoint (.pptx) documents, shown as formatted text."""

    def load(self, path: str) -> bool:
        try:
            self.content = to_html(path)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Could not read this file: {e}") from e
        self.browser.setHtml(self.content)
        self.current_path = path
        self.document_loaded.emit(self.doc_id)
        return True
