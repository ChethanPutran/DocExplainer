from ..base.base_viewer import BaseDocumentViewer


class DocumentViewer(BaseDocumentViewer):
    """Base document viewer with common functionality"""
    
    def load(self, path: str) -> bool:
        """Load document - to be overridden"""
        self.current_path = path
        return True
    
    def clear(self) -> bool:
        """Clear document"""
        self.current_path = ""
        self.current_page = 1
        self.current_position = 0
        return True