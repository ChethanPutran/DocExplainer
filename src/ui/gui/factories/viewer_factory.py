import os
from typing import Optional
from ..base.exceptions import UnsupportedFileTypeError
from ..widgets.viewers import PDFViewer, TextViewer,  HTMLViewer, DocumentViewer
from ..models.signals import UISignals


class ViewerFactory:
    """Factory for creating document viewers"""
    
    _viewers = {
        '.pdf': PDFViewer,
        '.txt': TextViewer,
        '.text': TextViewer,
        '.html': HTMLViewer,
        '.htm': HTMLViewer,
    }
    
    @classmethod
    def register_viewer(cls, extension: str, viewer_class):
        """Register a new viewer for an extension"""
        cls._viewers[extension.lower()] = viewer_class
    
    @classmethod
    def create_viewer(cls, path: str, signals: UISignals = None) -> DocumentViewer:
        """Create appropriate viewer for file"""
        ext = os.path.splitext(path)[1].lower()
        
        viewer_class = cls._viewers.get(ext)
        if viewer_class:
            return viewer_class(signals=signals)
        
        raise UnsupportedFileTypeError(f"No viewer available for {ext} files")
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """Get list of supported extensions"""
        return list(cls._viewers.keys())