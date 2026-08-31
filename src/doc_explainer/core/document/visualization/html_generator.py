import json
import os
import base64
from typing import Dict, Any, List, Optional
from .base import DocumentVisualizer
from ..models.structure import Document
from ..models.tree import DocumentTree


class HTMLGenerator(DocumentVisualizer):
    """Generate HTML visualization from document"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or os.path.dirname(__file__)
    
    def visualize_document(self, document: Document):
        """Visualize document - not implemented for HTML generator"""
        raise NotImplementedError("Use generate_from_document or generate_from_json instead")
    
    def visualize_tree(self, tree: DocumentTree):
        """Visualize tree - not implemented for HTML generator"""
        raise NotImplementedError("Use generate_from_document or generate_from_json instead")
    
    def generate_from_document(self, document: Document, output_path: str) -> str:
        """Generate HTML from document object"""
        html = self._generate_html(document.to_dict())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def generate_from_json(self, json_path: str, output_path: Optional[str] = None) -> str:
        """Generate HTML from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        html = self._generate_html(data)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"HTML saved to: {output_path}")
        
        return html
    
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML from document data"""
        css = self._get_css()
        js = self._get_javascript()
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document: {data.get('title', 'Untitled')}</title>
    {css}
    {js}
</head>
<body>
    <div class="navigation">
        <button class="nav-button" onclick="toggleSections()">Toggle Subsections</button>
        <input type="text" id="searchInput" class="search-box" placeholder="Search...">
        <button class="nav-button" onclick="searchContent()">Search</button>
    </div>
    
    <div class="document-container">
        {self._render_header(data)}
        {self._render_statistics(data)}
        {self._render_toc(data)}
        {self._render_sections(data.get('sections', []))}
    </div>
</body>
</html>"""
    
    def _get_css(self) -> str:
        """Get CSS styles"""
        return """<style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                   line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px; }
            .document-container { max-width: 1200px; margin: 0 auto; background: white;
                                 box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; }
            .document-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                              color: white; padding: 30px; }
            .document-title { font-size: 2.5em; margin-bottom: 10px; }
            .section { padding: 20px; border-bottom: 1px solid #e9ecef; }
            .section-title { font-size: 1.8em; color: #2c3e50; }
            .paragraph { background: #f8f9fa; padding: 15px; margin: 10px 0;
                        border-left: 4px solid #007bff; }
            .navigation { position: sticky; top: 0; background: white; padding: 10px 20px;
                         box-shadow: 0 2px 5px rgba(0,0,0,0.1); display: flex; gap: 10px; }
            .nav-button { padding: 8px 15px; background: #007bff; color: white;
                         border: none; border-radius: 5px; cursor: pointer; }
            .search-box { flex-grow: 1; padding: 8px; border: 1px solid #dee2e6;
                         border-radius: 5px; }
            .highlight { background-color: #fff3cd; }
        </style>"""
    
    def _get_javascript(self) -> str:
        """Get JavaScript"""
        return """<script>
            function searchContent() {
                const term = document.getElementById('searchInput').value.toLowerCase();
                const elements = document.querySelectorAll('.paragraph');
                elements.forEach(el => {
                    if (el.textContent.toLowerCase().includes(term) && term.length > 0) {
                        el.style.backgroundColor = '#fff3cd';
                        el.scrollIntoView({ behavior: 'smooth' });
                    } else {
                        el.style.backgroundColor = '';
                    }
                });
            }
            function toggleSections() {
                document.querySelectorAll('.subsection').forEach(el => {
                    el.style.display = el.style.display === 'none' ? 'block' : 'none';
                });
            }
        </script>"""
    
    def _render_header(self, data: Dict[str, Any]) -> str:
        """Render document header"""
        return f"""
        <div class="document-header">
            <h1 class="document-title">{data.get('title', 'Untitled')}</h1>
            <div class="document-meta">ID: {data.get('document_id', 'N/A')}</div>
        </div>"""
    
    def _render_statistics(self, data: Dict[str, Any]) -> str:
        """Render statistics"""
        return ""  # Simplified for brevity
    
    def _render_toc(self, data: Dict[str, Any]) -> str:
        """Render table of contents"""
        return ""  # Simplified for brevity
    
    def _render_sections(self, sections: List, level: int = 1) -> str:
        """Render sections recursively"""
        html = []
        for i, section in enumerate(sections):
            html.append(f"""
            <div class="section">
                <h{level+1} class="section-title">{section.get('title', f'Section {i+1}')}</h{level+1}>
                {self._render_paragraphs(section.get('paragraphs', []))}
                {self._render_sections(section.get('subsections', []), level + 1)}
            </div>
            """)
        return '\n'.join(html)
    
    def _render_paragraphs(self, paragraphs: List) -> str:
        """Render paragraphs"""
        html = []
        for para in paragraphs:
            html.append(f'<div class="paragraph">{para.get("text", "")}</div>')
        return '\n'.join(html)