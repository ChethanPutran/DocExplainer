import json
import os
import base64
from typing import Dict, Any, List
from pathlib import Path

class DocumentHTMLGenerator:
    """Generate HTML visualization from document JSON"""
    
    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir or os.path.dirname(__file__)
        
    def generate(self, json_path: str, output_html: str = None) -> str:
        """
        Generate HTML from JSON file
        
        Args:
            json_path: Path to the JSON file
            output_html: Path to save the HTML file (optional)
        
        Returns:
            HTML content as string
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        html = self._generate_html(data)
        
        if output_html:
            with open(output_html, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"HTML saved to: {output_html}")
        
        return html
    
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML from document data"""
        
        # CSS styles
        css = """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
                padding: 20px;
            }
            
            .document-container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            
            .document-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
            }
            
            .document-title {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .document-meta {
                font-size: 0.9em;
                opacity: 0.9;
            }
            
            .toc {
                background: #f8f9fa;
                padding: 20px;
                border-bottom: 1px solid #dee2e6;
            }
            
            .toc-title {
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 10px;
                color: #495057;
            }
            
            .toc-list {
                list-style: none;
                padding-left: 20px;
            }
            
            .toc-item {
                margin: 5px 0;
                cursor: pointer;
                color: #007bff;
            }
            
            .toc-item:hover {
                text-decoration: underline;
            }
            
            .section {
                padding: 20px;
                border-bottom: 1px solid #e9ecef;
            }
            
            .section-header {
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #dee2e6;
            }
            
            .section-title {
                font-size: 1.8em;
                color: #2c3e50;
            }
            
            .section-meta {
                font-size: 0.9em;
                color: #6c757d;
                margin-top: 5px;
            }
            
            .subsection {
                margin-left: 30px;
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            
            .subsection-title {
                font-size: 1.4em;
                color: #34495e;
                margin-bottom: 15px;
            }
            
            .content-block {
                margin: 15px 0;
                padding: 15px;
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            
            .content-block:hover {
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .paragraph {
                background-color: #ffffff;
                border-left: 4px solid #007bff;
            }
            
            .paragraph-text {
                font-size: 1em;
                margin-bottom: 10px;
            }
            
            .sentence-list {
                margin-top: 10px;
                padding-left: 20px;
                font-size: 0.95em;
                color: #6c757d;
            }
            
            .sentence-item {
                margin: 3px 0;
                list-style-type: circle;
            }
            
            .image-container {
                background-color: #f8f9fa;
                border: 2px dashed #dee2e6;
                text-align: center;
            }
            
            .image-content {
                max-width: 100%;
                margin: 10px 0;
            }
            
            .image-content img {
                max-width: 100%;
                max-height: 400px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            
            .image-caption {
                font-style: italic;
                color: #6c757d;
                margin: 10px 0;
                padding: 10px;
                background-color: #e9ecef;
                border-radius: 5px;
            }
            
            .table-container {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                overflow-x: auto;
            }
            
            .table-content {
                width: 100%;
                border-collapse: collapse;
            }
            
            .table-content th {
                background-color: #007bff;
                color: white;
                padding: 10px;
                font-weight: bold;
            }
            
            .table-content td {
                padding: 8px;
                border: 1px solid #dee2e6;
            }
            
            .table-content tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            
            .table-caption {
                padding: 10px;
                background-color: #e9ecef;
                font-style: italic;
                margin-top: 10px;
            }
            
            .equation-container {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                font-family: 'Courier New', monospace;
            }
            
            .equation-text {
                font-size: 1.1em;
                padding: 15px;
                background-color: #272822;
                color: #f8f8f2;
                border-radius: 5px;
                overflow-x: auto;
            }
            
            .equation-caption {
                padding: 10px;
                background-color: #e9ecef;
                font-style: italic;
            }
            
            .badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 0.8em;
                font-weight: bold;
                margin-right: 10px;
            }
            
            .badge-primary { background-color: #007bff; color: white; }
            .badge-success { background-color: #28a745; color: white; }
            .badge-info { background-color: #17a2b8; color: white; }
            .badge-warning { background-color: #ffc107; color: black; }
            
            .page-info {
                font-size: 0.85em;
                color: #6c757d;
                margin-bottom: 5px;
            }
            
            .bbox-info {
                font-size: 0.8em;
                color: #adb5bd;
                font-family: monospace;
            }
            
            .statistics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                padding: 20px;
                background-color: #e9ecef;
            }
            
            .stat-card {
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }
            
            .stat-label {
                color: #6c757d;
                margin-top: 5px;
            }
            
            .navigation {
                position: sticky;
                top: 0;
                background-color: white;
                z-index: 1000;
                padding: 10px 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                display: flex;
                gap: 10px;
            }
            
            .nav-button {
                padding: 8px 15px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 0.9em;
            }
            
            .nav-button:hover {
                background-color: #0056b3;
            }
            
            .search-box {
                flex-grow: 1;
                padding: 8px;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                font-size: 0.9em;
            }
            
            .highlight {
                background-color: #fff3cd;
                padding: 2px;
            }
            
            .missing-image {
                padding: 20px;
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
            }
            
            @media print {
                .navigation, .toc, .statistics {
                    display: none;
                }
            }
        </style>
        """
        
        # JavaScript for interactivity
        js = """
        <script>
            function searchContent() {
                const searchTerm = document.getElementById('searchInput').value.toLowerCase();
                const content = document.querySelectorAll('.paragraph-text, .image-caption, .table-caption, .equation-caption');
                
                content.forEach(element => {
                    const text = element.textContent.toLowerCase();
                    if (text.includes(searchTerm) && searchTerm.length > 0) {
                        element.innerHTML = element.textContent.replace(
                            new RegExp(searchTerm, 'gi'),
                            match => `<span class="highlight">${match}</span>`
                        );
                        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                });
            }
            
            function scrollToSection(sectionId) {
                const element = document.getElementById(sectionId);
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth' });
                }
            }
            
            function toggleAllSections() {
                const sections = document.querySelectorAll('.subsection');
                sections.forEach(section => {
                    if (section.style.display === 'none') {
                        section.style.display = 'block';
                    } else {
                        section.style.display = 'none';
                    }
                });
            }
        </script>
        """
        
        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document Visualization: {data.get('title', 'Untitled')}</title>
            {css}
            {js}
        </head>
        <body>
            <div class="navigation">
                <button class="nav-button" onclick="toggleAllSections()">Toggle Subsections</button>
                <input type="text" id="searchInput" class="search-box" placeholder="Search in document...">
                <button class="nav-button" onclick="searchContent()">Search</button>
            </div>
            
            <div class="document-container">
                {self._render_header(data)}
                {self._render_statistics(data)}
                {self._render_toc(data)}
                {self._render_sections(data.get('sections', []))}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _render_header(self, data: Dict[str, Any]) -> str:
        """Render document header"""
        return f"""
        <div class="document-header">
            <h1 class="document-title">{data.get('title', 'Untitled Document')}</h1>
            <div class="document-meta">
                <div>Document ID: {data.get('doc_id', 'N/A')}</div>
                <div>Path: {data.get('path', 'N/A')}</div>
            </div>
        </div>
        """
    
    def _render_statistics(self, data: Dict[str, Any]) -> str:
        """Render document statistics"""
        def count_elements(sections):
            stats = {'paragraphs': 0, 'images': 0, 'tables': 0, 'equations': 0}
            for section in sections:
                stats['paragraphs'] += len(section.get('paragraphs', []))
                stats['images'] += len(section.get('images', []))
                stats['tables'] += len(section.get('tables', []))
                stats['equations'] += len(section.get('equations', []))
                if 'subsections' in section:
                    sub_stats = count_elements(section['subsections'])
                    for k in stats:
                        stats[k] += sub_stats[k]
            return stats
        
        stats = count_elements(data.get('sections', []))
        
        return f"""
        <div class="statistics">
            <div class="stat-card">
                <div class="stat-number">{stats['paragraphs']}</div>
                <div class="stat-label">Paragraphs</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['images']}</div>
                <div class="stat-label">Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['tables']}</div>
                <div class="stat-label">Tables</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['equations']}</div>
                <div class="stat-label">Equations</div>
            </div>
        </div>
        """
    
    def _render_toc(self, data: Dict[str, Any]) -> str:
        """Render table of contents"""
        def render_toc_items(sections, level=0):
            items = []
            for i, section in enumerate(sections):
                section_id = f"section-{section.get('sec_id', i)}"
                items.append(f"""
                    <li class="toc-item" style="margin-left: {level*20}px" 
                        onclick="scrollToSection('{section_id}')">
                        {section.get('title', f'Section {i+1}')}
                    </li>
                """)
                if section.get('subsections'):
                    items.extend(render_toc_items(section['subsections'], level + 1))
            return items
        
        toc_items = render_toc_items(data.get('sections', []))
        if not toc_items:
            return ""
            
        return f"""
        <div class="toc">
            <div class="toc-title">Table of Contents</div>
            <ul class="toc-list">
                {''.join(toc_items)}
            </ul>
        </div>
        """
    
    def _render_sections(self, sections: List[Dict[str, Any]], level: int = 1) -> str:
        """Render sections recursively"""
        html = []
        for i, section in enumerate(sections):
            section_id = f"section-{section.get('sec_id', i)}"
            
            html.append(f"""
            <div class="section" id="{section_id}">
                <div class="section-header">
                    <h{level+1} class="section-title">{section.get('title', f'Section {i+1}')}</h{level+1}>
                    <div class="section-meta">
                        <span class="badge badge-primary">Page {section.get('page_start', 1)}</span>
                        <span class="badge badge-info">{len(section.get('paragraphs', []))} paragraphs</span>
                        <span class="badge badge-success">{len(section.get('images', []))} images</span>
                        <span class="badge badge-warning">{len(section.get('tables', []))} tables</span>
                    </div>
                </div>
                
                {self._render_paragraphs(section.get('paragraphs', []))}
                {self._render_images(section.get('images', []))}
                {self._render_tables(section.get('tables', []))}
                {self._render_equations(section.get('equations', []))}
                
                {self._render_sections(section.get('subsections', []), level + 1)}
            </div>
            """)
        
        return '\n'.join(html)
    
    def _render_paragraphs(self, paragraphs: List[Dict[str, Any]]) -> str:
        """Render paragraphs"""
        html = []
        for para in paragraphs:
            html.append(f"""
            <div class="content-block paragraph">
                <div class="page-info">
                    <span class="badge badge-primary">Page {para.get('page', 1)}</span>
                    <span class="badge badge-info">ID: {para.get('para_id', 'N/A')}</span>
                </div>
                <div class="paragraph-text">{para.get('raw_text', '')}</div>
                <div class="sentence-list">
                    <strong>Sentences ({len(para.get('sentences', []))}):</strong>
                    <ul>
                        {''.join([f'<li class="sentence-item">{s.get("raw_text", "")}</li>' 
                                 for s in para.get('sentences', [])])}
                    </ul>
                </div>
                {self._render_bbox(para.get('bbox', []))}
            </div>
            """)
        return '\n'.join(html)
    
    def _render_images(self, images: List[Dict[str, Any]]) -> str:
        """Render images"""
        html = []
        for img in images:
            img_path = img.get('raw_img', '')
            img_exists = os.path.exists(img_path) if img_path else False
            
            caption_text = ''
            if img.get('caption'):
                caption_text = ' '.join([s.get('raw_text', '') for s in img['caption']])
            
            html.append(f"""
            <div class="content-block image-container">
                <div class="page-info">
                    <span class="badge badge-success">Image</span>
                    <span class="badge badge-primary">Page {img.get('page', 1)}</span>
                    <span class="badge badge-info">ID: {img.get('img_id', 'N/A')}</span>
                </div>
                <div class="image-content">
                    {self._render_image_element(img_path, img_exists)}
                </div>
                {self._render_caption(caption_text, 'Image Caption') if caption_text else ''}
                {self._render_bbox(img.get('bbox', []))}
            </div>
            """)
        return '\n'.join(html)
    
    def _render_image_element(self, img_path: str, exists: bool) -> str:
        """Render image element with fallback"""
        if not exists:
            return f'<div class="missing-image">Image not found: {img_path}</div>'
        
        # Try to embed image as base64 for standalone HTML
        try:
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
                ext = os.path.splitext(img_path)[1][1:] or 'png'
                return f'<img src="data:image/{ext};base64,{img_data}" alt="Document Image">'
        except:
            # Fallback to file path reference
            return f'<img src="file://{os.path.abspath(img_path)}" alt="Document Image">'
    
    def _render_tables(self, tables: List[Dict[str, Any]]) -> str:
        """Render tables"""
        html = []
        for table in tables:
            caption_text = ''
            if table.get('caption'):
                caption_text = ' '.join([s.get('raw_text', '') for s in table['caption']])
            
            # Parse table data if it's a string
            table_data = table.get('data', '')
            if isinstance(table_data, str):
                try:
                    import ast
                    table_data = ast.literal_eval(table_data)
                except:
                    table_data = [row.split(' | ') for row in table_data.split('\n') if row.strip()]
            
            html.append(f"""
            <div class="content-block table-container">
                <div class="page-info">
                    <span class="badge badge-warning">Table</span>
                    <span class="badge badge-primary">Page {table.get('page', 1)}</span>
                    <span class="badge badge-info">ID: {table.get('table_id', 'N/A')}</span>
                </div>
                {self._render_table_data(table_data)}
                {self._render_caption(caption_text, 'Table Caption') if caption_text else ''}
                {self._render_bbox(table.get('bbox', []))}
            </div>
            """)
        return '\n'.join(html)
    
    def _render_table_data(self, data) -> str:
        """Render table data as HTML table"""
        if not data:
            return '<div>No table data available</div>'
        
        if isinstance(data, list) and len(data) > 0:
            html = ['<table class="table-content">']
            
            # Header row
            if len(data[0]) > 0:
                html.append('<thead><tr>')
                for cell in data[0]:
                    html.append(f'<th>{cell}</th>')
                html.append('</tr></thead>')
            
            # Body rows
            html.append('<tbody>')
            for row in data[1:]:
                html.append('<tr>')
                for cell in row:
                    html.append(f'<td>{cell}</td>')
                html.append('</tr>')
            html.append('</tbody>')
            html.append('</table>')
            return '\n'.join(html)
        
        return f'<pre>{data}</pre>'
    
    def _render_equations(self, equations: List[Dict[str, Any]]) -> str:
        """Render equations"""
        html = []
        for eq in equations:
            caption_text = ''
            if eq.get('caption'):
                caption_text = ' '.join([s.get('raw_text', '') for s in eq['caption']])
            
            html.append(f"""
            <div class="content-block equation-container">
                <div class="page-info">
                    <span class="badge badge-info">Equation</span>
                    <span class="badge badge-primary">Page {eq.get('page', 1)}</span>
                    <span class="badge badge-info">ID: {eq.get('equation_id', 'N/A')}</span>
                </div>
                <div class="equation-text">{eq.get('raw_text', '')}</div>
                {self._render_caption(caption_text, 'Equation Caption') if caption_text else ''}
                {self._render_bbox(eq.get('bbox', []))}
            </div>
            """)
        return '\n'.join(html)
    
    def _render_caption(self, caption: str, caption_type: str) -> str:
        """Render caption"""
        return f"""
        <div class="image-caption">
            <strong>{caption_type}:</strong> {caption}
        </div>
        """
    
    def _render_bbox(self, bbox: List[float]) -> str:
        """Render bounding box info"""
        if not bbox or len(bbox) != 4:
            return ""
        return f"""
        <div class="bbox-info">
            BBox: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})
        </div>
        """

# Utility function for easy use
def generate_document_html(json_path: str, output_html: str = None) -> str:
    """Generate HTML from document JSON"""
    generator = DocumentHTMLGenerator()
    return generator.generate(json_path, output_html)

