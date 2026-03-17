from typing import Optional
from .base import DocumentVisualizer
from ..models.structure import Document, Section
from ..models.tree import DocumentTree, DocumentNode


class ConsolePrinter(DocumentVisualizer):
    """Print document structure to console"""
    
    def visualize_document(self, document: Document):
        """Print document to console"""
        print(f"\nDocument: {document.path}")
        print(f"Title: {document.title}")
        print("=" * 60)
        
        for section in document.sections:
            self._print_section(section, 0)
    
    def visualize_tree(self, tree: DocumentTree):
        """Print document tree to console"""
        print(f"\nDocument Tree: {tree.title}")
        print("=" * 60)
        self._print_node(tree.root)
    
    def _print_section(self, section: Section, level: int):
        """Print section recursively"""
        indent = "  " * level
        print(f"\n{indent}📂 Section: {section.title}")
        
        for para in section.paragraphs:
            print(f"{indent}  📄 {para.text[:100]}...")
        
        for sub in section.subsections:
            self._print_section(sub, level + 1)
    
    def _print_node(self, node: DocumentNode, indent: str = "", is_last: bool = True):
        """Print node recursively"""
        marker = "└── " if is_last else "├── "
        
        chunk = node.chunk
        display_text = ""
        
        if chunk.chunk_type.value == 1:  # DOCUMENT
            display_text = f"📄 DOCUMENT: {chunk.text[:100]}..."
        elif chunk.chunk_type.value == 2:  # SECTION
            display_text = f"📂 SECTION: {chunk.text[:100]}..."
        elif chunk.chunk_type.value == 3:  # PARAGRAPH
            display_text = f"📝 PARA: {chunk.text[:75]}..."
        else:
            return
        
        print(f"{indent}{marker}{display_text}")
        
        new_indent = indent + ("    " if is_last else "│   ")
        children = list(node.children.values())
        
        for i, child in enumerate(children):
            self._print_node(child, new_indent, is_last=(i == len(children) - 1))