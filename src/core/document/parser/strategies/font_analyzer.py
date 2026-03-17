from typing import List, Dict, Tuple
from collections import Counter
import fitz


class FontAnalyzer:
    """Analyzes font hierarchy in PDF documents"""
    
    def __init__(self):
        self.font_sizes: List[float] = []
        self.font_levels: List[float] = []
        self.body_font: float = 0.0
        self.font_frequency: Dict[float, int] = {}
    
    def analyze(self, doc: fitz.Document) -> Tuple[List[float], float]:
        """
        Analyze font hierarchy in document
        
        Returns:
            Tuple of (font_levels, body_font)
        """
        sizes = []
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        sizes.append(round(span["size"], 1))
        
        # Get unique sizes sorted descending
        self.font_sizes = sorted(set(sizes), reverse=True)
        self.font_levels = self.font_sizes
        
        # Find most frequent font (body text)
        self.font_frequency = Counter(sizes)
        if self.font_frequency:
            self.body_font = max(self.font_frequency, key=self.font_frequency.get)
        
        return self.font_levels, self.body_font
    
    def get_font_level(self, size: float) -> int:
        """Get hierarchical level of font size"""
        if size not in self.font_sizes:
            return -1
        
        idx = self.font_sizes.index(size)
        if idx == 0:
            return -1  # Document title
        
        return idx - 1
    
    def is_title_font(self, size: float, page_num: int = 0) -> bool:
        """Check if font size indicates title"""
        return size > self.body_font and page_num == 0
    
    def is_heading_font(self, size: float) -> bool:
        """Check if font size indicates heading"""
        return size > self.body_font