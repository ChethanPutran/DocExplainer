import re
from typing import Dict, Optional, Any, Tuple
from ..models import FontInfo


class StructureDetector:
    """Detects document structure elements"""
    
    def __init__(self, font_analyzer):
        self.font_analyzer = font_analyzer
        
        # Compile patterns
        self.section_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+")
        self.figure_pattern = re.compile(r"^(Figure|Fig\.)\s*\d+", re.I)
        self.table_pattern = re.compile(r"^(Table)\s*\d+", re.I)
        self.equation_pattern = re.compile(r"^\(?\d+\)?$")
        self.abstract_pattern = re.compile(r"^abstract$", re.I)
    
    def classify(self, text: str, font_info: 'FontInfo', 
                 page_num: int, bbox: Tuple) -> Optional[Dict[str, Any]]:
        """
        Classify text block type
        
        Returns:
            Dictionary with type and metadata, or None if should be skipped
        """
        text = re.sub(r"\s+", " ", text.strip())
        
        if not text:
            return None
        
        # Check for table caption
        if self.table_pattern.match(text):
            return {"type": "table_caption", "text": text}
        
        # Check for abstract
        if self.abstract_pattern.match(text):
            return {"type": "abstract", "level": 0, "text": text}
        
        # Check for figure caption
        if self.figure_pattern.match(text):
            return {"type": "figure_caption", "text": text}
        
        # Check for equation
        if self.equation_pattern.match(text):
            return {"type": "equation", "text": text}
        
        # Check for numbered section
        match = self.section_pattern.match(text)
        if match:
            section_num = match.group(1)
            return {
                "type": "section",
                "level": section_num.count("."),
                "title": re.sub(r"^\d+(?:\.\d+)*\s+", "", text),
                "text": text
            }
        
        # Check for document title
        if (page_num == 0 and
            self.font_analyzer.is_title_font(font_info.size, page_num) and
            bbox[1] < 200 and
            len(text) > 15):
            return {
                "type": "document_title",
                "title": text,
                "text": text
            }
        
        # Check for section heading by font
        if (self.font_analyzer.is_heading_font(font_info.size) and
            len(text) < 120 and
            (font_info.is_bold or text.isupper() or text.endswith(":"))):
            
            level = self.font_analyzer.get_font_level(font_info.size)
            return {
                "type": "section",
                "level": max(level, 0),
                "title": text,
                "text": text
            }
        
        # Default to paragraph
        return {"type": "paragraph", "text": text}


@dataclass
class FontInfo:
    """Font information for text span"""
    size: float
    name: str
    flags: int
    
    @property
    def is_bold(self) -> bool:
        return "Bold" in self.name or (self.flags & 16)