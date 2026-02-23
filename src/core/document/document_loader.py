from __future__ import annotations

import os
import re
import uuid
import fitz  
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.document.document import Document, Section,Paragraphs,Sentence,Image

from typing import Dict, List, Tuple, Any, Optional

# ==========================================
# PDF TREE PARSER (Structural Extraction)
# ==========================================

class PDFTreeParser:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def is_heading(self, span: Dict) -> bool:
        text = span["text"].strip()
        numbered_pattern = r"^(?:\d+(?:\.\d+)*|Section|Chapter|Part)\s+.*"
        return (span["size"] > 12) or re.match(numbered_pattern, text, re.I)

    def is_minor_title(self, text: str, span: Dict) -> bool:
        clean_text = text.strip()
        word_count = len(clean_text.split())
        is_bold = span.get("flags", 0) & 2 
        return 1 <= word_count <= 8 and is_bold

    def parse(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        root = {"title": "Full Document", "level": 100, "content": [], "children": [], "page": 1,"path":pdf_path}
        stack = [root]
        global_cursor = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            curr_p = page_num + 1
            blocks = page.get_text("dict")["blocks"]
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

            # Handle page images
            page_images = []
            for img in page.get_image_info():
                page_images.append({"type": "image", "bbox": img["bbox"], "xref": img["xref"], "page": curr_p})

            for b in blocks:
                if b["type"] == 0:  # Text block
                    spans = [s for l in b["lines"] for s in l["spans"]]
                    block_text = " ".join([s["text"] for s in spans]).strip()
                    
                    if not block_text or (len(block_text.split()) < 3 and not self.is_heading(spans[0])):
                        continue

                    meta = {
                        "text": block_text, "start": global_cursor, 
                        "end": global_cursor + len(block_text), 
                        "page": curr_p, "bbox": b["bbox"]
                    }

                    if self.is_heading(spans[0]):
                        new_node = {"title": block_text, "level": spans[0]["size"], 
                                    "content": [], "children": [], **meta}
                        while len(stack) > 1 and stack[-1]["level"] <= new_node["level"]:
                            stack.pop()
                        stack[-1]["children"].append(new_node)
                        stack.append(new_node)
                    elif self.is_minor_title(block_text, spans[0]):
                        stack[-1]["content"].append({"type": "sub_title", **meta})
                    else:
                        stack[-1]["content"].append({"type": "paragraph", **meta})
                    
                    global_cursor += len(block_text) + 1
        return root

# ==========================================
# MODEL CONVERTER (Recursive Transformation)
# ==========================================

class TreeToModelConverter:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading SpaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _to_sentences(self, text: str, p_start: int, p_page: int, p_bbox: Tuple) -> List[Sentence]:
        doc = self.nlp(text)
        return [Sentence(
            sen_id=str(uuid.uuid4())[:8],
            raw_text=s.text.strip(),
            start=p_start + s.start_char,
            end=p_start + s.end_char,
            page=p_page,
            bbox=p_bbox
        ) for s in doc.sents if len(s.text.split()) >= 2]

    def convert(self, raw_tree):
        model_sections = [self.convert_node(s) for s in raw_tree["children"]]
        
        # Create Document
        document = Document(
            doc_id=int(uuid.uuid4()),
            path=raw_tree['path'],
            raw_text="\n".join([s.raw_text for s in model_sections]),
            sections=model_sections
        )
        return document
    
    def convert_node(self, node_dict: Dict) -> Section:
        paras, imgs, full_text = [], [], []
        
        for item in node_dict.get("content", []):
            if item.get("type") == "paragraph" or item.get("type") == "sub_title":
                p_text = item["text"]
                full_text.append(p_text)
                sents = self._to_sentences(p_text, item["start"], item["page"], item["bbox"])
                paras.append(Paragraphs(
                    para_id=str(uuid.uuid4())[:8], raw_text=p_text, sentences=sents,
                    start=item["start"], end=item["end"], page=item["page"], bbox=item["bbox"]
                ))
            # (Note: Image logic would go here if extracted)

        return Section(
            sec_id=str(uuid.uuid4())[:8],
            title=node_dict.get("title", "Untitled"),
            raw_text="\n".join(full_text),
            page_start=node_dict.get("page", 1),
            paragraphs=paras,
            subsections=[self.convert_node(c) for c in node_dict.get("children", [])]
        )

# ==========================================
# UTILITIES (TOC & Highlighting)
# ==========================================

def get_toc(sections: List[Section], level: int = 0) -> List[Dict]:
    entries = []
    for s in sections:
        entries.append({"level": level, "title": s.title, "page": s.page_start})
        entries.extend(get_toc(s.subsections, level + 1))
    return entries

def highlight_sentence(pdf_path: str, sentence: Sentence, out_path: str):
    doc = fitz.open(pdf_path)
    page = doc[sentence.page - 1]
    annot = page.add_highlight_annot(sentence.bbox)
    annot.update()
    doc.save(out_path)
    doc.close()

