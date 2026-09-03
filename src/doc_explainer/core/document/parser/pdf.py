import os
from pathlib import Path
import re
import uuid
from typing import Iterator, List, Dict, Any, Optional
from collections import defaultdict
import pymupdf as fitz
import spacy

from doc_explainer.core.document.models.base import DocumentMetadata

from .base import DocumentParser
from ..builder.strategies.font_analyzer import FontAnalyzer
from ..builder.strategies.structure_detector import StructureDetector, FontInfo
from ..builder.strategies.image_extractor import ImageExtractor
from ..models.content import Sentence, Paragraph, Image, Table, Equation
from ..models.structure import Section, Document
from ..models.metadata import SimpleMetadataCreator


class PDFParser(DocumentParser):
    """Parser for PDF documents"""

    def __init__(self, output_dir: str = "output", spacy_model: str = "en_core_web_sm"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize NLP
        try:
            self.nlp = spacy.load(spacy_model)
        except:
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)

        # Initialize strategies
        self.font_analyzer = FontAnalyzer()
        self.detector = StructureDetector(self.font_analyzer)
        self.image_extractor = ImageExtractor(output_dir)
        self.metadata_creator = SimpleMetadataCreator()

        # State for current parse
        self.global_cursor = 0
        self.images_on_page = defaultdict(list)
        self.pending_captions = []

    def parse_metadata(self, file_path: str | Path) -> DocumentMetadata:
        """
        Extract document-level metadata without parsing the complete document.
        """

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        stat = path.stat()

        document_id = self._generate_document_id(path)

        # PDF-specific metadata
        title = ""
        author = ""
        page_count = 0
        pdf_metadata = {}

        try:
            with fitz.open(path) as doc:
                page_count = len(doc)

                pdf_metadata = doc.metadata or {}

                title = pdf_metadata.get("title") or ""
                author = pdf_metadata.get("author") or ""

        except Exception as exc:
            raise RuntimeError(
                f"Failed to read document metadata: {path}"
            ) from exc

        return DocumentMetadata(
            document_id=document_id,
            file_path=str(path),
            filename=path.name,
            title=title,
            author=author,
            page_count=page_count,
            file_size=stat.st_size,
            metadata=pdf_metadata,
        )

    def iter_sections(
        self,
        file_path: str | Path,
    ) -> Iterator[Section]:
        """
        Stream sections from the document.

        IMPORTANT:
        This method does NOT build the complete document tree.

        Memory usage is approximately:
            O(size of current section)
        """

        metadata = self.parse_metadata(file_path)
        current_section: Optional[Section] = None
        section_index = 0

        with fitz.open(metadata.file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if not text.strip():
                    continue
                lines = self._clean_lines(text)
                line_index = 0
                while line_index < len(lines):
                    line = lines[line_index]

                    # Some PDFs place section numbers on their own line.
                    if (
                        re.fullmatch(r"\d+", line)
                        and line_index + 1 < len(lines)
                    ):
                        line = f"{line} {lines[line_index + 1]}"
                        line_index += 1

                    if self._is_section_heading(line):
                        # Emit previous section
                        if current_section is not None:
                            yield current_section

                        section_id = (
                            f"{metadata.document_id}:s:{section_index}"
                        )

                        current_section = Section(
                            section_id=section_id,
                            document_id=metadata.document_id,
                            title=line.strip(),
                            level=self._detect_section_level(line),
                            page=page_number,
                        )

                        section_index += 1

                    else:

                        # If text occurs before the first heading,
                        # create an implicit section.
                        if current_section is None:
                            current_section = Section(
                                section_id=f"{metadata.document_id}:s:0",
                                document_id=metadata.document_id,
                                title="Introduction",
                                level=0,
                                page=page_number,
                            )

                            section_index = 1

                        paragraph = self._parse_paragraph(
                            line=line,
                            document_id=metadata.document_id,
                            section_id=current_section.id,
                            page_number=page_number,
                            paragraph_index=len(
                                current_section.paragraphs
                            ),
                        )

                        if paragraph is not None:
                            current_section.paragraphs.append(
                                paragraph
                            )

                    line_index += 1

        # Emit final section
        if current_section is not None:
            yield current_section

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_document_id(self, path: Path) -> str:
        """
        Generate a stable document ID.

        For production, hashing file content is even better if documents
        can be replaced while keeping the same filename.
        """

        import hashlib

        value = f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}"

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()[:16]

    def _clean_lines(self, text: str) -> list[str]:
        lines = []

        for line in text.splitlines():

            line = re.sub(r"\s+", " ", line).strip()

            if line:
                lines.append(line)

        return lines


    def _is_section_heading(self, line: str) -> bool:
        """
        Basic heading detection.

        Replace this later with a proper PDF/layout-aware detector.
        """

        # Examples:
        # 1 Introduction
        # 1.1 Background
        # 2 Methodology
        # 3.1 Results

        pattern = r"^\d+(\.\d+)*\.?\s+.+$"

        if re.match(pattern, line):
            return True

        # Markdown-like headings
        if line.startswith("#"):
            return True

        return False

    def _detect_section_level(self, line: str) -> int:

        line = line.strip()

        match = re.match(
            r"^(\d+(?:\.\d+)*)\.?\s+",
            line,
        )

        if match:
            number = match.group(1)

            return number.count(".") + 1

        if line.startswith("###"):
            return 3

        if line.startswith("##"):
            return 2

        if line.startswith("#"):
            return 1

        return 1

    def _parse_paragraph(
        self,
        line: str,
        document_id: str,
        section_id: str,
        page_number: int,
        paragraph_index: int,
    ) -> Optional[Paragraph]:

        if not line.strip():
            return None

        paragraph_id = (
            f"{section_id}:p:{paragraph_index}"
        )

        sentences = self._split_sentences(
            line,
            paragraph_id,
            page_number,
        )

        return Paragraph(
            id=paragraph_id,
            text=line,
            page=page_number,
            bbox=(0.0, 0.0, 0.0, 0.0),
            sentences=sentences,
            start=0,
            end=0,
        )

    def _split_sentences(
        self,
        text: str,
        paragraph_id: str,
        page_number: int,
    ) -> list[Sentence]:

        parts = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        sentences = []

        for i, sentence_text in enumerate(parts):

            sentence_text = sentence_text.strip()

            if not sentence_text:
                continue

            sentence_id = (
                f"{paragraph_id}:t:{i}"
            )

            sentences.append(
                Sentence(
                    id=sentence_id,
                    text=sentence_text,
                    page=page_number,
                    start=0,
                    end=len(sentence_text),
                    bbox=(0.0, 0.0, 0.0, 0.0),
                )
            )

        return sentences
    
    def parse(self, file_path: str) -> Document:
        """Parse PDF document"""
        doc = fitz.open(file_path)

        # Analyze fonts first
        self.font_analyzer.analyze(doc)

        # Build document tree
        root = self._build_document_tree(doc, file_path)
        doc.close()

        # Convert to Document object
        return self._convert_to_document(root, file_path)

    def _build_document_tree(self, doc: fitz.Document, file_path: str) -> Dict:
        """Build raw document tree"""
        root = {
            "title": "Document",
            "text": "",
            "level": -1,
            "content": [],
            "children": [],
            "page": 1,
            "path": file_path
        }

        stack = [root]
        self.global_cursor = 0
        self.images_on_page.clear()
        self.pending_captions.clear()

        for page_num, page in enumerate(doc):
            current_page = page_num + 1
            self._process_page(page, current_page, stack)

        # Link any pending captions
        self._link_pending_captions()

        return root

    def _process_page(self, page: fitz.Page, page_num: int, stack: List[Dict]):
        """Process a single page"""
        # Extract images
        images = self.image_extractor.extract_from_page(page, page_num)
        for img in images:
            img_obj = {
                "type": "image",
                "path": img["path"],
                "caption": None,
                "bbox": img["bbox"],
                "page": page_num
            }
            self.images_on_page[page_num].append(img_obj)
            stack[-1]["content"].append(img_obj)

        # Process text blocks
        blocks = page.get_text("dict")["blocks"]
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        for block in blocks:
            if block["type"] != 0:
                continue
            self._process_text_block(block, page_num, stack)

    def _process_text_block(self, block: Dict, page_num: int, stack: List[Dict]):
        """Process a text block"""
        spans = [s for line in block["lines"] for s in line["spans"]]
        if not spans:
            return

        text = re.sub(
            r"\s+", " ", " ".join([s["text"] for s in spans])).strip()
        if not text:
            return

        span = spans[0]
        font_info = FontInfo(
            size=round(span["size"], 1),
            name=span["font"],
            flags=span["flags"]
        )

        structure = self.detector.classify(
            text, font_info, page_num - 1, block["bbox"]
        )

        if structure is None:
            return

        meta = {
            "text": text,
            "start": self.global_cursor,
            "end": self.global_cursor + len(text),
            "page": page_num,
            "bbox": block["bbox"]
        }

        self._handle_structured_content(structure, meta, stack)
        self.global_cursor = meta["end"] + 1

    def _handle_structured_content(self, structure: Dict, meta: Dict, stack: List[Dict]):
        """Handle different types of structured content"""
        if structure["type"] == "section":
            self._handle_section(structure, meta, stack)

        elif structure["type"] == "paragraph":
            stack[-1]["content"].append({"type": "paragraph", **meta})

        elif structure["type"] == "figure_caption":
            self._handle_figure_caption(structure, meta, stack)

        elif structure["type"] == "document_title":
            stack[0]["title"] = structure["title"]

        elif structure["type"] in ["table_caption", "equation"]:
            stack[-1]["content"].append({"type": structure["type"], **meta})

    def _handle_section(self, structure: Dict, meta: Dict, stack: List[Dict]):
        """Handle section heading"""
        new_node = {
            "title": structure["title"],
            "level": structure["level"],
            "content": [],
            "children": [],
            **meta
        }

        while stack and stack[-1]["level"] >= new_node["level"]:
            stack.pop()

        stack[-1]["children"].append(new_node)
        stack.append(new_node)

    def _handle_figure_caption(self, structure: Dict, meta: Dict, stack: List[Dict]):
        """Handle figure caption"""
        linked_image = self._find_nearest_image(
            self.images_on_page[meta["page"]], meta["bbox"]
        )

        if linked_image:
            linked_image["caption"] = structure["text"]
            meta["linked_image"] = linked_image["path"]
        else:
            self.pending_captions.append({
                "caption": structure["text"],
                "bbox": meta["bbox"],
                "page": meta["page"],
                "meta": meta
            })

        stack[-1]["content"].append({"type": "figure_caption", **meta})

    def _find_nearest_image(self, images: List[Dict], text_bbox: tuple,
                            max_distance: float = 200) -> Optional[Dict]:
        """Find nearest image to text bbox"""
        if not images:
            return None

        def bbox_distance(bbox1, bbox2):
            center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
            return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

        nearest = None
        min_dist = float('inf')

        for img in images:
            if img.get("caption") is not None:
                continue

            distance = bbox_distance(img["bbox"], text_bbox)

            # Captions are typically below images
            if text_bbox[1] > img["bbox"][3]:
                distance *= 0.8

            if distance < min_dist and distance < max_distance:
                min_dist = distance
                nearest = img

        return nearest

    def _link_pending_captions(self):
        """Link captions that couldn't be linked during first pass"""
        for caption in self.pending_captions:
            if caption["page"] in self.images_on_page:
                linked = self._find_nearest_image(
                    self.images_on_page[caption["page"]], caption["bbox"]
                )
                if linked:
                    linked["caption"] = caption["caption"]

    def _convert_to_document(self, raw_tree: Dict, file_path: str) -> Document:
        """Convert raw tree to Document object"""
        sections = [self._convert_section(child)
                    for child in raw_tree["children"]]

        return Document(
            path=file_path,
            title=raw_tree.get("title", ""),
            sections=sections
        )

    def _convert_section(self, raw_section: Dict) -> Section:
        """Convert raw section to Section object"""
        paragraphs = []
        images = []
        tables = []
        equations = []
        full_text = []

        for item in raw_section.get("content", []):
            if item["type"] == "paragraph":
                paragraph = self._create_paragraph(item)
                paragraphs.append(paragraph)
                full_text.append(paragraph.text)

            elif item["type"] == "image":
                images.append(self._create_image(item))

            elif item["type"] == "table_caption":
                # Handle table
                pass

            elif item["type"] == "equation":
                equations.append(self._create_equation(item))

        subsections = [self._convert_section(
            child) for child in raw_section.get("children", [])]

        return Section(
            title=raw_section["title"],
            text="\n".join(full_text),
            page_start=raw_section["page"],
            paragraphs=paragraphs,
            images=images,
            tables=tables,
            equations=equations,
            subsections=subsections
        )

    def _create_paragraph(self, item: Dict) -> Paragraph:
        """Create Paragraph from item"""
        sentences = self._text_to_sentences(
            item["text"], item["start"], item["page"], item["bbox"]
        )
        return Paragraph(
            text=item["text"],
            sentences=sentences,
            start=item["start"],
            end=item["end"],
            page=item["page"],
            bbox=item["bbox"]
        )

    def _create_image(self, item: Dict) -> Image:
        """Create Image from item"""
        caption_sentences = []
        if item.get("caption"):
            caption_sentences = self._text_to_sentences(
                item["caption"], 0, item["page"], item["bbox"]
            )

        return Image(
            image_path=item["path"],
            caption=caption_sentences,
            page=item["page"],
            bbox=item["bbox"]
        )

    def _create_equation(self, item: Dict) -> Equation:
        """Create Equation from item"""
        caption_sentences = self._text_to_sentences(
            item["text"], item["start"], item["page"], item["bbox"]
        )
        return Equation(
            text=item["text"],
            caption=caption_sentences,
            start=item["start"],
            end=item["end"],
            page=item["page"],
            bbox=item["bbox"]
        )

    def _text_to_sentences(self, text: str, start: int, page: int,
                           bbox: tuple) -> List[Sentence]:
        """Split text into sentences"""
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            sentences.append(Sentence(
                text=sent.text.strip(),
                start=start + sent.start_char,
                end=start + sent.end_char,
                page=page,
                bbox=bbox
            ))

        return sentences

    def to_json(self, document: Document, output_path: str):
        """Save document to JSON"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

    def from_json(self, json_path: str) -> Optional[Document]:
        """Load document from JSON"""
        import json
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Document.from_dict(data)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None
