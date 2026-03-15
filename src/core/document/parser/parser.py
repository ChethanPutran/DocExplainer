from __future__ import annotations
import json
from collections import defaultdict
import os
import re
import uuid
import fitz
import spacy
from typing import List
from ..document_modals import Paragraphs, Sentence, Image, Section, Document, Table, Equation
from .structure_detector import StructureDetector
import pdfplumber


class PDFTreeParser:

    def __init__(self, output_dir="output"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir,"images"), exist_ok=True)
        self.detector = StructureDetector()


    def table_to_text(self, table):

        rows = []

        for row in table:
            rows.append(" | ".join(str(x) for x in row))

        return "\n".join(rows)
    
    def extract_tables(self, pdf_path):
        """Extract tables from PDF with their positions"""
        tables_by_page = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables with their bounding boxes
                page_tables = page.extract_tables()
                print(page_tables)
                table_bboxes = []
                
                # Try to get table bounding boxes if available
                if hasattr(page, 'find_tables'):
                    tables = page.find_tables()
                    table_bboxes = [(table.bbox) for table in tables]
                
                tables_by_page[page_num + 1] = {
                    'data': page_tables,
                    'bboxes': table_bboxes
                }
                
        return tables_by_page

    def find_table_at_position(self, tables_dict, page_num, bbox, tolerance=50):
        """Find if there's a table at or near the given bbox position"""
        if page_num not in tables_dict:
            return None
            
        page_tables = tables_dict[page_num]
        
        # If we have bboxes from pdfplumber's table detection
        if page_tables['bboxes']:
            for i, table_bbox in enumerate(page_tables['bboxes']):
                # Check if bboxes overlap or are close
                if (abs(table_bbox[0] - bbox[0]) < tolerance and 
                    abs(table_bbox[1] - bbox[1]) < tolerance):
                    if i < len(page_tables['data']):
                        return page_tables['data'][i]
        
        # Fallback: just return the first table on the page if any
        # (you might want more sophisticated logic here)
        if page_tables['data']:
            return page_tables['data'][0]
            
        return None


    def parse(self, pdf_path):
        doc = fitz.open(pdf_path)


        root = {
            "title": "Document",
            "level": -1,
            "content": [],
            "children": [],
            "page": 1,
            "path": pdf_path
        }

        stack = [root]
        global_cursor = 0
        self.detector.analyze_fonts(doc)

        # Extract all tables first
        tables_dict = self.extract_tables(pdf_path)
        table_counter = 0

        # Track images and their potential captions per page
        images_on_page = defaultdict(list)  # page_num -> list of image dicts
        pending_captions = []  # Store captions that need to be linked
        

        for page_num, page in enumerate(doc):
            curr_page = page_num + 1

            blocks = page.get_text("dict")["blocks"]
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

            # IMAGE EXTRACTION - Fixed version
            image_infos = page.get_image_info()
            
            # Get all images from the page
            images_list = page.get_images(full=True)
            
            for img_index, img in enumerate(images_list):
                xref = img[0]  # xref is the first element in the tuple
                
                # Find matching image info by bbox or other criteria
                matching_info = None
                for info in image_infos:
                    # You might need to adjust this matching logic based on your PDF structure
                    if 'bbox' in info and len(info['bbox']) == 4:
                        # Simple matching - you might need more sophisticated matching
                        matching_info = info
                        break
                
                if matching_info is None and image_infos:
                    # Fallback to first image info if available
                    matching_info = image_infos[img_index] if img_index < len(image_infos) else None
                
                if matching_info:
                    bbox = matching_info.get('bbox', (0, 0, 0, 0))
                else:
                    # Fallback bbox
                    bbox = (0, 0, 0, 0)
                    print(f"Warning: No bbox found for image xref {xref} on page {curr_page}")
                
                try:
                    # Extract and save the image
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Handle different colorspace cases
                    if pix.n - pix.alpha < 4:  # Can be converted to RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    img_name = f"{uuid.uuid4().hex}.png"
                    img_path = os.path.join(self.output_dir,"images", img_name)
                    pix.save(img_path)
                    pix = None  # Free the pixmap
                    
                    # Create image object without caption yet
                    img_obj = {
                        "type": "image",
                        "raw_img": img_path,
                        "caption": None,  # Will be filled later
                        "bbox": bbox,
                        "page": curr_page,
                        "xref": xref  # Store xref for reference
                    }
                    
                    # Add to current page's images
                    images_on_page[curr_page].append(img_obj)
                    
                    # Also add to content for now (caption will be updated)
                    stack[-1]["content"].append(img_obj)
                    
                except Exception as e:
                    print(f"Error extracting image xref {xref} on page {curr_page}: {e}")
                    continue

            # TEXT BLOCKS
            for b in blocks:

                if b["type"] != 0:
                    continue

                spans = [s for l in b["lines"] for s in l["spans"]]

                if not spans:
                    continue

                text = re.sub(
                    r"\s+", " ", " ".join([s["text"] for s in spans])).strip()

                if not text:
                    continue

                span = spans[0]
                structure = self.detector.classify_block(
                    text, span, page_num, b["bbox"])

                meta = {
                    "text": text,
                    "start": global_cursor,
                    "end": global_cursor + len(text),
                    "page": curr_page,
                    "bbox": b["bbox"]
                }
                if structure is None:
                    continue

                if structure["type"] == "section":

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

                elif structure["type"] == "paragraph":

                    stack[-1]["content"].append({
                        "type": "paragraph",
                        **meta
                    })
                     # Try to find a table near this caption
                    table_data = self.find_table_at_position(tables_dict, curr_page, b["bbox"])
                    
                    if table_data:
                        # Convert table data to text representation
                        table_text = self.table_to_text(table_data)
                        
                        stack[-1]["content"].append({
                            "type": "table",
                            "caption": text,
                            "data": table_data,
                            "raw_text": table_text,
                            **meta
                        })
                        table_counter += 1
                    else:
                        # Still add as table caption even if no table found
                        stack[-1]["content"].append({
                            "type": "table_caption",
                            "caption": text,
                            **meta
                        })
                elif structure["type"] == "table_caption":

                    stack[-1]["content"].append({
                        "type": "table_caption",
                        "caption": text,
                        **meta
                    })

                elif structure["type"] == "figure_caption":
                    linked_image = None
                    # Try to find the nearest image on the same page
                    if curr_page in images_on_page:
                        linked_image = self.find_nearest_image(images_on_page[curr_page], b["bbox"])
                  
                    if linked_image:
                        # Update the image with this caption
                        linked_image["caption"] = text
                        
                        # Add the caption as a separate item for reference
                        stack[-1]["content"].append({
                            "type": "figure_caption",
                            "caption": text,
                            "linked_image": linked_image["raw_img"],  # Reference to the image
                            **meta
                        })
                    else:
                        # Store caption for later linking if no image found yet
                        pending_captions.append({
                            "caption": text,
                            "bbox": b["bbox"],
                            "page": curr_page,
                            "meta": meta
                        })
                        

                elif structure["type"] == "equation":

                    stack[-1]["content"].append({
                        "type": "equation",
                        "raw_text": text,
                        **meta
                    })
                elif structure["type"] == "document_title":

                    root["title"] = structure["title"]

                global_cursor = meta["end"] + 1

        # Third pass: Try to link any pending captions
        self.link_pending_captions(pending_captions, images_on_page, stack)

        doc.close()

        return self.convert(root)

    def find_nearest_image(self, images, text_bbox, max_distance=200):
        """Find the nearest image to a text bbox (likely the caption)"""
        if not images:
            return None
            
        def bbox_distance(bbox1, bbox2):
            # Calculate center points
            center1_x = (bbox1[0] + bbox1[2]) / 2
            center1_y = (bbox1[1] + bbox1[3]) / 2
            center2_x = (bbox2[0] + bbox2[2]) / 2
            center2_y = (bbox2[1] + bbox2[3]) / 2
            
            # Calculate Euclidean distance
            return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        # Find image with minimum distance to text
        nearest_image = None
        min_distance = float('inf')
        
        for img in images:
            # Only consider images without captions yet
            if img.get("caption") is None:
                distance = bbox_distance(img["bbox"], text_bbox)
                
                # Captions are typically below images
                if text_bbox[1] > img["bbox"][3]:  # Text is below image
                    distance *= 0.8  # Favor captions below images
                    
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    nearest_image = img
        
        return nearest_image

    def link_pending_captions(self, pending_captions, images_on_page, stack):
        """Link captions that couldn't be linked during first pass"""
        for caption in pending_captions:
            page = caption["page"]
            if page in images_on_page:
                linked_image = self.find_nearest_image(images_on_page[page], caption["bbox"])
                
                if linked_image:
                    linked_image["caption"] = caption["caption"]
                    
                    # Update the caption reference in content
                    for item in stack[-1]["content"]:
                        if (item.get("type") == "figure_caption" and 
                            item.get("caption") == caption["caption"]):
                            item["linked_image"] = linked_image["raw_img"]
                            break

    def to_sentences(self, text, start, page, bbox)->List[Sentence]:
        doc = self.nlp(text)
        sentences = []

        for s in doc.sents:

            sentences.append(
                Sentence(
                    sen_id=uuid.uuid4().int,
                    raw_text=s.text.strip(),
                    start=start + s.start_char,
                    end=start + s.end_char,
                    page=page,
                    bbox=bbox
                )
            )

        return sentences

    def convert_node(self, node):
        paragraphs = []
        images = []
        full_text = []
        tables = []
        equations = []

        # First pass: collect all items
        for item in node.get("content", []):
            if item["type"] == "paragraph":
                p_text = item["text"]
                full_text.append(p_text)
                sentences = self.to_sentences(p_text, item["start"], item["page"], item["bbox"])
                paragraphs.append(
                    Paragraphs(
                        para_id=uuid.uuid4().int,
                        raw_text=p_text,
                        sentences=sentences,
                        start=item["start"],
                        end=item["end"],
                        page=item["page"],
                        bbox=item["bbox"]
                    )
                )

            elif item["type"] == "image":
                # Convert caption to sentences if it exists
                caption_sentences = [Sentence(-100,'')]
                if item.get("caption"):
                    # Estimate start/end for caption (since it's not in the main text flow)
                    caption_sentences = self.to_sentences(
                        item["caption"],
                        0,  # Start position (not used for images)
                        item["page"],
                        item["bbox"]  # Using image bbox as reference
                    )
                
                images.append(
                    Image(
                        img_id=str(uuid.uuid4())[:8],
                        raw_img=item["raw_img"],
                        caption=caption_sentences,
                        page=item["page"],
                        bbox=item["bbox"]
                    )
                )

            elif item["type"] == "table":
                caption_sent = self.to_sentences(
                    item["caption"],
                    item["start"],
                    item["page"],
                    item["bbox"]
                )
                
                tables.append(
                    Table(
                        table_id=uuid.uuid4().int,
                        raw_text=item["raw_text"],
                        data=item["data"],
                        caption=caption_sent,
                        start=item["start"],
                        end=item["end"],
                        page=item["page"],
                        bbox=item["bbox"]
                    )
                )
                
            elif item["type"] == "equation":
                caption_sent = self.to_sentences(
                    item["raw_text"],
                    item["start"],
                    item["page"],
                    item["bbox"]
                )
                equations.append(
                    Equation(
                        equation_id=uuid.uuid4().int,
                        raw_text=item["raw_text"],
                        caption=caption_sent,
                        start=item["start"],
                        end=item["end"],
                        page=item["page"],
                        bbox=item["bbox"]
                    )
                )
            
            # We don't add figure_caption items directly - they're already linked to images

        subsections = [self.convert_node(c) for c in node.get("children", [])]

        return Section(
            sec_id=uuid.uuid4().int,
            title=node["title"],
            raw_text="\n".join(full_text),
            page_start=node["page"],
            paragraphs=paragraphs,
            images=images,  # Now images have their captions
            tables=tables,
            equations=equations,
            subsections=subsections
        )
    def convert(self, raw_tree):

        sections = [
            self.convert_node(s) for s in raw_tree["children"]
        ]

        return Document(
            doc_id=uuid.uuid4().int,
            path=raw_tree["path"],
            raw_text="",
            title=raw_tree.get("title", ""),
            sections=sections

        )
if __name__ == "__main__":
    
    from src.core.document.utils import display_document
    from src.core.document.html.generator import generate_document_html
    import json
    
    parser = PDFTreeParser()
    document = parser.parse("data/report2.pdf")

    # # # Display document structure in console
    # # display_document(document)
    
    # # Save to JSON
    # json_path = os.path.join(parser.output_dir, 'document.json')
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
    
    # print(f"\nJSON saved to: {json_path}")
    
    # # Generate HTML for visualization
    # html_path = os.path.join(parser.output_dir, 'document_visualization.html')
    # generate_document_html(json_path, html_path)
    
    # print(f"HTML visualization saved to: {html_path}")
    
    # # Automatically open in browser
    # import webbrowser
    # webbrowser.open(f'file://{os.path.abspath(html_path)}')