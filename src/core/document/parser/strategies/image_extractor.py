import os
import uuid
from typing import List, Dict, Any, Optional
import fitz
from ...models.content import Image


class ImageExtractor:
    """Extracts images from PDF documents"""
    
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract all images from a page"""
        images = []
        image_infos = page.get_image_info()
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]  # xref is the first element
            
            # Find matching image info
            matching_info = None
            for info in image_infos:
                if 'bbox' in info and len(info['bbox']) == 4:
                    matching_info = info
                    break
            
            if matching_info is None and image_infos:
                matching_info = image_infos[img_index] if img_index < len(image_infos) else None
            
            bbox = matching_info.get('bbox', (0, 0, 0, 0)) if matching_info else (0, 0, 0, 0)
            
            try:
                # Extract and save image
                pix = fitz.Pixmap(page.parent, xref)
                
                # Convert if needed
                if pix.n - pix.alpha < 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Save image
                img_name = f"{uuid.uuid4().hex}.png"
                img_path = os.path.join(self.output_dir, img_name)
                pix.save(img_path)
                pix = None
                
                images.append({
                    "path": img_path,
                    "bbox": bbox,
                    "page": page_num,
                    "xref": xref
                })
                
            except Exception as e:
                print(f"Error extracting image xref {xref} on page {page_num}: {e}")
                continue
        
        return images
    
    def create_image_object(self, image_data: Dict[str, Any], caption: Optional[str] = None) -> Image:
        """Create an Image object from extracted data"""
        return Image(
            image_path=image_data["path"],
            page=image_data["page"],
            bbox=image_data["bbox"]
        )