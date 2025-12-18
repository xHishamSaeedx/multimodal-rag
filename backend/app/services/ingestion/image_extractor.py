"""
Image extraction service.

Extracts images from PDF and DOCX files.
Based on the proof of concept from scripts/extract_images.py.
"""

import io
import logging
import hashlib
import time
from pathlib import Path
from typing import BinaryIO, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# PDF image extraction
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# DOCX image extraction
try:
    from docx import Document as DocxDocument
    from docx.oxml import parse_xml
    from docx.oxml.ns import qn
except ImportError:
    DocxDocument = None
    parse_xml = None
    qn = None

# Image processing
try:
    from PIL import Image
except ImportError:
    Image = None

# OCR (optional)
try:
    import easyocr
except ImportError:
    easyocr = None

from app.utils.exceptions import (
    ExtractionError,
    UnsupportedFileTypeError,
    FileReadError,
)
from app.utils.metrics import (
    image_ocr_duration_seconds,
    image_ocr_processed_total,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """
    Represents an extracted image with metadata.
    
    Attributes:
        image_index: Index of the image within the document (1-based)
        page: Page number where image was found (None for DOCX)
        image_bytes: Raw image bytes
        image_ext: Image file extension (jpg, png, etc.)
        width: Image width in pixels
        height: Image height in pixels
        image_type: Classified image type (diagram, chart, photo, screenshot, unknown)
        position: Position on page (for PDF) - dict with x0, y0, x1, y1
        extracted_text: OCR text if applicable (optional)
        metadata: Additional metadata
    """
    
    image_index: int
    page: Optional[int]
    image_bytes: bytes
    image_ext: str
    width: int
    height: int
    image_type: str
    position: Optional[Dict[str, float]] = None
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageExtractor:
    """
    Service for extracting images from various document formats.
    
    Supports:
    - PDF: Using PyMuPDF (fitz)
    - DOCX: Using python-docx
    
    Features:
    - Image type classification
    - Optional OCR text extraction
    - Position tracking (for PDF)
    """
    
    # Supported file types for image extraction
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
    }
    
    def __init__(
        self,
        extract_ocr: bool = False,
        ocr_verbose: bool = False,
    ):
        """
        Initialize image extractor.
        
        Args:
            extract_ocr: Whether to extract text from images using OCR (default: False)
            ocr_verbose: Whether to print OCR progress (default: False)
        """
        self.extract_ocr = extract_ocr
        self.ocr_verbose = ocr_verbose
        self._easyocr_reader = None
        
        if extract_ocr and easyocr is None:
            logger.warning(
                "OCR requested but easyocr not installed. "
                "Install with: pip install easyocr"
            )
            self.extract_ocr = False
    
    def _get_easyocr_reader(self):
        """Get or initialize EasyOCR reader (lazy loading)."""
        if self._easyocr_reader is None and easyocr is not None:
            try:
                import torch
                use_gpu = torch.cuda.is_available() if torch else False
                if self.ocr_verbose:
                    if use_gpu:
                        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                        logger.info(f"Using GPU for OCR: {gpu_name}")
                    else:
                        logger.info("GPU not available for OCR, using CPU (slower)")
                
                self._easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.extract_ocr = False
        return self._easyocr_reader
    
    def _extract_text_from_image(self, image_bytes: bytes) -> Optional[str]:
        """Extract text from image using EasyOCR."""
        if not self.extract_ocr:
            return None
        
        try:
            # Start timing OCR
            ocr_start = time.time()
            
            reader = self._get_easyocr_reader()
            if reader is None:
                return None
            
            # Read text from image bytes
            import numpy as np
            from PIL import Image as PILImage
            
            image = PILImage.open(io.BytesIO(image_bytes))
            # Convert to numpy array for EasyOCR
            image_array = np.array(image)
            
            results = reader.readtext(image_array)
            
            # Calculate OCR duration
            ocr_duration = time.time() - ocr_start
            
            # Record metrics
            image_ocr_duration_seconds.observe(ocr_duration)
            image_ocr_processed_total.inc()
            
            # Combine all detected text
            if results:
                text_parts = [text for (bbox, text, confidence) in results if text.strip()]
                combined_text = ' '.join(text_parts)
                extracted_text = combined_text.strip() if combined_text.strip() else None
                if self.ocr_verbose:
                    logger.info(f"OCR completed in {ocr_duration:.2f}s, extracted {len(extracted_text) if extracted_text else 0} characters")
                return extracted_text
            
            if self.ocr_verbose:
                logger.info(f"OCR completed in {ocr_duration:.2f}s, no text found")
            return None
            
        except Exception as e:
            if self.ocr_verbose:
                logger.warning(f"OCR failed: {e}")
            return None
    
    def _classify_image_type(self, width: int, height: int) -> str:
        """
        Classify image type based on dimensions.
        
        Returns: 'diagram', 'chart', 'screenshot', 'photo', or 'unknown'
        """
        if height == 0:
            return 'unknown'
        
        aspect_ratio = width / height
        
        # Wide images are often diagrams or charts
        if aspect_ratio > 2.0:
            return 'diagram'
        elif aspect_ratio < 0.5:
            return 'diagram'  # Tall diagrams
        
        # Square-ish images could be charts or photos
        if 0.8 <= aspect_ratio <= 1.2:
            if width < 500:
                return 'screenshot'
            else:
                return 'chart'
        
        # Medium-sized images
        if 500 <= width <= 2000 and 500 <= height <= 2000:
            return 'chart'
        
        # Large images are often photos
        if width > 2000 or height > 2000:
            return 'photo'
        
        # Small images are often screenshots or icons
        if width < 500 and height < 500:
            return 'screenshot'
        
        return 'unknown'
    
    def is_supported(self, file_path: str) -> bool:
        """
        Check if file type is supported for image extraction.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if supported, False otherwise
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.SUPPORTED_EXTENSIONS
    
    def extract_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        file_type: Optional[str] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from document bytes.
        
        Args:
            file_bytes: Document content as bytes
            file_name: Original filename (for type detection)
            file_type: Optional file type override (pdf, docx)
        
        Returns:
            List of ExtractedImage objects
        
        Raises:
            ExtractionError: If extraction fails
            UnsupportedFileTypeError: If file type is not supported
        """
        if file_type is None:
            file_ext = Path(file_name).suffix.lower()
            file_type = self.SUPPORTED_EXTENSIONS.get(file_ext)
        
        if file_type is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type for image extraction: {file_name}",
                {"file_name": file_name, "supported_types": list(self.SUPPORTED_EXTENSIONS.keys())},
            )
        
        if file_type == "pdf":
            return self._extract_from_pdf(file_bytes, file_name)
        elif file_type == "docx":
            return self._extract_from_docx(file_bytes, file_name)
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_type}",
                {"file_name": file_name, "file_type": file_type},
            )
    
    def _extract_from_pdf(self, pdf_bytes: bytes, file_name: str) -> List[ExtractedImage]:
        """Extract images from PDF using PyMuPDF."""
        if fitz is None:
            raise ExtractionError(
                "PyMuPDF is not installed. Install it with: pip install PyMuPDF",
                {"file_name": file_name},
            )
        
        try:
            logger.info(f"Extracting images from PDF: {file_name}")
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            extracted_images = []
            image_index = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_idx, img in enumerate(image_list):
                    try:
                        # Get image reference
                        xref = img[0]
                        
                        # Extract image bytes
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_width = base_image["width"]
                        image_height = base_image["height"]
                        
                        # Get image position on page (if available)
                        image_rects = page.get_image_rects(xref)
                        position = None
                        if image_rects:
                            rect = image_rects[0]  # First occurrence
                            position = {
                                "x0": round(rect.x0, 2),
                                "y0": round(rect.y0, 2),
                                "x1": round(rect.x1, 2),
                                "y1": round(rect.y1, 2),
                            }
                        
                        # Classify image type
                        image_type = self._classify_image_type(image_width, image_height)
                        
                        # Extract OCR text if requested
                        extracted_text = None
                        if self.extract_ocr:
                            extracted_text = self._extract_text_from_image(image_bytes)
                        
                        image_index += 1
                        extracted_images.append(
                            ExtractedImage(
                                image_index=image_index,
                                page=page_num + 1,
                                image_bytes=image_bytes,
                                image_ext=image_ext,
                                width=image_width,
                                height=image_height,
                                image_type=image_type,
                                position=position,
                                extracted_text=extracted_text,
                                metadata={
                                    "xref": xref,
                                    "colorspace": base_image.get("colorspace", "unknown"),
                                    "bpc": base_image.get("bpc", 0),  # bits per component
                                },
                            )
                        )
                        
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_idx + 1} from page {page_num + 1}: {e}"
                        )
                        continue
            
            doc.close()
            
            if extracted_images:
                logger.info(f"Found {len(extracted_images)} image(s) in PDF")
            else:
                logger.info("No images found in PDF")
            
            return extracted_images
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}", exc_info=True)
            raise ExtractionError(
                f"Failed to extract images from PDF: {str(e)}",
                {"file_name": file_name, "error": str(e)},
            ) from e
    
    def _extract_from_docx(self, docx_bytes: bytes, file_name: str) -> List[ExtractedImage]:
        """Extract images from DOCX using python-docx."""
        if DocxDocument is None:
            raise ExtractionError(
                "python-docx is not installed. Install it with: pip install python-docx",
                {"file_name": file_name},
            )
        
        if Image is None:
            raise ExtractionError(
                "Pillow is not installed. Install it with: pip install pillow",
                {"file_name": file_name},
            )
        
        try:
            logger.info(f"Extracting images from DOCX: {file_name}")
            doc = DocxDocument(io.BytesIO(docx_bytes))
            
            extracted_images = []
            image_index = 0
            
            # Extract images from document relationships
            try:
                for element in doc.element.body:
                    if element.tag.endswith('p'):  # Paragraph
                        from docx.text.paragraph import Paragraph
                        paragraph = Paragraph(element, doc)
                        # Check for images in runs
                        for run in paragraph.runs:
                            for drawing in run.element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/picture}pic'):
                                # Try to extract image reference
                                blip = drawing.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
                                if blip is not None:
                                    rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                                    if rId:
                                        try:
                                            image_part = doc.part.related_parts[rId]
                                            image_bytes = image_part.blob
                                            
                                            # Determine image format from content type
                                            content_type = image_part.content_type
                                            if 'jpeg' in content_type or 'jpg' in content_type:
                                                image_ext = 'jpg'
                                            elif 'png' in content_type:
                                                image_ext = 'png'
                                            elif 'gif' in content_type:
                                                image_ext = 'gif'
                                            elif 'bmp' in content_type:
                                                image_ext = 'bmp'
                                            else:
                                                image_ext = 'png'  # Default
                                            
                                            # Get image dimensions using PIL
                                            img = Image.open(io.BytesIO(image_bytes))
                                            image_width, image_height = img.size
                                            
                                            # Classify image type
                                            image_type = self._classify_image_type(image_width, image_height)
                                            
                                            # Extract OCR text if requested
                                            extracted_text = None
                                            if self.extract_ocr:
                                                extracted_text = self._extract_text_from_image(image_bytes)
                                            
                                            image_index += 1
                                            extracted_images.append(
                                                ExtractedImage(
                                                    image_index=image_index,
                                                    page=None,  # DOCX doesn't have pages
                                                    image_bytes=image_bytes,
                                                    image_ext=image_ext,
                                                    width=image_width,
                                                    height=image_height,
                                                    image_type=image_type,
                                                    position=None,
                                                    extracted_text=extracted_text,
                                                    metadata={
                                                        "content_type": content_type,
                                                        "relationship_id": rId,
                                                    },
                                                )
                                            )
                                        except Exception as e:
                                            logger.warning(
                                                f"Failed to extract image from relationship {rId}: {e}"
                                            )
                                            continue
            except Exception as e:
                logger.warning(f"Advanced DOCX image extraction failed: {e}")
                # Fallback: try to extract from document part relationships directly
                try:
                    for rel_id, rel in doc.part.rels.items():
                        if "image" in rel.target_ref:
                            try:
                                image_part = rel.target_part
                                image_bytes = image_part.blob
                                
                                # Determine format
                                content_type = image_part.content_type
                                if 'jpeg' in content_type or 'jpg' in content_type:
                                    image_ext = 'jpg'
                                elif 'png' in content_type:
                                    image_ext = 'png'
                                elif 'gif' in content_type:
                                    image_ext = 'gif'
                                else:
                                    image_ext = 'png'
                                
                                # Get dimensions
                                img = Image.open(io.BytesIO(image_bytes))
                                image_width, image_height = img.size
                                
                                image_type = self._classify_image_type(image_width, image_height)
                                
                                extracted_text = None
                                if self.extract_ocr:
                                    extracted_text = self._extract_text_from_image(image_bytes)
                                
                                image_index += 1
                                extracted_images.append(
                                    ExtractedImage(
                                        image_index=image_index,
                                        page=None,
                                        image_bytes=image_bytes,
                                        image_ext=image_ext,
                                        width=image_width,
                                        height=image_height,
                                        image_type=image_type,
                                        position=None,
                                        extracted_text=extracted_text,
                                        metadata={
                                            "content_type": content_type,
                                            "relationship_id": rel_id,
                                        },
                                    )
                                )
                            except Exception as e:
                                logger.warning(f"Failed to extract image from relationship: {e}")
                                continue
                except Exception as e:
                    logger.error(f"Error in fallback extraction: {e}")
            
            if extracted_images:
                logger.info(f"Found {len(extracted_images)} image(s) in DOCX")
            else:
                logger.info("No images found in DOCX")
            
            return extracted_images
            
        except Exception as e:
            logger.error(f"Error extracting images from DOCX: {e}", exc_info=True)
            raise ExtractionError(
                f"Failed to extract images from DOCX: {str(e)}",
                {"file_name": file_name, "error": str(e)},
            ) from e

