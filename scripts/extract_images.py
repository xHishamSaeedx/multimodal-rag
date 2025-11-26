#!/usr/bin/env python3
"""
Standalone script to extract images from PDF and DOCX files.

Usage:
    python extract_images.py <file_path> [--output-dir <dir>]

Requirements:
    - For PDF: PyMuPDF (pip install PyMuPDF)
    - For DOCX: python-docx (pip install python-docx)
    - Optional: easyocr (pip install easyocr) for OCR
    - Optional: timm, torch, torchvision (pip install timm torch torchvision) for SigLIP embeddings
    - Optional: sentence-transformers, torch (pip install sentence-transformers torch) for CLIP embeddings
    - Image processing: Pillow (pip install pillow)
"""

import sys
import json
import argparse
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def classify_image_type(image_path: str, width: int, height: int) -> str:
    """
    Classify image type based on dimensions and filename.
    
    Returns: 'diagram', 'chart', 'screenshot', 'photo', or 'unknown'
    """
    # Simple heuristics based on dimensions
    aspect_ratio = width / height if height > 0 else 1.0
    
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


# Global EasyOCR reader instance (lazy loaded)
_easyocr_reader = None
_easyocr_using_gpu = False


def _check_gpu_available() -> bool:
    """Check if GPU (CUDA) is available for PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_easyocr_reader(verbose: bool = False):
    """
    Get or initialize EasyOCR reader (lazy loading).
    
    Args:
        verbose: If True, print GPU status information.
    
    Returns:
        EasyOCR reader instance.
    """
    global _easyocr_reader, _easyocr_using_gpu
    if _easyocr_reader is None:
        import easyocr
        
        # Check if GPU is available
        use_gpu = _check_gpu_available()
        _easyocr_using_gpu = use_gpu
        
        if verbose:
            if use_gpu:
                import torch
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                print(f"  Using GPU: {gpu_name}")
            else:
                print("  GPU not available, using CPU (slower)")
        
        # Initialize with English language support
        # First run will download models automatically
        _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _easyocr_reader


def extract_text_from_image(image_path: str, verbose: bool = False) -> Optional[str]:
    """
    Extract text from image using EasyOCR (optional).
    
    Args:
        image_path: Path to the image file
        verbose: If True, print warnings on failure. If False, silently return None.
    
    Returns: Extracted text or None if OCR is not available or fails.
    """
    try:
        reader = _get_easyocr_reader(verbose=verbose)
        
        # Read text from image
        results = reader.readtext(str(image_path))
        
        # Combine all detected text
        if results:
            # results is a list of tuples: (bbox, text, confidence)
            text_parts = [text for (bbox, text, confidence) in results if text.strip()]
            combined_text = ' '.join(text_parts)
            return combined_text.strip() if combined_text.strip() else None
        return None
        
    except ImportError:
        if verbose:
            print(f"  Note: easyocr not installed. Install it with: pip install easyocr")
            print(f"        OCR skipped for {Path(image_path).name}")
        return None
    except Exception as e:
        if verbose:
            print(f"  Warning: OCR failed for {Path(image_path).name}: {e}")
        return None


# Global embedding model instances (lazy loaded)
_embedding_model = None
_embedding_device = None
_embedding_model_type = None  # 'clip' or 'siglip'


def _check_gpu_available_for_embeddings() -> str:
    """Check if GPU (CUDA) is available and return device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def _get_embedding_model(model_type: str = 'siglip', verbose: bool = False):
    """
    Get or initialize embedding model (CLIP or SigLIP).
    
    Args:
        model_type: 'clip' or 'siglip' (default: 'siglip')
        verbose: If True, print model loading information.
    
    Returns:
        Model instance and device
    """
    global _embedding_model, _embedding_device, _embedding_model_type
    
    # Return cached model if same type
    if _embedding_model is not None and _embedding_model_type == model_type:
        return _embedding_model, _embedding_device
    
    device = _check_gpu_available_for_embeddings()
    _embedding_device = device
    _embedding_model_type = model_type
    
    try:
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        
        if verbose:
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                print(f"  Loading {model_type.upper()} model on GPU: {gpu_name}")
            else:
                print(f"  Loading {model_type.upper()} model on CPU (slower)")
        
        if model_type.lower() == 'siglip':
            # Use timm for SigLIP (PyTorch, no TensorFlow)
            try:
                import timm
                
                # Load SigLIP model from timm
                # timm has: 'vit_base_patch16_siglip_224' (768 dim), 'vit_large_patch16_siglip_384' (1024 dim), etc.
                # Using large model for better quality (1024 dimensions)
                model_name = 'vit_large_patch16_siglip_384'  # 1024 dimensions
                _embedding_model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 for features only
                _embedding_model = _embedding_model.to(device)
                _embedding_model.eval()
                
                # Create image preprocessing transform
                from timm.data import resolve_data_config
                from timm.data.transforms_factory import create_transform
                config = resolve_data_config({}, model=_embedding_model)
                _embedding_model.transform = create_transform(**config)
                
                if verbose:
                    # Get actual feature dimension
                    feat_dim = getattr(_embedding_model, 'num_features', 1024)
                    print(f"  SigLIP model loaded successfully ({feat_dim} dimensions)")
                    
            except ImportError:
                raise ImportError(
                    "timm is required for SigLIP embeddings. "
                    "Install with: pip install timm torch torchvision"
                )
        
        elif model_type.lower() == 'clip':
            # Use sentence-transformers for CLIP
            try:
                from sentence_transformers import SentenceTransformer
                
                model_name = "sentence-transformers/clip-ViT-L-14"
                _embedding_model = SentenceTransformer(model_name, device=device)
                
                if verbose:
                    embedding_dim = _embedding_model.get_sentence_embedding_dimension()
                    print(f"  CLIP model loaded successfully ({embedding_dim} dimensions)")
                    
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CLIP embeddings. "
                    "Install with: pip install sentence-transformers torch"
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'clip' or 'siglip'")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_type.upper()} model: {e}")
    
    return _embedding_model, _embedding_device


def generate_image_embedding(image_path: str, model_type: str = 'siglip', verbose: bool = False) -> Optional[np.ndarray]:
    """
    Generate image embedding using CLIP or SigLIP.
    
    Args:
        image_path: Path to the image file
        model_type: 'clip' or 'siglip' (default: 'siglip')
        verbose: If True, print progress information.
    
    Returns: NumPy array of embedding (1024 dimensions for SigLIP, 768 for CLIP) or None if failed.
    """
    try:
        from PIL import Image
        import torch
        
        # Load model
        model, device = _get_embedding_model(model_type=model_type, verbose=verbose)
        
        if model_type.lower() == 'siglip':
            # SigLIP via timm
            image = Image.open(image_path).convert("RGB")
            
            # Apply transform
            if hasattr(model, 'transform'):
                input_tensor = model.transform(image).unsqueeze(0).to(device)
            else:
                # Fallback transform
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Generate embedding
            with torch.no_grad():
                # timm models return features directly
                features = model.forward_features(input_tensor)
                
                # Handle different feature formats
                if isinstance(features, (list, tuple)):
                    features = features[-1]  # Get last layer
                elif isinstance(features, dict):
                    # Get the main feature tensor
                    if 'x_norm_clstoken' in features:
                        embedding = features['x_norm_clstoken']
                    elif 'x_norm_patchtokens' in features:
                        embedding = features['x_norm_patchtokens'].mean(dim=1)
                    else:
                        embedding = list(features.values())[0]
                        if len(embedding.shape) > 2:
                            embedding = embedding.mean(dim=1)
                else:
                    # Tensor output - use CLS token (first token) or global pool
                    if len(features.shape) == 3:
                        # [batch, seq_len, dim] - use CLS token (first token)
                        embedding = features[:, 0, :]
                    else:
                        embedding = features
                
                # Normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding_np = embedding.cpu().numpy().flatten()
            
            return embedding_np
            
        elif model_type.lower() == 'clip':
            # CLIP via sentence-transformers
            embedding = model.encode(image_path, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.flatten()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    except ImportError as e:
        if verbose:
            print(f"  Note: Required libraries not installed. Embedding skipped for {Path(image_path).name}")
        return None
    except Exception as e:
        if verbose:
            print(f"  Warning: Embedding generation failed for {Path(image_path).name}: {e}")
        return None


def save_embedding(embedding: np.ndarray, output_path: Path, model_type: str = 'siglip', verbose: bool = False):
    """
    Save embedding to file (both .npy and .json formats).
    
    Args:
        embedding: NumPy array of embedding
        output_path: Base path for saving (without extension)
        model_type: 'clip' or 'siglip' (for metadata)
        verbose: If True, print save information.
    """
    try:
        # Save as .npy (binary, efficient)
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, embedding)
        
        # Also save as .json (human-readable, for compatibility)
        json_path = output_path.with_suffix('.json')
        embedding_list = embedding.tolist()
        model_name = 'timm/vit_large_patch16_siglip_384' if model_type == 'siglip' else 'sentence-transformers/clip-ViT-L-14'
        with open(json_path, 'w') as f:
            json.dump({
                'embedding': embedding_list,
                'dimension': len(embedding_list),
                'model': model_name
            }, f, indent=2)
        
        if verbose:
            print(f"    Saved embedding: {npy_path.name} ({len(embedding)} dimensions)")
            
    except Exception as e:
        if verbose:
            print(f"    Warning: Failed to save embedding: {e}")


def extract_images_from_pdf(pdf_path: str, output_dir: Optional[Path] = None, extract_ocr: bool = False, generate_embeddings: bool = False, embedding_model: str = 'siglip') -> List[Dict[str, Any]]:
    """Extract images from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Error: PyMuPDF is not installed. Install it with: pip install PyMuPDF")
        sys.exit(1)
    
    try:
        print(f"Extracting images from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        extracted_images = []
        image_index = 0
        
        # Create output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
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
                    
                    # Generate unique filename
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    image_filename = f"image_p{page_num + 1}_i{img_idx + 1}_{image_hash}.{image_ext}"
                    
                    # Save image if output directory is specified
                    image_path = None
                    if output_dir:
                        image_path = output_dir / image_filename
                        with open(image_path, 'wb') as img_file:
                            img_file.write(image_bytes)
                    
                    # Get image position on page (if available)
                    image_rects = page.get_image_rects(xref)
                    position = None
                    if image_rects:
                        rect = image_rects[0]  # First occurrence
                        position = {
                            "x0": round(rect.x0, 2),
                            "y0": round(rect.y0, 2),
                            "x1": round(rect.x1, 2),
                            "y1": round(rect.y1, 2)
                        }
                    
                    # Classify image type
                    image_type = classify_image_type(image_filename, image_width, image_height)
                    
                    # Extract OCR text if requested
                    extracted_text = None
                    if extract_ocr and image_path:
                        extracted_text = extract_text_from_image(str(image_path), verbose=True)
                    
                    # Generate embeddings if requested
                    embedding_path = None
                    embedding_dimension = None
                    if generate_embeddings and image_path:
                        try:
                            embedding = generate_image_embedding(str(image_path), model_type=embedding_model, verbose=True)
                            if embedding is not None:
                                # Save embedding to file
                                embedding_base_path = image_path.with_suffix('')
                                save_embedding(embedding, embedding_base_path, model_type=embedding_model, verbose=True)
                                embedding_path = str(embedding_base_path.with_suffix('.npy'))
                                embedding_dimension = len(embedding)
                        except Exception as e:
                            print(f"    Warning: Failed to generate embedding for {image_filename}: {e}")
                    
                    image_index += 1
                    extracted_images.append({
                        'image_index': image_index,
                        'page': page_num + 1,
                        'filename': image_filename,
                        'image_path': str(image_path) if image_path else None,
                        'width': image_width,
                        'height': image_height,
                        'format': image_ext,
                        'size_bytes': len(image_bytes),
                        'image_type': image_type,
                        'position': position,
                        'extracted_text': extracted_text,
                        'embedding_path': embedding_path,
                        'embedding_dimension': embedding_dimension,
                        'embedding_model': f'timm/vit_large_patch16_siglip_384' if (embedding_path and embedding_model == 'siglip') else ('sentence-transformers/clip-ViT-L-14' if embedding_path else None),
                        'metadata': {
                            'xref': xref,
                            'colorspace': base_image.get("colorspace", "unknown"),
                            'bpc': base_image.get("bpc", 0),  # bits per component
                        }
                    })
                    
                except Exception as e:
                    print(f"  Warning: Failed to extract image {img_idx + 1} from page {page_num + 1}: {e}")
                    continue
        
        doc.close()
        
        if extracted_images:
            print(f"  Found {len(extracted_images)} image(s)")
        else:
            print("  Warning: No images found in PDF")
        
        return extracted_images
        
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
        return []


def extract_images_from_docx(docx_path: str, output_dir: Optional[Path] = None, extract_ocr: bool = False, generate_embeddings: bool = False, embedding_model: str = 'siglip') -> List[Dict[str, Any]]:
    """Extract images from DOCX using python-docx."""
    try:
        from docx import Document
        from docx.document import Document as DocxDocument
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        print("Error: python-docx is not installed. Install it with: pip install python-docx")
        sys.exit(1)
    
    try:
        print(f"Extracting images from DOCX: {docx_path}")
        doc = Document(docx_path)
        
        extracted_images = []
        image_index = 0
        
        # Create output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract images from document relationships
        # python-docx stores images in document.part.related_parts
        # We need to access the underlying XML to get images
        try:
            from docx.oxml import parse_xml
            from docx.oxml.ns import qn
            
            # Get all image parts
            image_parts = []
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_parts.append(rel)
            
            # Extract images from paragraphs and tables
            for element in doc.element.body:
                if element.tag.endswith('p'):  # Paragraph
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
                                        try:
                                            from PIL import Image
                                            import io
                                            img = Image.open(io.BytesIO(image_bytes))
                                            image_width, image_height = img.size
                                        except Exception:
                                            # Fallback if PIL fails
                                            image_width = 0
                                            image_height = 0
                                        
                                        # Generate unique filename
                                        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                                        image_filename = f"image_i{image_index + 1}_{image_hash}.{image_ext}"
                                        
                                        # Save image if output directory is specified
                                        image_path = None
                                        if output_dir:
                                            image_path = output_dir / image_filename
                                            with open(image_path, 'wb') as img_file:
                                                img_file.write(image_bytes)
                                        
                                        # Classify image type
                                        image_type = classify_image_type(image_filename, image_width, image_height)
                                        
                                        # Extract OCR text if requested
                                        extracted_text = None
                                        if extract_ocr and image_path:
                                            extracted_text = extract_text_from_image(str(image_path), verbose=True)
                                        
                                        # Generate embeddings if requested
                                        embedding_path = None
                                        embedding_dimension = None
                                        if generate_embeddings and image_path:
                                            try:
                                                embedding = generate_image_embedding(str(image_path), model_type=embedding_model, verbose=True)
                                                if embedding is not None:
                                                    embedding_base_path = image_path.with_suffix('')
                                                    save_embedding(embedding, embedding_base_path, model_type=embedding_model, verbose=True)
                                                    embedding_path = str(embedding_base_path.with_suffix('.npy'))
                                                    embedding_dimension = len(embedding)
                                            except Exception as e:
                                                print(f"    Warning: Failed to generate embedding for {image_filename}: {e}")
                                        
                                        image_index += 1
                                        extracted_images.append({
                                            'image_index': image_index,
                                            'page': None,  # DOCX doesn't have pages in the same way
                                            'filename': image_filename,
                                            'image_path': str(image_path) if image_path else None,
                                            'width': image_width,
                                            'height': image_height,
                                            'format': image_ext,
                                            'size_bytes': len(image_bytes),
                                            'image_type': image_type,
                                            'position': None,  # DOCX position is more complex
                                            'extracted_text': extracted_text,
                                            'embedding_path': embedding_path,
                                            'embedding_dimension': embedding_dimension,
                                            'embedding_model': f'timm/vit_large_patch16_siglip_384' if (embedding_path and embedding_model == 'siglip') else ('sentence-transformers/clip-ViT-L-14' if embedding_path else None),
                                            'metadata': {
                                                'content_type': content_type,
                                                'relationship_id': rId,
                                            }
                                        })
                                    except Exception as e:
                                        print(f"  Warning: Failed to extract image from relationship {rId}: {e}")
                                        continue
        except Exception as e:
            print(f"  Warning: Advanced DOCX image extraction failed, trying simpler method: {e}")
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
                            try:
                                from PIL import Image
                                import io
                                img = Image.open(io.BytesIO(image_bytes))
                                image_width, image_height = img.size
                            except Exception:
                                image_width = 0
                                image_height = 0
                            
                            # Generate filename
                            image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                            image_filename = f"image_i{image_index + 1}_{image_hash}.{image_ext}"
                            
                            # Save image
                            image_path = None
                            if output_dir:
                                image_path = output_dir / image_filename
                                with open(image_path, 'wb') as img_file:
                                    img_file.write(image_bytes)
                            
                            image_type = classify_image_type(image_filename, image_width, image_height)
                            
                            extracted_text = None
                            if extract_ocr and image_path:
                                extracted_text = extract_text_from_image(str(image_path), verbose=True)
                            
                            # Generate embeddings if requested
                            embedding_path = None
                            embedding_dimension = None
                            if generate_embeddings and image_path:
                                try:
                                    embedding = generate_image_embedding(str(image_path), model_type=embedding_model, verbose=True)
                                    if embedding is not None:
                                        embedding_base_path = image_path.with_suffix('')
                                        save_embedding(embedding, embedding_base_path, model_type=embedding_model, verbose=True)
                                        embedding_path = str(embedding_base_path.with_suffix('.npy'))
                                        embedding_dimension = len(embedding)
                                except Exception as e:
                                    print(f"    Warning: Failed to generate embedding for {image_filename}: {e}")
                            
                            image_index += 1
                            extracted_images.append({
                                'image_index': image_index,
                                'page': None,
                                'filename': image_filename,
                                'image_path': str(image_path) if image_path else None,
                                'width': image_width,
                                'height': image_height,
                                'format': image_ext,
                                'size_bytes': len(image_bytes),
                                'image_type': image_type,
                                'position': None,
                                'extracted_text': extracted_text,
                                'embedding_path': embedding_path,
                                'embedding_dimension': embedding_dimension,
                                'embedding_model': f'timm/vit_large_patch16_siglip_384' if (embedding_path and embedding_model == 'siglip') else ('sentence-transformers/clip-ViT-L-14' if embedding_path else None),
                                'metadata': {
                                    'content_type': content_type,
                                    'relationship_id': rel_id,
                                }
                            })
                        except Exception as e:
                            print(f"  Warning: Failed to extract image from relationship: {e}")
                            continue
            except Exception as e:
                print(f"  Error in fallback extraction: {e}")
        
        if extracted_images:
            print(f"  Found {len(extracted_images)} image(s)")
        else:
            print("  Warning: No images found in DOCX")
        
        return extracted_images
        
    except Exception as e:
        print(f"Error extracting images from DOCX: {e}")
        return []


def format_image_markdown(image: Dict[str, Any]) -> str:
    """Format an image entry as markdown."""
    markdown_lines = []
    markdown_lines.append(f"\n## Image {image['image_index']}")
    
    metadata = []
    if image.get('page'):
        metadata.append(f"Page: {image['page']}")
    if image.get('image_type'):
        metadata.append(f"Type: {image['image_type']}")
    if image.get('width') and image.get('height'):
        metadata.append(f"Dimensions: {image['width']}x{image['height']}")
    if image.get('format'):
        metadata.append(f"Format: {image['format']}")
    if image.get('size_bytes'):
        size_kb = image['size_bytes'] / 1024
        metadata.append(f"Size: {size_kb:.2f} KB")
    
    if metadata:
        markdown_lines.append(" | ".join(metadata))
    
    markdown_lines.append("")
    
    if image.get('image_path'):
        markdown_lines.append(f"**File:** `{image['filename']}`")
        markdown_lines.append(f"**Path:** `{image['image_path']}`")
    
    if image.get('position'):
        pos = image['position']
        markdown_lines.append(f"**Position:** ({pos.get('x0', 'N/A')}, {pos.get('y0', 'N/A')}) to ({pos.get('x1', 'N/A')}, {pos.get('y1', 'N/A')})")
    
    if image.get('extracted_text'):
        markdown_lines.append("")
        markdown_lines.append("**Extracted Text (OCR):**")
        markdown_lines.append(f"```\n{image['extracted_text']}\n```")
    
    if image.get('embedding_path'):
        markdown_lines.append("")
        markdown_lines.append("**Embedding:**")
        markdown_lines.append(f"- Model: {image.get('embedding_model', 'N/A')}")
        markdown_lines.append(f"- Dimensions: {image.get('embedding_dimension', 'N/A')}")
        markdown_lines.append(f"- Path: `{image['embedding_path']}`")
    
    return "\n".join(markdown_lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description='Extract images from PDF and DOCX files'
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the PDF or DOCX file'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file path (JSON format). If not specified, prints to stdout'
    )
    parser.add_argument(
        '--output-dir',
        '-d',
        type=str,
        help='Directory to save extracted images. If not specified, images are not saved'
    )
    parser.add_argument(
        '--format',
        '-f',
        choices=['json', 'markdown'],
        default='json',
        help='Output format: json or markdown (default: json)'
    )
    parser.add_argument(
        '--ocr',
        action='store_true',
        help='Extract text from images using OCR (requires easyocr)'
    )
    parser.add_argument(
        '--embed',
        action='store_true',
        help='Generate image embeddings (requires timm for SigLIP or sentence-transformers for CLIP)'
    )
    parser.add_argument(
        '--embedding-model',
        choices=['siglip', 'clip'],
        default='siglip',
        help='Embedding model to use: siglip (default) or clip'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Prepare output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Images will be saved to: {output_dir}")
    
    # Determine file type and extract images
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        images = extract_images_from_pdf(str(file_path), output_dir, extract_ocr=args.ocr, generate_embeddings=args.embed, embedding_model=args.embedding_model)
    elif file_ext in ['.docx', '.doc']:
        images = extract_images_from_docx(str(file_path), output_dir, extract_ocr=args.ocr, generate_embeddings=args.embed, embedding_model=args.embedding_model)
    else:
        print(f"Error: Unsupported file type: {file_ext}")
        print("Supported formats: .pdf, .docx")
        sys.exit(1)
    
    # Output results
    if args.format == 'json':
        output = json.dumps({
            'file_path': str(file_path),
            'file_type': file_ext,
            'image_count': len(images),
            'extracted_at': datetime.now().isoformat(),
            'images': images
        }, indent=2, ensure_ascii=False)
    else:  # markdown
        output_lines = [
            f"# Images extracted from: {file_path.name}",
            f"**File type:** {file_ext}",
            f"**Total images:** {len(images)}",
            f"**Extracted at:** {datetime.now().isoformat()}\n"
        ]
        
        for image in images:
            output_lines.append(format_image_markdown(image))
        
        output = "\n".join(output_lines)
    
    # Write to file or stdout
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "="*80)
        print(output)
    
    print(f"\nExtracted {len(images)} image(s) from {file_path.name}")
    if output_dir:
        print(f"Images saved to: {output_dir}")


if __name__ == '__main__':
    main()

