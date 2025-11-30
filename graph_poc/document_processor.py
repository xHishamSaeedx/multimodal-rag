"""
Document processor for extracting text, sections, chunks, and entities.

This module processes documents and prepares data for graph storage.
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logging.warning("PyMuPDF not installed. PDF processing will not work.")

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section in a document."""
    section_id: str
    title: str
    index: int
    text: str
    chunks: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    entity_id: str
    entity_name: str
    entity_type: str
    confidence: float = 0.5
    context: Optional[str] = None
    entity_value: Optional[str] = None


class DocumentProcessor:
    """
    Processes documents to extract text, sections, chunks, and entities.
    """
    
    # Chunking configuration
    DEFAULT_CHUNK_SIZE = 800  # Target tokens (~3200 chars)
    DEFAULT_CHUNK_OVERLAP = 150  # Overlap tokens (~600 chars)
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap size in tokens
        """
        self.chunk_size_chars = chunk_size * 4  # ~4 chars per token
        self.chunk_overlap_chars = chunk_overlap * 4
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install pymupdf")
        
        try:
            doc = fitz.open(file_path)
            text_parts = []
            page_count = len(doc)
            file_size = Path(file_path).stat().st_size
            
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()
                text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            doc.close()
            
            metadata = {
                "page_count": page_count,
                "file_size": file_size,
            }
            
            logger.info(f"Extracted {len(full_text)} characters from {page_count} pages")
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def extract_text_from_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from various file types.
        
        Args:
            file_path: Path to file
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text, {"file_size": len(text.encode('utf-8'))}
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def identify_sections(self, text: str) -> List[DocumentSection]:
        """
        Identify sections in document by headings.
        
        Uses simple heuristics:
        - Lines that are all caps or start with numbers
        - Lines followed by blank lines
        - Lines shorter than 100 chars
        
        Args:
            text: Full document text
        
        Returns:
            List of DocumentSection objects
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        current_text = []
        section_index = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if line looks like a heading
            is_heading = (
                len(stripped) > 0 and
                len(stripped) < 100 and
                (
                    stripped.isupper() or
                    re.match(r'^\d+[\.\)]\s+', stripped) or  # Numbered heading
                    re.match(r'^[A-Z][A-Z\s]+$', stripped) or  # All caps
                    (i > 0 and lines[i-1].strip() == '' and i < len(lines) - 1 and lines[i+1].strip() == '')
                )
            )
            
            if is_heading:
                # Save previous section
                if current_section:
                    current_section.text = '\n'.join(current_text).strip()
                    sections.append(current_section)
                
                # Start new section
                section_id = f"sec_{uuid.uuid4().hex[:8]}"
                current_section = DocumentSection(
                    section_id=section_id,
                    title=stripped,
                    index=section_index,
                    text=""
                )
                section_index += 1
                current_text = []
            else:
                if current_section:
                    current_text.append(line)
                else:
                    # Text before first section
                    if not current_text:
                        current_section = DocumentSection(
                            section_id=f"sec_{uuid.uuid4().hex[:8]}",
                            title="Introduction",
                            index=0,
                            text=""
                        )
                        section_index = 1
                    current_text.append(line)
        
        # Save last section
        if current_section:
            current_section.text = '\n'.join(current_text).strip()
            sections.append(current_section)
        
        # If no sections found, create one
        if not sections:
            sections.append(DocumentSection(
                section_id=f"sec_{uuid.uuid4().hex[:8]}",
                title="Document",
                index=0,
                text=text
            ))
        
        logger.info(f"Identified {len(sections)} sections")
        return sections
    
    def chunk_text(self, text: str, chunk_index_start: int = 0) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            chunk_index_start: Starting index for chunks
        
        Returns:
            List of chunk dictionaries
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        chunk_index = chunk_index_start
        
        # Simple chunking: split by paragraphs first, then by size
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If adding this paragraph would exceed chunk size, save current chunk
            if current_size + para_size > self.chunk_size_chars and current_chunk:
                chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_type": "text",
                    "text": current_chunk.strip(),
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap_chars:] if len(current_chunk) > self.chunk_overlap_chars else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size = len(current_chunk)
        
        # Save last chunk
        if current_chunk.strip():
            chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_type": "text",
                "text": current_chunk.strip(),
            })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def extract_entities_simple(self, text: str) -> List[ExtractedEntity]:
        """
        Simple entity extraction using patterns.
        
        This is a basic implementation for POC. In production, use proper NER.
        
        Args:
            text: Text to extract entities from
        
        Returns:
            List of ExtractedEntity objects
        """
        entities = []
        entity_map = {}  # To deduplicate
        
        # Pattern 1: Capitalized phrases (potential organizations/concepts)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(org_pattern, text):
            name = match.group(1)
            if len(name) > 3 and name not in entity_map:
                entity_id = f"ent_{uuid.uuid4().hex[:8]}"
                entity = ExtractedEntity(
                    entity_id=entity_id,
                    entity_name=name,
                    entity_type="ORGANIZATION",
                    confidence=0.6
                )
                entities.append(entity)
                entity_map[name] = entity
        
        # Pattern 2: Numbers with units (metrics)
        metric_pattern = r'\b(\d+\.?\d*)\s*([%$€£¥]|million|billion|thousand|M|B|K)\b'
        for match in re.finditer(metric_pattern, text, re.IGNORECASE):
            value = match.group(1)
            unit = match.group(2)
            name = f"{value} {unit}"
            if name not in entity_map:
                entity_id = f"ent_{uuid.uuid4().hex[:8]}"
                entity = ExtractedEntity(
                    entity_id=entity_id,
                    entity_name=name,
                    entity_type="METRIC",
                    confidence=0.7,
                    entity_value=f"{value} {unit}"
                )
                entities.append(entity)
                entity_map[name] = entity
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and extract all components.
        
        Args:
            file_path: Path to document file
        
        Returns:
            Dictionary with document structure ready for graph storage
        """
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        text, metadata = self.extract_text_from_file(file_path)
        
        # Identify sections
        sections = self.identify_sections(text)
        
        # Chunk each section
        global_chunk_index = 0
        all_entities = []
        
        for section in sections:
            chunks = self.chunk_text(section.text, chunk_index_start=global_chunk_index)
            section.chunks = chunks
            global_chunk_index += len(chunks)
            
            # Extract entities from section text
            section_entities = self.extract_entities_simple(section.text)
            all_entities.extend(section_entities)
        
        # Deduplicate entities by name
        unique_entities = {}
        for entity in all_entities:
            if entity.entity_name not in unique_entities:
                unique_entities[entity.entity_name] = entity
        
        # Build document structure
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        path_obj = Path(file_path)
        
        document_data = {
            "document_id": document_id,
            "title": path_obj.stem,
            "source": str(file_path),
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "index": section.index,
                    "chunks": section.chunks
                }
                for section in sections
            ],
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type,
                    "entity_value": getattr(entity, 'entity_value', None),
                    "confidence": entity.confidence,
                }
                for entity in unique_entities.values()
            ],
            "metadata": metadata
        }
        
        logger.info(f"Processed document: {len(sections)} sections, {sum(len(s.chunks) for s in sections)} chunks, {len(unique_entities)} entities")
        
        return document_data

