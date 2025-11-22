"""
Text chunking service.

Implements semantic chunking with overlap for optimal retrieval.
Uses langchain's RecursiveCharacterTextSplitter for structure preservation.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class ChunkingError(BaseAppException):
    """Raised when chunking fails."""
    pass


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.
    
    Attributes:
        text: The chunk text content
        chunk_index: Index of the chunk within the document (0-based)
        chunk_type: Type of chunk (default: 'text')
        start_char_index: Character index where chunk starts in original text
        end_char_index: Character index where chunk ends in original text
        token_count: Estimated token count (approximate)
        metadata: Additional metadata (page_number, section, etc.)
        created_at: Timestamp when chunk was created
    """
    
    text: str
    chunk_index: int
    chunk_type: str = "text"
    start_char_index: int = 0
    end_char_index: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count == 0 and self.text:
            self.token_count = self._estimate_token_count(self.text)
    
    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """
        Estimate token count using a simple heuristic.
        
        Approximate: ~4 characters per token (including spaces).
        More accurate methods could use tiktoken or transformers tokenizer.
        
        Args:
            text: Text to estimate tokens for
        
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        # This is a rough estimate, actual tokens may vary
        char_count = len(text)
        return max(1, char_count // 4)


class TextChunker:
    """
    Service for chunking text documents with overlap and structure preservation.
    
    Features:
    - Semantic chunking with overlap
    - Preserves document structure (paragraphs, headers, sections)
    - Configurable chunk size and overlap
    - Token-aware chunking (approximate)
    """
    
    # Default configuration matching Phase 1 requirements
    DEFAULT_CHUNK_SIZE = 800  # Target: 500-1000 tokens (using middle value)
    DEFAULT_CHUNK_OVERLAP = 150  # Target: 100-200 tokens (using middle value)
    DEFAULT_CHUNK_SIZE_CHARS = DEFAULT_CHUNK_SIZE * 4  # Convert tokens to chars (~4 chars/token)
    DEFAULT_CHUNK_OVERLAP_CHARS = DEFAULT_CHUNK_OVERLAP * 4
    
    # Separators for recursive splitting (order matters!)
    # More specific separators come first
    SEPARATORS = [
        "\n\n\n",  # Paragraph breaks (multiple newlines)
        "\n\n",    # Paragraph breaks (double newlines)
        "\n",      # Single line breaks
        ". ",      # Sentence endings
        "! ",      # Exclamation endings
        "? ",      # Question endings
        "; ",      # Semicolons
        ", ",      # Commas
        " ",       # Spaces
        "",        # Characters (fallback)
    ]
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 800)
            chunk_overlap: Overlap size in tokens (default: 150)
            separators: Custom list of separators (default: uses predefined)
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain is not installed. Install it with: pip install langchain"
            )
        
        # Convert token targets to character counts (approximate)
        self.chunk_size_tokens = chunk_size
        self.chunk_overlap_tokens = chunk_overlap
        self.chunk_size_chars = chunk_size * 4  # ~4 chars per token
        self.chunk_overlap_chars = chunk_overlap * 4
        
        # Use custom separators or default
        self.separators = separators or self.SEPARATORS
        
        # Initialize langchain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size_chars,
            chunk_overlap=self.chunk_overlap_chars,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.debug(
            f"Initialized TextChunker: chunk_size={chunk_size} tokens "
            f"({self.chunk_size_chars} chars), overlap={chunk_overlap} tokens "
            f"({self.chunk_overlap_chars} chars)"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        preserve_structure: bool = True,
    ) -> List[Chunk]:
        """
        Chunk text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to all chunks
            preserve_structure: Whether to preserve document structure (default: True)
        
        Returns:
            List of Chunk objects
        
        Raises:
            ChunkingError: If chunking fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            # Normalize text for better chunking
            if preserve_structure:
                text = self._normalize_text(text)
            
            # Split text using langchain splitter
            logger.debug(f"Chunking text: {len(text)} characters")
            split_texts = self.splitter.split_text(text)
            logger.info(f"Split text into {len(split_texts)} chunks")
            
            # Create Chunk objects with metadata
            chunks = []
            current_index = 0
            
            for idx, chunk_text in enumerate(split_texts):
                if not chunk_text.strip():
                    # Skip empty chunks
                    continue
                
                # Find position in original text
                start_char = text.find(chunk_text, current_index)
                if start_char == -1:
                    # Fallback: estimate position
                    start_char = current_index
                end_char = start_char + len(chunk_text)
                current_index = end_char
                
                # Create chunk metadata
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": idx,
                    "chunk_size_chars": len(chunk_text),
                    "preserved_structure": preserve_structure,
                }
                
                # Estimate token count
                token_count = Chunk._estimate_token_count(chunk_text)
                
                chunk = Chunk(
                    text=chunk_text.strip(),
                    chunk_index=idx,
                    chunk_type="text",
                    start_char_index=start_char,
                    end_char_index=end_char,
                    token_count=token_count,
                    metadata=chunk_metadata,
                )
                
                chunks.append(chunk)
            
            logger.info(
                f"Created {len(chunks)} chunks with average size: "
                f"{sum(c.token_count for c in chunks) // max(1, len(chunks))} tokens"
            )
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}", exc_info=True)
            raise ChunkingError(
                f"Failed to chunk text: {str(e)}",
                {"error": str(e), "text_length": len(text) if text else 0},
            ) from e
    
    def chunk_document(
        self,
        text: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        preserve_structure: bool = True,
    ) -> List[Chunk]:
        """
        Chunk a document with document-level metadata.
        
        Convenience method that automatically includes document metadata.
        
        Args:
            text: Document text to chunk
            document_metadata: Document-level metadata (filename, type, etc.)
            preserve_structure: Whether to preserve document structure
        
        Returns:
            List of Chunk objects with document metadata
        """
        return self.chunk_text(
            text=text,
            metadata=document_metadata,
            preserve_structure=preserve_structure,
        )
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text while preserving structure.
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Preserve multiple newlines (structure markers)
        # Don't collapse all multiple newlines to single
        
        # Remove excessive whitespace within lines
        import re
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split("\n")
        normalized_lines = [re.sub(r"[ \t]+", " ", line) for line in lines]
        text = "\n".join(normalized_lines)
        
        # Strip leading/trailing whitespace but keep structure
        return text.strip()
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunks to analyze
        
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "average_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }
        
        token_counts = [chunk.token_count for chunk in chunks]
        total_tokens = sum(token_counts)
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "average_tokens": total_tokens // len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_characters": sum(len(chunk.text) for chunk in chunks),
        }
