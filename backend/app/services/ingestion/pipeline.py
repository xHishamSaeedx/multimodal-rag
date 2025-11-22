"""
Document ingestion pipeline.

Orchestrates the complete ingestion flow:
Raw Document → Extraction → Chunking → Storage → Indexing
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import UUID

from app.services.ingestion.extractor import TextExtractor, ExtractedContent
from app.services.ingestion.chunker import TextChunker, Chunk, ChunkingError
from app.services.embedding.text_embedder import TextEmbedder, EmbeddingError
from app.services.storage import MinIOStorage
from app.repositories.document_repository import DocumentRepository, RepositoryError
from app.repositories.vector_repository import VectorRepository, VectorRepositoryError
from app.utils.exceptions import (
    ExtractionError,
    StorageError,
)

logger = logging.getLogger(__name__)


class IngestionPipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


class IngestionPipeline:
    """
    Complete document ingestion pipeline.
    
    Orchestrates:
    1. Store raw document in MinIO (data lake)
    2. Extract text from document
    3. Chunk extracted text
    4. Store document and chunks in Supabase
    5. Generate embeddings for chunks
    6. Store embeddings in Qdrant (Vector DB)
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 800)
            chunk_overlap: Overlap size in tokens (default: 150)
        """
        self.extractor = TextExtractor()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedder = TextEmbedder()  # Initialize embedding service first
        
        # Initialize vector repository with embedding dimension from embedder
        # This ensures Qdrant collection matches the actual model dimensions
        self.vector_repo = VectorRepository(
            vector_size=self.embedder.embedding_dim,  # Use actual embedding dimension
        )
        
        self.storage = MinIOStorage()
        self.repository = DocumentRepository()
        
        logger.info(
            f"Initialized IngestionPipeline: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}, "
            f"embedding_model={self.embedder.model_name}, "
            f"vector_dim={self.embedder.embedding_dim}"
        )
    
    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        source: str = "api",
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            source: Source identifier (default: "api")
            document_metadata: Optional additional document metadata
        
        Returns:
            Dictionary containing:
            - document_id: UUID of created document
            - object_key: MinIO object key
            - chunks_count: Number of chunks created
            - chunk_ids: List of chunk UUIDs
            - extracted_content: ExtractedContent object
            - stats: Chunking statistics
        
        Raises:
            IngestionPipelineError: If pipeline execution fails
        """
        try:
            # Step 1: Store raw document in MinIO
            logger.info(f"Step 1: Storing raw document in MinIO: {filename}")
            file_type = self.extractor._infer_file_type(Path(filename))
            
            object_key = self.storage.upload_raw_document(
                file_bytes=file_bytes,
                filename=filename,
                source=source,
                document_type=file_type,
                add_timestamp=True,
            )
            logger.info(f"✓ Stored raw document: {object_key}")
            
            # Step 2: Extract text
            logger.info(f"Step 2: Extracting text from: {filename}")
            extracted_content = self.extractor.extract_from_bytes(
                file_bytes=file_bytes,
                file_name=filename,
            )
            logger.info(
                f"✓ Extracted {len(extracted_content.text)} characters "
                f"({extracted_content.page_count or 'N/A'} pages)"
            )
            
            # Step 3: Chunk text
            logger.info(f"Step 3: Chunking extracted text")
            
            # Prepare document metadata for chunks
            chunk_metadata = {
                "filename": filename,
                "document_type": file_type,
                "source": source,
                "source_path": object_key,
                "page_count": extracted_content.page_count,
                "extracted_at": extracted_content.extracted_at.isoformat(),
                **(extracted_content.metadata or {}),
                **(document_metadata or {}),
            }
            
            chunks = self.chunker.chunk_document(
                text=extracted_content.text,
                document_metadata=chunk_metadata,
                preserve_structure=True,
            )
            
            stats = self.chunker.get_chunk_statistics(chunks)
            logger.info(
                f"✓ Created {len(chunks)} chunks "
                f"(avg: {stats['average_tokens']} tokens/chunk)"
            )
            
            # Step 4: Store document and chunks in Supabase
            logger.info(f"Step 4: Storing document and chunks in database")
            
            document_id = self.repository.create_document(
                source_path=object_key,
                filename=filename,
                document_type=file_type,
                extracted_text=extracted_content.text,
                metadata={
                    "source": source,
                    "file_size": extracted_content.file_size,
                    "page_count": extracted_content.page_count,
                    "extracted_at": extracted_content.extracted_at.isoformat(),
                    "chunking_stats": stats,
                    **(extracted_content.metadata or {}),
                    **(document_metadata or {}),
                },
            )
            logger.info(f"✓ Created document record: {document_id}")
            
            chunk_ids = self.repository.create_chunks(
                document_id=document_id,
                chunks=chunks,
            )
            logger.info(f"✓ Created {len(chunk_ids)} chunk records")
            
            # Step 5: Generate embeddings for chunks
            logger.info(f"Step 5: Generating embeddings for {len(chunks)} chunks")
            
            # Extract chunk texts for embedding
            chunk_texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = self.embedder.embed_batch(
                texts=chunk_texts,
                show_progress=True,
            )
            logger.info(
                f"✓ Generated {len(embeddings)} embeddings "
                f"(dimension: {self.embedder.embedding_dim})"
            )
            
            # Step 6: Store embeddings in Qdrant
            logger.info(f"Step 6: Storing embeddings in Qdrant")
            
            # Prepare payloads for Qdrant (metadata for each vector)
            payloads = []
            for chunk, chunk_id in zip(chunks, chunk_ids):
                payload = {
                    "chunk_id": str(chunk_id),
                    "document_id": str(document_id),
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "filename": filename,
                    "document_type": file_type,
                    "source": source,
                    "source_path": object_key,
                    **chunk.metadata,
                }
                payloads.append(payload)
            
            # Store vectors in Qdrant
            self.vector_repo.store_vectors(
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                payloads=payloads,
            )
            logger.info(f"✓ Stored {len(embeddings)} embeddings in Qdrant")
            
            logger.info(
                f"Pipeline complete: Document {document_id} ingested with "
                f"{len(chunks)} chunks and embeddings stored in Qdrant"
            )
            
            return {
                "document_id": str(document_id),
                "object_key": object_key,
                "chunks_count": len(chunks),
                "chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
                "embeddings_count": len(embeddings),
                "extracted_content": extracted_content,
                "stats": stats,
            }
        
        except (ExtractionError, StorageError, ChunkingError, RepositoryError, EmbeddingError, VectorRepositoryError) as e:
            logger.error(f"Pipeline error: {e.message}", exc_info=True)
            raise IngestionPipelineError(
                f"Ingestion pipeline failed: {e.message}",
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            raise IngestionPipelineError(
                f"Unexpected error in ingestion pipeline: {str(e)}",
            ) from e
