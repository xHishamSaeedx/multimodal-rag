"""
Document ingestion pipeline.

Orchestrates the complete ingestion flow:
Raw Document → Extraction → Chunking → Storage → Indexing
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from app.services.ingestion.extractor import TextExtractor, ExtractedContent
from app.services.ingestion.chunker import TextChunker, Chunk, ChunkingError
from app.services.ingestion.extraction_runner import ExtractionRunner
from app.services.ingestion.table_processor import TableProcessor, ProcessedTable
from app.services.embedding.text_embedder import TextEmbedder, EmbeddingError
from app.services.storage import MinIOStorage
from app.repositories.document_repository import DocumentRepository, RepositoryError
from app.repositories.vector_repository import VectorRepository, VectorRepositoryError
from app.repositories.sparse_repository import SparseRepository, SparseRepositoryError
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
    7. Index chunks in Elasticsearch (BM25 sparse index)
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
        self.extractor = TextExtractor()  # Keep for backward compatibility
        self.extraction_runner = ExtractionRunner(enable_deduplication=True)
        self.table_processor = TableProcessor()
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
        
        # Initialize sparse repository for BM25 indexing
        self.sparse_repo = SparseRepository()
        
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
            
            # Step 2: Extract text and tables in parallel
            logger.info(f"Step 2: Extracting text and tables from: {filename}")
            extracted_content, extracted_tables = self.extraction_runner.extract_parallel_from_bytes(
                file_bytes=file_bytes,
                file_name=filename,
                file_type=file_type,
            )
            logger.info(
                f"✓ Extracted {len(extracted_content.text)} characters "
                f"({extracted_content.page_count or 'N/A'} pages), "
                f"{len(extracted_tables)} table(s)"
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
            
            # Step 4.5: Process and store tables
            table_chunk_ids = []
            processed_tables = []
            if extracted_tables:
                logger.info(f"Step 4.5: Processing and storing {len(extracted_tables)} table(s)")
                
                # Process tables (convert to JSON, markdown, flattened text)
                processed_tables = [
                    self.table_processor.process_table(table) for table in extracted_tables
                ]
                logger.info(f"✓ Processed {len(processed_tables)} table(s)")
                
                # Create table chunks (one chunk per table)
                table_chunks = []
                for i, processed_table in enumerate(processed_tables):
                    table_chunk = Chunk(
                        text=processed_table.table_text,  # Flattened text for embedding
                        chunk_index=len(chunks) + i,  # Continue chunk index after text chunks
                        chunk_type="table",
                        start_char_index=0,
                        end_char_index=len(processed_table.table_text),
                        token_count=Chunk._estimate_token_count(processed_table.table_text),
                        metadata={
                            **chunk_metadata,
                            "table_index": processed_table.table_index,
                            "page": processed_table.page,
                            "row_count": processed_table.metadata.get("row_count", 0),
                            "col_count": processed_table.metadata.get("col_count", 0),
                            "headers": processed_table.metadata.get("headers", []),
                        },
                        created_at=datetime.utcnow(),
                    )
                    table_chunks.append(table_chunk)
                
                # Store table chunks in database
                table_chunk_ids = self.repository.create_chunks(
                    document_id=document_id,
                    chunks=table_chunks,
                )
                logger.info(f"✓ Created {len(table_chunk_ids)} table chunk records")
                
                # Update chunks with table_data and embedding_type
                # Note: We need to update the chunks we just created with table_data
                # This is a limitation - we should ideally pass table_data during creation
                # For now, we'll store it in the tables table and link via chunk_id
                
                # Store tables in tables table
                tables_data = []
                for processed_table, table_chunk_id in zip(processed_tables, table_chunk_ids):
                    tables_data.append({
                        "chunk_id": table_chunk_id,
                        "table_data": processed_table.table_data,
                        "table_markdown": processed_table.table_markdown,
                        "table_text": processed_table.table_text,
                        "metadata": processed_table.metadata,
                    })
                
                table_ids = self.repository.create_tables_batch(
                    document_id=document_id,
                    tables_data=tables_data,
                )
                logger.info(f"✓ Stored {len(table_ids)} table(s) in tables table")
            
            # Step 5: Generate embeddings for text chunks
            logger.info(f"Step 5: Generating embeddings for {len(chunks)} text chunks")
            
            # Extract chunk texts for embedding
            chunk_texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batch
            text_embeddings = self.embedder.embed_batch(
                texts=chunk_texts,
                show_progress=True,
            )
            logger.info(
                f"✓ Generated {len(text_embeddings)} text embeddings "
                f"(dimension: {self.embedder.embedding_dim})"
            )
            
            # Step 5.5: Generate embeddings for table chunks
            table_embeddings = []
            if processed_tables:
                logger.info(f"Step 5.5: Generating embeddings for {len(processed_tables)} table chunk(s)")
                
                # Extract flattened table texts for embedding
                table_texts = [processed_table.table_text for processed_table in processed_tables]
                
                # Generate embeddings in batch (same model as text)
                table_embeddings = self.embedder.embed_batch(
                    texts=table_texts,
                    show_progress=True,
                )
                logger.info(
                    f"✓ Generated {len(table_embeddings)} table embeddings "
                    f"(dimension: {self.embedder.embedding_dim})"
                )
            
            # Step 6: Store text embeddings in Qdrant
            logger.info(f"Step 6: Storing text embeddings in Qdrant")
            
            # Prepare payloads for Qdrant (metadata for each vector)
            text_payloads = []
            for chunk, chunk_id in zip(chunks, chunk_ids):
                payload = {
                    "chunk_id": str(chunk_id),
                    "document_id": str(document_id),
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "embedding_type": "text",
                    "filename": filename,
                    "document_type": file_type,
                    "source": source,
                    "source_path": object_key,
                    **chunk.metadata,
                }
                text_payloads.append(payload)
            
            # Store text vectors in Qdrant
            self.vector_repo.store_vectors(
                chunk_ids=chunk_ids,
                embeddings=text_embeddings,
                payloads=text_payloads,
            )
            logger.info(f"✓ Stored {len(text_embeddings)} text embeddings in Qdrant")
            
            # Step 6.5: Store table embeddings in Qdrant (table_chunks collection)
            if processed_tables and table_embeddings:
                logger.info(f"Step 6.5: Storing table embeddings in Qdrant (table_chunks collection)")
                
                # Prepare payloads for table chunks
                table_payloads = []
                for processed_table, table_chunk_id in zip(processed_tables, table_chunk_ids):
                    payload = {
                        "chunk_id": str(table_chunk_id),
                        "document_id": str(document_id),
                        "text": processed_table.table_text,  # Flattened text
                        "table_data": processed_table.table_data,  # JSON format
                        "table_markdown": processed_table.table_markdown,  # Markdown format
                        "chunk_index": len(chunks) + processed_tables.index(processed_table),
                        "chunk_type": "table",
                        "embedding_type": "table",
                        "filename": filename,
                        "document_type": file_type,
                        "source": source,
                        "source_path": object_key,
                        **processed_table.metadata,
                    }
                    table_payloads.append(payload)
                
                # Store table vectors in Qdrant table_chunks collection
                self.vector_repo.store_table_vectors(
                    chunk_ids=table_chunk_ids,
                    embeddings=table_embeddings,
                    payloads=table_payloads,
                )
                logger.info(f"✓ Stored {len(table_embeddings)} table embeddings in Qdrant (table_chunks)")
            
            # Step 7: Index text chunks in Elasticsearch (BM25)
            logger.info(f"Step 7: Indexing text chunks in Elasticsearch (BM25)")
            
            # Prepare data for BM25 indexing
            chunk_texts = [chunk.text for chunk in chunks]
            filenames = [filename] * len(chunks)
            document_types = [file_type] * len(chunks)
            source_paths = [object_key] * len(chunks)
            created_at_list = [chunk.created_at for chunk in chunks]
            
            # Index text chunks in Elasticsearch
            text_chunk_types = ["text"] * len(chunks)
            text_embedding_types = ["text"] * len(chunks)
            self.sparse_repo.index_chunks(
                chunk_ids=chunk_ids,
                chunk_texts=chunk_texts,
                document_ids=[document_id] * len(chunks),
                filenames=filenames,
                document_types=document_types,
                source_paths=source_paths,
                metadata_list=[chunk.metadata for chunk in chunks],
                created_at_list=created_at_list,
                chunk_types=text_chunk_types,
                embedding_types=text_embedding_types,
            )
            logger.info(f"✓ Indexed {len(chunks)} text chunks in Elasticsearch (BM25)")
            
            # Step 7.5: Index table chunks in Elasticsearch (BM25)
            if processed_tables:
                logger.info(f"Step 7.5: Indexing table chunks in Elasticsearch (BM25)")
                
                # Use table_markdown for searchable text (better than flattened text)
                table_chunk_texts = [processed_table.table_markdown for processed_table in processed_tables]
                table_filenames = [filename] * len(processed_tables)
                table_document_types = [file_type] * len(processed_tables)
                table_source_paths = [object_key] * len(processed_tables)
                table_created_at_list = [datetime.utcnow()] * len(processed_tables)
                table_chunk_types = ["table"] * len(processed_tables)
                table_embedding_types = ["table"] * len(processed_tables)
                table_markdown_list = [processed_table.table_markdown for processed_table in processed_tables]
                
                # Prepare metadata with table-specific fields
                table_metadata_list = []
                for processed_table in processed_tables:
                    table_metadata = {
                        **chunk_metadata,
                        "table_index": processed_table.table_index,
                        "page": processed_table.page,
                        "row_count": processed_table.metadata.get("row_count", 0),
                        "col_count": processed_table.metadata.get("col_count", 0),
                        "headers": processed_table.metadata.get("headers", []),
                    }
                    table_metadata_list.append(table_metadata)
                
                # Index table chunks in Elasticsearch
                self.sparse_repo.index_chunks(
                    chunk_ids=table_chunk_ids,
                    chunk_texts=table_chunk_texts,
                    document_ids=[document_id] * len(processed_tables),
                    filenames=table_filenames,
                    document_types=table_document_types,
                    source_paths=table_source_paths,
                    metadata_list=table_metadata_list,
                    created_at_list=table_created_at_list,
                    chunk_types=table_chunk_types,
                    embedding_types=table_embedding_types,
                    table_markdown_list=table_markdown_list,
                )
                logger.info(f"✓ Indexed {len(processed_tables)} table chunks in Elasticsearch (BM25)")
            
            total_chunks = len(chunks) + len(processed_tables)
            logger.info(
                f"Pipeline complete: Document {document_id} ingested with "
                f"{len(chunks)} text chunks, {len(processed_tables)} table chunks, "
                f"embeddings stored in Qdrant, and BM25 index in Elasticsearch"
            )
            
            return {
                "document_id": str(document_id),
                "object_key": object_key,
                "chunks_count": len(chunks),
                "table_chunks_count": len(processed_tables),
                "total_chunks_count": total_chunks,
                "chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
                "table_chunk_ids": [str(chunk_id) for chunk_id in table_chunk_ids],
                "text_embeddings_count": len(text_embeddings),
                "table_embeddings_count": len(table_embeddings),
                "extracted_content": extracted_content,
                "extracted_tables": extracted_tables,
                "stats": stats,
            }
        
        except (ExtractionError, StorageError, ChunkingError, RepositoryError, EmbeddingError, VectorRepositoryError, SparseRepositoryError) as e:
            logger.error(f"Pipeline error: {e.message}", exc_info=True)
            raise IngestionPipelineError(
                f"Ingestion pipeline failed: {e.message}",
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
            raise IngestionPipelineError(
                f"Unexpected error in ingestion pipeline: {str(e)}",
            ) from e
