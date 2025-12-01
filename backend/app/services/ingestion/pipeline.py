"""
Document ingestion pipeline.

Orchestrates the complete ingestion flow:
Raw Document → Extraction → Chunking → Storage → Indexing
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime

from app.services.ingestion.extractor import TextExtractor, ExtractedContent
from app.utils.metrics import (
    text_embedding_duration_seconds,
    text_embeddings_generated_total,
    text_embedding_batch_size,
    image_embedding_duration_seconds,
    image_embeddings_generated_total,
    image_embedding_batch_size,
)
from app.services.ingestion.chunker import TextChunker, Chunk, ChunkingError
from app.services.ingestion.extraction_runner import ExtractionRunner
from app.services.ingestion.table_processor import TableProcessor, ProcessedTable
from app.services.ingestion.image_extractor import ExtractedImage
from app.services.embedding.text_embedder import TextEmbedder, EmbeddingError
from app.services.embedding.image_embedder import ImageEmbedder
from app.services.storage import MinIOStorage
from app.services.storage.supabase_storage import SupabaseImageStorage
from app.services.vision import VisionProcessorFactory
from app.repositories.document_repository import DocumentRepository, RepositoryError
from app.repositories.vector_repository import VectorRepository, VectorRepositoryError
from app.repositories.sparse_repository import SparseRepository, SparseRepositoryError
from app.repositories.graph_repository import GraphRepository
from app.core.config import settings
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
        text_embedder: Optional[TextEmbedder] = None,
        image_embedder: Optional[ImageEmbedder] = None,
        vision_processor: Optional[Any] = None,
        enable_text: bool = True,
        enable_tables: bool = True,
        enable_images: bool = True,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 800)
            chunk_overlap: Overlap size in tokens (default: 150)
            text_embedder: Optional pre-initialized TextEmbedder instance (default: None, creates new)
            image_embedder: Optional pre-initialized ImageEmbedder instance (default: None, creates new)
            vision_processor: Optional pre-initialized VisionProcessor instance (default: None, creates new)
            enable_text: Whether to process text (default: True)
            enable_tables: Whether to process tables (default: True)
            enable_images: Whether to process images (default: True)
        """
        self.enable_text = enable_text
        self.enable_tables = enable_tables
        self.enable_images = enable_images
        
        self.extractor = TextExtractor()  # Keep for backward compatibility
        self.extraction_runner = ExtractionRunner(
            extract_ocr=True,  # Enable OCR to extract text from chart images
            enable_text=enable_text,
            enable_tables=enable_tables,
            enable_image_extraction=enable_images,
        )
        self.table_processor = TableProcessor()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Use pre-initialized embedders if provided, otherwise create new instances
        self.embedder = text_embedder if text_embedder is not None else TextEmbedder()
        self.image_embedder = image_embedder if image_embedder is not None else ImageEmbedder(model_type="clip")
        
        # Initialize vector repository with embedding dimension from embedder
        # This ensures Qdrant collection matches the actual model dimensions
        self.vector_repo = VectorRepository(
            vector_size=self.embedder.embedding_dim,  # Use actual embedding dimension
        )
        
        # Initialize image vector repository (separate collection for images)
        # Note: This will need to be updated to support image_chunks collection
        # For now, we'll use a separate vector repo instance
        self.image_vector_repo = VectorRepository(
            vector_size=self.image_embedder.embedding_dim,
            collection_name="image_chunks",  # Different collection for images
        )
        
        # Initialize sparse repository for BM25 indexing
        self.sparse_repo = SparseRepository()
        
        # Use pre-initialized vision processor if provided, otherwise create new
        if vision_processor is not None:
            self.vision_processor = vision_processor
            logger.info("Using pre-initialized vision processor for image captioning during ingestion")
        else:
            # Initialize captioning processor (always generate captions during ingestion)
            # Vision LLM mode can still be used at query time separately
            try:
                self.vision_processor = VisionProcessorFactory.create_processor(mode="captioning")
                logger.info("Initialized captioning processor for image captioning during ingestion")
            except Exception as e:
                logger.warning(f"Failed to initialize captioning processor: {e}. Continuing without captioning.")
                self.vision_processor = None
        
        self.storage = MinIOStorage()
        self.image_storage = SupabaseImageStorage()
        self.repository = DocumentRepository()
        
        # Initialize graph repository if Neo4j is enabled
        self.graph_repo = None
        if settings.neo4j_enabled:
            try:
                self.graph_repo = GraphRepository()
                logger.info("Initialized graph repository for Neo4j")
            except Exception as e:
                logger.warning(f"Failed to initialize graph repository: {e}. Continuing without graph building.")
        
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
            
            # Step 2: Extract text, tables, and images in parallel
            logger.info(f"Step 2: Extracting text, tables, and images from: {filename}")
            extracted_content, extracted_tables, extracted_images = self.extraction_runner.extract_parallel_from_bytes(
                file_bytes=file_bytes,
                file_name=filename,
                file_type=file_type,
            )
            logger.info(
                f"✓ Extracted {len(extracted_content.text) if extracted_content else 0} characters "
                f"({extracted_content.page_count if extracted_content else 'N/A'} pages), "
                f"{len(extracted_tables)} table(s), "
                f"{len(extracted_images)} image(s)"
            )
            
            # Step 3: Chunk text (only if text processing is enabled)
            chunks = []
            stats = {}
            chunk_metadata = {
                "filename": filename,
                "document_type": file_type,
                "source": source,
                "source_path": object_key,
                "page_count": extracted_content.page_count if extracted_content else 0,
                "extracted_at": extracted_content.extracted_at.isoformat() if extracted_content else datetime.utcnow().isoformat(),
                **(extracted_content.metadata or {}),
                **(document_metadata or {}),
            }
            
            if self.enable_text and extracted_content and extracted_content.text:
                logger.info(f"Step 3: Chunking extracted text")
                
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
            else:
                logger.info(f"Step 3: Skipping text chunking (text processing disabled or no text)")
                stats = {"average_tokens": 0, "total_tokens": 0, "chunk_count": 0}
            
            # Step 4: Store document and chunks in Supabase
            logger.info(f"Step 4: Storing document and chunks in database")
            
            document_id = self.repository.create_document(
                source_path=object_key,
                filename=filename,
                document_type=file_type,
                extracted_text=extracted_content.text if extracted_content else "",
                metadata={
                    "source": source,
                    "file_size": extracted_content.file_size if extracted_content else len(file_bytes),
                    "page_count": extracted_content.page_count if extracted_content else 0,
                    "extracted_at": extracted_content.extracted_at.isoformat() if extracted_content else datetime.utcnow().isoformat(),
                    "chunking_stats": stats,
                    **(extracted_content.metadata if extracted_content else {}),
                    **(document_metadata or {}),
                },
            )
            logger.info(f"✓ Created document record: {document_id}")
            
            chunk_ids = self.repository.create_chunks(
                document_id=document_id,
                chunks=chunks,
            )
            logger.info(f"✓ Created {len(chunk_ids)} chunk records")
            
            # Step 4.5: Process and store tables (only if table processing is enabled)
            table_chunk_ids = []
            processed_tables = []
            if extracted_tables and self.enable_tables:
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
            
            # Step 5: Generate embeddings for text chunks (only if text processing is enabled)
            text_embeddings = []
            if self.enable_text and chunks:
                logger.info(f"Step 5: Generating embeddings for {len(chunks)} text chunks")
                
                # Extract chunk texts for embedding
                chunk_texts = [chunk.text for chunk in chunks]
                
                # Generate embeddings in batch with metrics
                embedding_start = time.time()
                text_embeddings = self.embedder.embed_batch(
                    texts=chunk_texts,
                    show_progress=True,
                )
                embedding_duration = time.time() - embedding_start
                
                # Record text embedding metrics
                text_embedding_duration_seconds.observe(embedding_duration)
                text_embeddings_generated_total.inc(len(text_embeddings))
                text_embedding_batch_size.observe(len(chunk_texts))
                
                logger.info(
                    f"✓ Generated {len(text_embeddings)} text embeddings "
                    f"(dimension: {self.embedder.embedding_dim})"
                )
            else:
                logger.info(f"Step 5: Skipping text embeddings (text processing disabled or no chunks)")
            
            # Step 5.5: Generate embeddings for table chunks
            table_embeddings = []
            if processed_tables:
                logger.info(f"Step 5.5: Generating embeddings for {len(processed_tables)} table chunk(s)")
                
                # Extract flattened table texts for embedding
                table_texts = [processed_table.table_text for processed_table in processed_tables]
                
                # Generate embeddings in batch (same model as text) with metrics
                embedding_start = time.time()
                table_embeddings = self.embedder.embed_batch(
                    texts=table_texts,
                    show_progress=True,
                )
                embedding_duration = time.time() - embedding_start
                
                # Record text embedding metrics (tables use same text embedding model)
                text_embedding_duration_seconds.observe(embedding_duration)
                text_embeddings_generated_total.inc(len(table_embeddings))
                text_embedding_batch_size.observe(len(table_texts))
                
                logger.info(
                    f"✓ Generated {len(table_embeddings)} table embeddings "
                    f"(dimension: {self.embedder.embedding_dim})"
                )
            
            # Step 6: Store text embeddings in Qdrant (only if text processing is enabled)
            if self.enable_text and chunks and text_embeddings:
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
            else:
                logger.info(f"Step 6: Skipping text embedding storage (text processing disabled or no embeddings)")
            
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
            
            # Step 6.6: Process and store images (only if image processing is enabled)
            image_ids = []
            image_chunk_ids = []
            image_storage_paths = []  # Track storage paths for each image
            processed_extracted_images = []  # Track which images were successfully processed
            image_embeddings = []  # Initialize to empty list in case no images are extracted
            if extracted_images and self.enable_images:
                logger.info(f"Step 6.6: Processing {len(extracted_images)} image(s)")
                
                # Upload images to Supabase storage and store in database
                for extracted_image in extracted_images:
                    try:
                        # Generate caption if vision processor is available (regardless of vision mode)
                        caption = None
                        if self.vision_processor:
                            try:
                                logger.debug(f"Generating caption for image {extracted_image.image_index}")
                                vision_result = self.vision_processor.process_image(
                                    image_bytes=extracted_image.image_bytes
                                )
                                caption = vision_result.description
                                # Combine OCR text (if any) with caption
                                if extracted_image.extracted_text:
                                    extracted_image.extracted_text = f"{extracted_image.extracted_text}\n\nCaption: {caption}"
                                else:
                                    extracted_image.extracted_text = caption
                                logger.debug(f"✓ Generated caption for image {extracted_image.image_index}: {caption[:50]}...")
                            except Exception as e:
                                logger.warning(f"Failed to generate caption for image {extracted_image.image_index}: {e}")
                                # Continue without caption (use OCR text if available)
                        
                        # Upload image to Supabase storage
                        try:
                            storage_path = self.image_storage.upload_image(
                                image_bytes=extracted_image.image_bytes,
                                document_id=document_id,
                                image_index=extracted_image.image_index,
                                image_ext=extracted_image.image_ext,
                                add_timestamp=True,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to upload image {extracted_image.image_index} to Supabase: {e}")
                            continue
                        
                        # Store image record in database
                        image_id = self.repository.create_image(
                            document_id=document_id,
                            image_path=storage_path,
                            image_type=extracted_image.image_type,
                            extracted_text=extracted_image.extracted_text,
                            caption=caption,  # Store caption separately
                            metadata={
                                "width": extracted_image.width,
                                "height": extracted_image.height,
                                "format": extracted_image.image_ext,
                                "page": extracted_image.page,
                                "position": extracted_image.position,
                                **extracted_image.metadata,
                            },
                        )
                        image_ids.append(image_id)
                        image_storage_paths.append(storage_path)
                        processed_extracted_images.append(extracted_image)
                        
                        # Create a chunk for the image (for consistency with text/table chunks)
                        # The chunk will reference the image
                        from uuid import uuid4
                        image_chunk_id = uuid4()
                        image_chunk_ids.append(image_chunk_id)
                        
                        # Generate descriptive text for the image chunk
                        # This helps the LLM understand what the image contains
                        image_chunk_text = self._generate_image_chunk_text(
                            extracted_image=extracted_image,
                            extracted_content=extracted_content,
                        )
                        
                        # Store image chunk in database
                        self.repository.create_chunk(
                            document_id=document_id,
                            chunk_text=image_chunk_text,
                            chunk_index=len(chunks) + len(processed_tables) + len(image_ids) - 1,
                            chunk_type="image",
                            image_path=storage_path,
                            image_caption=extracted_image.extracted_text,
                            embedding_type="image",
                            metadata={
                                **chunk_metadata,
                                "image_id": str(image_id),
                                "image_type": extracted_image.image_type,
                                "width": extracted_image.width,
                                "height": extracted_image.height,
                                "page": extracted_image.page,
                            },
                        )
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image {extracted_image.image_index}: {e}")
                        continue
                
                logger.info(f"✓ Processed {len(image_ids)} image(s)")
                
                # Step 6.7: Generate image embeddings
                if image_ids:
                    logger.info(f"Step 6.7: Generating embeddings for {len(image_ids)} image(s)")
                    
                    # Extract image bytes for embedding (only successfully processed ones)
                    image_bytes_list = [img.image_bytes for img in processed_extracted_images]
                    
                    # Generate embeddings in batch with metrics
                    embedding_start = time.time()
                    image_embeddings = self.image_embedder.embed_batch(
                        image_bytes_list=image_bytes_list,
                        show_progress=True,
                    )
                    embedding_duration = time.time() - embedding_start
                    
                    # Record image embedding metrics
                    image_embedding_duration_seconds.observe(embedding_duration)
                    image_embeddings_generated_total.inc(len(image_embeddings))
                    image_embedding_batch_size.observe(len(image_bytes_list))
                    
                    logger.info(
                        f"✓ Generated {len(image_embeddings)} image embeddings "
                        f"(dimension: {self.image_embedder.embedding_dim})"
                    )
                    
                    # Step 6.8: Store image embeddings in Qdrant (image_chunks collection)
                    logger.info(f"Step 6.8: Storing image embeddings in Qdrant (image_chunks collection)")
                    
                    # Prepare payloads for image chunks
                    image_payloads = []
                    for i, (extracted_image, image_id, image_chunk_id, storage_path) in enumerate(
                        zip(processed_extracted_images, image_ids, image_chunk_ids, image_storage_paths)
                    ):
                        payload = {
                            "chunk_id": str(image_chunk_id),
                            "document_id": str(document_id),
                            "image_id": str(image_id),
                            "image_path": storage_path,  # Storage path in Supabase
                            "image_type": extracted_image.image_type,
                            "caption": extracted_image.extracted_text,
                            "chunk_index": len(chunks) + len(processed_tables) + i,
                            "chunk_type": "image",
                            "embedding_type": "image",
                            "filename": filename,
                            "document_type": file_type,
                            "source": source,
                            "source_path": object_key,
                            "metadata": {
                                "width": extracted_image.width,
                                "height": extracted_image.height,
                                "format": extracted_image.image_ext,
                                "page": extracted_image.page,
                            },
                        }
                        image_payloads.append(payload)
                    
                    # Store image vectors in Qdrant image_chunks collection
                    self.image_vector_repo.store_vectors(
                        chunk_ids=image_chunk_ids,
                        embeddings=image_embeddings,
                        payloads=image_payloads,
                    )
                    logger.info(f"✓ Stored {len(image_embeddings)} image embeddings in Qdrant (image_chunks)")
            
            # Step 7: Index text chunks in Elasticsearch (BM25) (only if text processing is enabled)
            if self.enable_text and chunks:
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
            else:
                logger.info(f"Step 7: Skipping text indexing (text processing disabled or no chunks)")
            
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
            
            # Step 8: Build knowledge graph (if Neo4j is enabled)
            if self.graph_repo and settings.neo4j_enabled:
                try:
                    logger.info(f"Step 8: Building knowledge graph for document {document_id}")
                    self._build_document_graph(
                        document_id=str(document_id),
                        title=filename,
                        source=object_key,
                        document_type=file_type,
                        chunks=chunks,
                        chunk_ids=chunk_ids,
                        table_chunks=processed_tables,
                        table_chunk_ids=table_chunk_ids,
                        image_chunk_ids=image_chunk_ids,
                        metadata=document_metadata,
                    )
                    logger.info("✓ Knowledge graph built successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to build knowledge graph for document {document_id}: {str(e)}. "
                        f"Chunks: {len(chunks)} text, {len(processed_tables)} tables, {len(image_chunk_ids)} images.",
                        exc_info=True,
                    )
                    logger.warning("Continuing without graph. Document will be available for vector/sparse search but not graph queries.")
            else:
                logger.info("Step 8: Skipping knowledge graph (Neo4j disabled or not available)")
            
            total_chunks = len(chunks) + len(processed_tables) + len(image_ids)
            logger.info(
                f"Pipeline complete: Document {document_id} ingested with "
                f"{len(chunks)} text chunks, {len(processed_tables)} table chunks, "
                f"{len(image_ids)} image chunks, "
                f"embeddings stored in Qdrant, and BM25 index in Elasticsearch"
            )
            
            return {
                "document_id": str(document_id),
                "object_key": object_key,
                "chunks_count": len(chunks),
                "table_chunks_count": len(processed_tables),
                "image_chunks_count": len(image_ids),
                "total_chunks_count": total_chunks + len(image_ids),
                "chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
                "table_chunk_ids": [str(chunk_id) for chunk_id in table_chunk_ids],
                "image_chunk_ids": [str(chunk_id) for chunk_id in image_chunk_ids],
                "image_ids": [str(img_id) for img_id in image_ids],
                "text_embeddings_count": len(text_embeddings),
                "table_embeddings_count": len(table_embeddings),
                "image_embeddings_count": len(image_embeddings),
                "extracted_content": extracted_content,
                "extracted_tables": extracted_tables,
                "extracted_images": extracted_images,
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
    
    def _get_spacy_nlp(self):
        """
        Lazily load spaCy model to avoid startup delays.
        
        Returns:
            spaCy Language model or None if unavailable
        """
        if not hasattr(self, '_spacy_nlp'):
            self._spacy_nlp = None
            try:
                import spacy
                try:
                    self._spacy_nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model: en_core_web_sm")
                except OSError:
                    logger.warning(
                        "spaCy model 'en_core_web_sm' not found. "
                        "Run: python -m spacy download en_core_web_sm"
                    )
            except ImportError:
                logger.warning("spaCy not installed. Using regex fallback for entity extraction.")
        return self._spacy_nlp
    
    def _extract_entities_spacy(
        self,
        chunks: List[Chunk],
        table_chunks: List[ProcessedTable]
    ) -> List[Dict[str, Any]]:
        """
        Extract named entities using spaCy NER.
        
        Extracts universal entity types that work across domains:
        - PERSON: People, including fictional
        - ORG: Organizations, companies, agencies
        - GPE: Geopolitical entities (countries, cities, states)
        - DATE: Absolute or relative dates
        - MONEY: Monetary values
        - PERCENT: Percentages
        - PRODUCT: Products, objects
        - EVENT: Named events
        - LAW: Named documents made into laws
        
        Uses normalized_name (lowercase, trimmed) for cross-document entity resolution.
        
        Args:
            chunks: List of text chunks
            table_chunks: List of processed table chunks
            
        Returns:
            List of entity dictionaries with normalized_name for cross-doc matching
        """
        import uuid as uuid_module
        
        nlp = self._get_spacy_nlp()
        if nlp is None:
            # Fallback to regex if spaCy unavailable
            return self._extract_entities_regex_fallback(chunks, table_chunks)
        
        entities = []
        entity_map = {}  # Deduplicate by normalized_name
        
        # Combine text from chunks (with size limit)
        MAX_TEXT_FOR_NER = 100000  # spaCy works better with smaller chunks
        
        all_chunks_text = [chunk.text for chunk in chunks if chunk.text]
        all_text = " ".join(all_chunks_text)
        
        # Sample if too large
        if len(all_text) > MAX_TEXT_FOR_NER:
            logger.info(f"Large document ({len(all_text)} chars), sampling for NER")
            sample_size = max(50, len(chunks) // 10)
            step = max(1, len(chunks) // sample_size)
            sampled_chunks = chunks[::step][:sample_size]
            all_text = " ".join([c.text for c in sampled_chunks if c.text])
            logger.info(f"Sampled {len(sampled_chunks)} chunks for NER")
        
        # Add table text
        for table in table_chunks[:10]:  # Limit tables
            if hasattr(table, 'table_markdown') and table.table_markdown:
                all_text += " " + table.table_markdown[:5000]
        
        # Entity types we want to extract (universal, domain-agnostic)
        WANTED_ENTITY_TYPES = {
            "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
            "MONEY", "PERCENT", "PRODUCT", "EVENT", "LAW", "WORK_OF_ART"
        }
        
        try:
            # Process with spaCy (use nlp.pipe for efficiency if needed)
            doc = nlp(all_text[:MAX_TEXT_FOR_NER])
            
            for ent in doc.ents:
                if ent.label_ not in WANTED_ENTITY_TYPES:
                    continue
                
                entity_name = ent.text.strip()
                if len(entity_name) < 2 or len(entity_name) > 100:
                    continue
                
                # Normalize for cross-document matching
                normalized_name = entity_name.lower().strip()
                
                # Deduplicate by normalized name
                if normalized_name in entity_map:
                    continue
                
                entity_id = f"ent_{uuid_module.uuid4().hex[:8]}"
                entity = {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "normalized_name": normalized_name,
                    "entity_type": ent.label_,
                    "confidence": 0.85,  # spaCy confidence
                }
                
                # Add entity_value for certain types
                if ent.label_ in ("MONEY", "PERCENT", "DATE", "TIME"):
                    entity["entity_value"] = entity_name
                
                entities.append(entity)
                entity_map[normalized_name] = entity
            
            logger.info(f"Extracted {len(entities)} entities using spaCy NER")
            
            # Log entity type distribution
            type_counts = {}
            for e in entities:
                t = e.get("entity_type", "UNKNOWN")
                type_counts[t] = type_counts.get(t, 0) + 1
            logger.debug(f"Entity types: {type_counts}")
            
            return entities
            
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}. Falling back to regex.")
            return self._extract_entities_regex_fallback(chunks, table_chunks)
    
    def _extract_entities_regex_fallback(
        self,
        chunks: List[Chunk],
        table_chunks: List[ProcessedTable]
    ) -> List[Dict[str, Any]]:
        """
        Fallback entity extraction using regex patterns.
        
        Used when spaCy is unavailable or fails.
        
        Args:
            chunks: List of text chunks
            table_chunks: List of processed table chunks
            
        Returns:
            List of entity dictionaries
        """
        import re
        import uuid as uuid_module
        
        entities = []
        entity_map = {}
        
        MAX_TEXT = 500000
        all_text = " ".join([c.text for c in chunks if c.text])
        
        if len(all_text) > MAX_TEXT:
            sample_size = max(100, len(chunks) // 10)
            step = max(1, len(chunks) // sample_size)
            all_text = " ".join([c.text for c in chunks[::step][:sample_size] if c.text])
        
        for table in table_chunks[:10]:
            if hasattr(table, 'table_markdown') and table.table_markdown:
                all_text += " " + table.table_markdown[:5000]
        
        # Pattern 1: Capitalized phrases (organizations, names)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(org_pattern, all_text):
            name = match.group(1)
            normalized = name.lower().strip()
            if len(name) > 3 and normalized not in entity_map and len(name.split()) <= 4:
                entity_id = f"ent_{uuid_module.uuid4().hex[:8]}"
                entities.append({
                    "entity_id": entity_id,
                    "entity_name": name,
                    "normalized_name": normalized,
                    "entity_type": "CONCEPT",
                    "confidence": 0.6
                })
                entity_map[normalized] = True
        
        # Pattern 2: Numbers with units (metrics)
        metric_pattern = r'\b(\d+\.?\d*)\s*([%$€£¥]|million|billion|thousand|M|B|K)\b'
        for match in re.finditer(metric_pattern, all_text, re.IGNORECASE):
            value, unit = match.group(1), match.group(2)
            name = f"{value} {unit}"
            normalized = name.lower().strip()
            if normalized not in entity_map:
                entity_id = f"ent_{uuid_module.uuid4().hex[:8]}"
                entities.append({
                    "entity_id": entity_id,
                    "entity_name": name,
                    "normalized_name": normalized,
                    "entity_type": "MONEY" if unit in "$€£¥" else "PERCENT",
                    "confidence": 0.7,
                    "entity_value": name
                })
                entity_map[normalized] = True
        
        logger.info(f"Extracted {len(entities)} entities using regex fallback")
        return entities
    
    def _extract_entities_simple(
        self,
        chunks: List[Chunk],
        table_chunks: List[ProcessedTable]
    ) -> List[Dict[str, Any]]:
        """
        Extract entities using spaCy NER with regex fallback.
        
        This is the main entry point for entity extraction.
        Uses spaCy for high-quality NER, falls back to regex if unavailable.
        
        Args:
            chunks: List of text chunks
            table_chunks: List of processed table chunks
            
        Returns:
            List of entity dictionaries with normalized_name for cross-doc matching
        """
        return self._extract_entities_spacy(chunks, table_chunks)
    
    def _extract_topics_from_chunks(
        self,
        chunks: List[Chunk],
        sections: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract topics from chunks and sections to create a global ontology layer.
        
        This creates thematic bridges across documents for cross-document retrieval.
        Uses:
        1. Section titles as primary topics
        2. Common entity types as topic indicators
        3. Keyword extraction from chunk content
        
        Args:
            chunks: List of text chunks
            sections: List of section dictionaries
            entities: List of extracted entities
            
        Returns:
            List of topic dictionaries with normalized_name for cross-doc matching
        """
        import uuid as uuid_module
        import re
        from collections import Counter
        
        topics = []
        topic_map = {}  # normalized_name -> topic dict
        
        # 1. Extract topics from section titles
        for section in sections:
            title = section.get("title", "").strip()
            if not title or title == "Document Content" or title == "Tables":
                continue
            
            # Clean the title - remove numbering like "1.2 " or "Chapter 3: "
            cleaned_title = re.sub(r'^\d+(\.\d+)*\s+', '', title)  # Remove "1.2 "
            cleaned_title = re.sub(r'^(Chapter|Section|Part)\s+[IVX\d]+:?\s*', '', cleaned_title, flags=re.IGNORECASE)
            
            if len(cleaned_title) > 3:  # Minimum length
                normalized = cleaned_title.lower().strip()
                
                if normalized not in topic_map:
                    topic_map[normalized] = {
                        "topic_id": str(uuid_module.uuid4()),
                        "topic_name": cleaned_title,
                        "normalized_name": normalized,
                        "keywords": [w.lower() for w in cleaned_title.split() if len(w) > 3],
                        "description": f"Topic from section: {cleaned_title}"
                    }
        
        # 2. Extract topics from entity types (group similar entities as topics)
        # E.g., if there are many ORG entities, create "Organizations" topic
        entity_type_counts = Counter([e.get("entity_type") for e in entities if e.get("entity_type")])
        
        topic_from_entities = {
            "ORG": "Organizations and Companies",
            "PERSON": "People and Individuals",
            "GPE": "Geographic Locations",
            "DATE": "Timeline and Events",
            "MONEY": "Financial Information",
            "PERCENT": "Statistics and Metrics",
            "PRODUCT": "Products and Services",
            "EVENT": "Events and Milestones",
            "LAW": "Legal and Regulatory",
        }
        
        for entity_type, count in entity_type_counts.items():
            if count >= 3 and entity_type in topic_from_entities:  # At least 3 entities of this type
                topic_name = topic_from_entities[entity_type]
                normalized = topic_name.lower()
                
                if normalized not in topic_map:
                    topic_map[normalized] = {
                        "topic_id": str(uuid_module.uuid4()),
                        "topic_name": topic_name,
                        "normalized_name": normalized,
                        "keywords": [w.lower() for w in topic_name.split()],
                        "description": f"Thematic category for {entity_type} entities"
                    }
        
        # 3. Extract common keywords from chunks as topics (simplified)
        # Only extract if we have few section-based topics
        if len(topic_map) < 3:
            MAX_TEXT = 100000
            all_text = " ".join([c.text[:500] for c in chunks[:50] if c.text])[:MAX_TEXT]
            
            # Extract capitalized phrases (potential topics)
            topic_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b'
            potential_topics = re.findall(topic_pattern, all_text)
            topic_freq = Counter(potential_topics)
            
            for topic_phrase, freq in topic_freq.most_common(10):
                if freq >= 3 and len(topic_phrase) > 8:  # Appears at least 3 times
                    normalized = topic_phrase.lower()
                    
                    if normalized not in topic_map:
                        topic_map[normalized] = {
                            "topic_id": str(uuid_module.uuid4()),
                            "topic_name": topic_phrase,
                            "normalized_name": normalized,
                            "keywords": [w.lower() for w in topic_phrase.split() if len(w) > 3],
                            "description": f"Recurring topic from content"
                        }
        
        topics = list(topic_map.values())
        logger.info(f"Extracted {len(topics)} topics from {len(sections)} sections and {len(entities)} entities")
        
        return topics
    
    def _extract_entity_relationships(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Chunk],
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities based on co-occurrence in chunks.
        
        Creates RELATED_TO edges between entities that appear together,
        enabling graph traversal across entity connections.
        
        Args:
            entities: List of entity dictionaries
            chunks: List of text chunks
            
        Returns:
            List of entity relationship dictionaries
        """
        from collections import defaultdict
        
        # Build entity -> chunks mapping
        entity_chunks = defaultdict(set)
        
        for entity in entities:
            entity_name = entity.get("entity_name", "")
            entity_normalized = entity.get("normalized_name", entity_name.lower())
            
            # Find chunks containing this entity
            for idx, chunk in enumerate(chunks):
                if not chunk.text:
                    continue
                
                # Case-insensitive search
                if entity_name.lower() in chunk.text.lower():
                    entity_chunks[entity_normalized].add(idx)
        
        # Find entity co-occurrences
        relationships = []
        entity_list = list(entity_chunks.keys())
        
        for i, entity1 in enumerate(entity_list):
            chunks1 = entity_chunks[entity1]
            
            for entity2 in entity_list[i + 1:]:  # Avoid duplicates and self-relations
                chunks2 = entity_chunks[entity2]
                
                # Check if entities co-occur in same chunks
                common_chunks = chunks1 & chunks2
                
                if common_chunks:
                    # Create bidirectional relationships
                    confidence = min(1.0, len(common_chunks) / 3.0)  # More co-occurrences = higher confidence
                    
                    relationships.append({
                        "from_entity_normalized_name": entity1,
                        "to_entity_normalized_name": entity2,
                        "context": f"Co-occur in {len(common_chunks)} chunks",
                        "confidence": confidence
                    })
                    
                    # Add reverse relationship
                    relationships.append({
                        "from_entity_normalized_name": entity2,
                        "to_entity_normalized_name": entity1,
                        "context": f"Co-occur in {len(common_chunks)} chunks",
                        "confidence": confidence
                    })
        
        logger.info(f"Extracted {len(relationships)} entity relationships from co-occurrences")
        return relationships
    
    def _link_entities_to_chunks(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Chunk],
        chunk_ids: List[UUID],
        table_chunks: Optional[List[ProcessedTable]] = None,
        table_chunk_ids: Optional[List[UUID]] = None,
        section_id: str = ""  # Unused but kept for API compatibility
    ) -> List[Dict[str, Any]]:
        """
        Link entities to chunks (text and tables) where they appear.
        
        Args:
            entities: List of entity dictionaries
            chunks: List of text chunks
            chunk_ids: List of text chunk UUIDs
            table_chunks: List of table chunks (optional)
            table_chunk_ids: List of table chunk UUIDs (optional)
            section_id: Section identifier
            
        Returns:
            List of relationship dictionaries
        """
        from app.repositories.graph_schema import NODE_LABELS, PROPERTY_KEYS, RELATIONSHIP_TYPES, RELATIONSHIP_PROPERTIES
        
        relationships = []
        
        # Create a mapping of entity names for quick lookup
        entity_map = {e["entity_name"].lower(): e for e in entities}
        
        # Link entities to TEXT chunks
        for chunk, chunk_id in zip(chunks, chunk_ids):
            chunk_text_lower = chunk.text.lower()
            
            for entity_name, entity in entity_map.items():
                # Count frequency in chunk
                frequency = chunk_text_lower.count(entity_name)
                
                if frequency > 0:
                    relationships.append({
                        "from_label": NODE_LABELS['CHUNK'],
                        "from_id_key": PROPERTY_KEYS['CHUNK_ID'],
                        "from_id_value": str(chunk_id),
                        "to_label": NODE_LABELS['ENTITY'],
                        "to_id_key": PROPERTY_KEYS['ENTITY_ID'],
                        "to_id_value": entity["entity_id"],
                        "relationship_type": RELATIONSHIP_TYPES['MENTIONS'],
                        "properties": {
                            RELATIONSHIP_PROPERTIES['FREQUENCY']: frequency,
                            RELATIONSHIP_PROPERTIES['IMPORTANCE']: min(0.9, 0.5 + (frequency * 0.1)),
                            RELATIONSHIP_PROPERTIES['CONTEXT']: 'text',
                        }
                    })
        
        # Link entities to TABLE chunks
        if table_chunks and table_chunk_ids:
            logger.debug(f"Linking entities to {len(table_chunks)} table chunks...")
            table_relationships_count = 0
            
            for table_chunk, table_chunk_id in zip(table_chunks, table_chunk_ids):
                # Get table text (markdown or text format)
                table_text = ""
                if hasattr(table_chunk, 'table_markdown') and table_chunk.table_markdown:
                    table_text = table_chunk.table_markdown.lower()
                elif hasattr(table_chunk, 'table_text') and table_chunk.table_text:
                    table_text = table_chunk.table_text.lower()
                
                if not table_text:
                    continue
                
                for entity_name, entity in entity_map.items():
                    # Count frequency in table
                    frequency = table_text.count(entity_name)
                    
                    if frequency > 0:
                        relationships.append({
                            "from_label": NODE_LABELS['CHUNK'],
                            "from_id_key": PROPERTY_KEYS['CHUNK_ID'],
                            "from_id_value": str(table_chunk_id),
                            "to_label": NODE_LABELS['ENTITY'],
                            "to_id_key": PROPERTY_KEYS['ENTITY_ID'],
                            "to_id_value": entity["entity_id"],
                            "relationship_type": RELATIONSHIP_TYPES['MENTIONS'],
                            "properties": {
                                RELATIONSHIP_PROPERTIES['FREQUENCY']: frequency,
                                RELATIONSHIP_PROPERTIES['IMPORTANCE']: min(0.9, 0.5 + (frequency * 0.1)),
                                RELATIONSHIP_PROPERTIES['CONTEXT']: 'table',
                            }
                        })
                        table_relationships_count += 1
            
            if table_relationships_count > 0:
                logger.info(f"Created {table_relationships_count} entity-table relationships")
        
        return relationships
    
    def _detect_section_headings(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Detect section headings from chunks based on text patterns.
        
        This groups chunks under their appropriate sections for better
        hierarchical retrieval in the knowledge graph.
        
        Args:
            chunks: List of text chunks
        
        Returns:
            List of section dictionaries with their chunks
        """
        import re
        import uuid as uuid_module
        
        sections = []
        current_section = None
        section_index = 0
        
        # Patterns that indicate a heading (numbered sections, all caps, short lines)
        heading_patterns = [
            r'^\d+(\.\d+)*\s+[A-Z]',  # "1.2 Title" or "2.4.1 Subsection"
            r'^Chapter\s+\d+',         # "Chapter 1"
            r'^Section\s+\d+',         # "Section 1"
            r'^Part\s+[IVX\d]+',       # "Part I" or "Part 1"
            r'^[A-Z][A-Z\s]{5,50}$',   # ALL CAPS (5-50 chars)
        ]
        
        for chunk, chunk_id in zip(chunks, range(len(chunks))):
            # Check if this chunk starts with a heading
            first_line = chunk.text.strip().split('\n')[0].strip() if chunk.text else ""
            is_heading = False
            detected_title = None
            
            for pattern in heading_patterns:
                if re.match(pattern, first_line):
                    is_heading = True
                    # Use first line as section title (truncate if too long)
                    detected_title = first_line[:100] if len(first_line) > 100 else first_line
                    break
            
            # Also check metadata for section info
            if not is_heading and chunk.metadata:
                meta_heading = chunk.metadata.get("heading") or chunk.metadata.get("section_title")
                if meta_heading:
                    is_heading = True
                    detected_title = meta_heading[:100] if len(meta_heading) > 100 else meta_heading
            
            if is_heading and detected_title:
                # Save current section if exists
                if current_section and current_section["chunks"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "section_id": f"sec_{uuid_module.uuid4().hex[:8]}",
                    "title": detected_title,
                    "index": section_index,
                    "chunks": [],
                }
                section_index += 1
            
            # Create a default section if none exists
            if current_section is None:
                current_section = {
                    "section_id": f"sec_{uuid_module.uuid4().hex[:8]}",
                    "title": "Document Content",
                    "index": 0,
                    "chunks": [],
                }
                section_index = 1
            
            # Note: We don't add chunk data here, just track sections
            # The actual chunk data will be added in _build_document_graph
        
        # Don't forget the last section
        if current_section:
            sections.append(current_section)
        
        # If no sections detected, create a default one
        if not sections:
            sections = [{
                "section_id": f"sec_{uuid_module.uuid4().hex[:8]}",
                "title": "Document Content",
                "index": 0,
                "chunks": [],
            }]
        
        return sections
    
    def _build_document_graph(
        self,
        document_id: str,
        title: str,
        source: str,
        document_type: str,
        chunks: List[Chunk],
        chunk_ids: List[UUID],
        table_chunks: List[ProcessedTable],
        table_chunk_ids: List[UUID],
        image_chunk_ids: List[UUID],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Build knowledge graph structure for a document.
        
        Creates Document -> Section -> Chunk hierarchy and links entities.
        Detects sections from headings in the text for better hierarchical retrieval.
        
        Args:
            document_id: Document UUID as string
            title: Document title
            source: Document source path
            document_type: Document type
            chunks: List of text chunks
            chunk_ids: List of chunk UUIDs
            table_chunks: List of processed table chunks
            table_chunk_ids: List of table chunk UUIDs
            image_chunk_ids: List of image chunk UUIDs
            metadata: Optional document metadata
        """
        import uuid
        import re
        from app.repositories.graph_schema import NODE_LABELS, PROPERTY_KEYS, RELATIONSHIP_TYPES
        
        # Validate inputs
        total_expected_chunks = len(chunks) + len(table_chunks) + len(image_chunk_ids)
        logger.info(
            f"Building graph for document {document_id}: "
            f"{len(chunks)} text chunks, {len(table_chunks)} table chunks, {len(image_chunk_ids)} image chunks "
            f"(total: {total_expected_chunks} chunks)"
        )
        
        if total_expected_chunks == 0:
            logger.warning(f"No chunks to add to graph for document {document_id}. Creating document node only.")
            self.graph_repo.create_document_node(
                document_id=document_id,
                title=title,
                source=source,
                document_type=document_type,
                metadata=metadata,
            )
            return
        
        # Detect sections from chunks based on headings
        # This groups chunks under their appropriate section headings
        sections = []
        current_section = None
        section_index = 0
        
        # Enhanced patterns for heading detection
        heading_patterns = [
            (r'^\d+(\.\d+)*\s+[A-Z]', True),  # "1.2 Title" or "2.4.1 Subsection" (high confidence)
            (r'^Chapter\s+\d+', True),         # "Chapter 1" (high confidence)
            (r'^Section\s+\d+', True),         # "Section 1" (high confidence)
            (r'^Part\s+[IVX\d]+', True),      # "Part I" or "Part 1" (high confidence)
            (r'^[A-Z][A-Z\s]{8,}$', False),   # All caps, 8+ chars (medium confidence, check context)
            (r'^\d+\.\s+[A-Z][a-z]+', True),  # "1. Introduction" (high confidence)
            (r'^[A-Z][a-z]+\s+\d+:', True),   # "Chapter 1:" or "Section 2:" (high confidence)
        ]
        
        def normalize_spaced_text(text: str) -> str:
            """Fix spaced text like 'L I N K' -> 'LINK' from PDF extraction."""
            # Check if text has pattern: single char, space, single char...
            if re.match(r'^([A-Z]\s){2,}[A-Z]$', text.strip()):
                return text.replace(' ', '')
            return text
        
        def is_likely_heading(text: str, chunk_text: str) -> tuple:
            """Check if text is likely a heading. Returns (is_heading, confidence, normalized_text)."""
            normalized = normalize_spaced_text(text)
            
            # Check patterns
            for pattern, high_conf in heading_patterns:
                if re.match(pattern, normalized):
                    return (True, "high" if high_conf else "medium", normalized)
                if re.match(pattern, text):
                    return (True, "high" if high_conf else "medium", text)
            
            # Additional heuristics for medium confidence:
            # Short line (<80 chars), starts with capital, no period at end
            if len(text) < 80 and text and text[0].isupper() and not text.endswith('.'):
                # Check if next line exists and looks like body text
                lines = chunk_text.strip().split('\n')
                if len(lines) >= 2:
                    second_line = lines[1].strip()
                    # If second line is longer and looks like paragraph, first line might be heading
                    if len(second_line) > 80 and second_line[0].islower():
                        return (True, "low", text)
            
            return (False, "none", text)
        
        for chunk, chunk_id in zip(chunks, chunk_ids):
            # Check if this chunk starts with a heading
            if not chunk.text:
                continue
                
            # Check first 3 lines for potential headings
            lines = chunk.text.strip().split('\n')[:3]
            is_heading = False
            detected_title = None
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                heading, confidence, normalized = is_likely_heading(line, chunk.text)
                
                # Accept high and medium confidence headings
                if heading and confidence in ["high", "medium"]:
                    is_heading = True
                    detected_title = normalized[:100] if len(normalized) > 100 else normalized
                    logger.debug(f"Detected heading (confidence: {confidence}): '{detected_title}'")
                    break
            
            # Also check metadata
            if not is_heading and chunk.metadata:
                meta_heading = chunk.metadata.get("heading") or chunk.metadata.get("section_title")
                if meta_heading:
                    is_heading = True
                    detected_title = str(meta_heading)[:100]
            
            if is_heading and detected_title:
                # Save current section
                if current_section and current_section["chunks"]:
                    sections.append(current_section)
                
                # Start new section with detected title
                current_section = {
                    "section_id": f"sec_{uuid.uuid4().hex[:8]}",
                    "title": detected_title,
                    "index": section_index,
                    "chunks": [],
                }
                section_index += 1
                logger.debug(f"Detected section heading: '{detected_title}'")
            
            # Create default section if needed
            if current_section is None:
                current_section = {
                    "section_id": f"sec_{uuid.uuid4().hex[:8]}",
            "title": "Document Content",
            "index": 0,
            "chunks": [],
        }
                section_index = 1
        
            # Add chunk to current section
            try:
                chunk_data = {
                    "chunk_id": str(chunk_id),
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "content": chunk.text[:50000] if chunk.text and len(chunk.text) > 50000 else (chunk.text or ""),
                    "metadata": chunk.metadata if chunk.metadata else {},
                }
                current_section["chunks"].append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to add chunk {chunk_id} to section: {e}")
        
        # Add last section
        if current_section and current_section["chunks"]:
            sections.append(current_section)
        
        # Create default section if no sections were detected
        if not sections:
            section = {
                "section_id": f"sec_{uuid.uuid4().hex[:8]}",
                "title": "Document Content",
                "index": 0,
                "chunks": [],
            }
            for chunk, chunk_id in zip(chunks, chunk_ids):
                chunk_data = {
                    "chunk_id": str(chunk_id),
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "content": chunk.text[:50000] if chunk.text and len(chunk.text) > 50000 else (chunk.text or ""),
                    "metadata": chunk.metadata if chunk.metadata else {},
                }
                section["chunks"].append(chunk_data)
            sections.append(section)
        
        logger.info(f"Detected {len(sections)} sections for document {document_id}")
        for sec in sections[:5]:  # Log first 5 sections
            logger.debug(f"  Section: '{sec['title']}' with {len(sec['chunks'])} chunks")
        
        # Add table chunks to the last section (or create a Tables section)
        if table_chunks:
            # Create a section for tables if we have many, otherwise add to last section
            if len(table_chunks) > 3:
                table_section = {
                    "section_id": f"sec_{uuid.uuid4().hex[:8]}",
                    "title": "Tables",
                    "index": len(sections),
                    "chunks": [],
                }
                target_section = table_section
                sections.append(table_section)
            else:
                target_section = sections[-1] if sections else None
                if not target_section:
                    target_section = {
                        "section_id": f"sec_{uuid.uuid4().hex[:8]}",
                        "title": "Document Content",
                        "index": 0,
                        "chunks": [],
                    }
                    sections.append(target_section)
            
        table_chunks_added = 0
        for idx, (table_chunk, table_chunk_id) in enumerate(zip(table_chunks, table_chunk_ids)):
            try:
                table_content = ""
                if hasattr(table_chunk, 'table_markdown') and table_chunk.table_markdown:
                    table_content = table_chunk.table_markdown[:50000] if len(table_chunk.table_markdown) > 50000 else table_chunk.table_markdown
                elif hasattr(table_chunk, 'table_text') and table_chunk.table_text:
                    table_content = table_chunk.table_text[:50000] if len(table_chunk.table_text) > 50000 else table_chunk.table_text
                
                # Add table chunk to section (fixed indentation - was incorrectly inside elif block)
                target_section["chunks"].append({
                    "chunk_id": str(table_chunk_id),
                    "chunk_index": len(chunks) + idx,
                    "chunk_type": "table",
                    "content": table_content,
                    "metadata": table_chunk.metadata if hasattr(table_chunk, 'metadata') and table_chunk.metadata else {},
                })
                table_chunks_added += 1
            except Exception as e:
                logger.warning(f"Failed to add table chunk {table_chunk_id} to graph: {e}")
        
        logger.debug(f"Added {table_chunks_added} table chunks")
        
        # Add image chunks to the last section
        if image_chunk_ids:
            target_section = sections[-1] if sections else None
            if not target_section:
                target_section = {
                    "section_id": f"sec_{uuid.uuid4().hex[:8]}",
                    "title": "Document Content",
                    "index": 0,
                    "chunks": [],
                }
                sections.append(target_section)
            
        image_chunks_added = 0
        for i, image_chunk_id in enumerate(image_chunk_ids):
            try:
                target_section["chunks"].append({
                    "chunk_id": str(image_chunk_id),
                    "chunk_index": len(chunks) + len(table_chunks) + i,
                    "chunk_type": "image",
                    "content": "",  # Image content (caption) stored in Supabase
                    "metadata": {},
                })
                image_chunks_added += 1
            except Exception as e:
                logger.warning(f"Failed to add image chunk {image_chunk_id} to graph: {e}")
        
        logger.debug(f"Added {image_chunks_added} image chunks")
        
        # Validate we have chunks to add
        total_chunks_in_sections = sum(len(s["chunks"]) for s in sections)
        logger.info(f"Graph prepared: {len(sections)} sections with {total_chunks_in_sections} total chunks (expected: {total_expected_chunks})")
        
        if total_chunks_in_sections == 0:
            logger.warning(f"No chunks were successfully added to sections for document {document_id}")
            self.graph_repo.create_document_node(
                document_id=document_id,
                title=title,
                source=source,
                document_type=document_type,
                metadata=metadata,
            )
            return
        
        # Build graph structure using the detected sections
        try:
            # Log sample chunk for debugging
            if sections and sections[0]["chunks"]:
                sample_chunk = sections[0]["chunks"][0]
                logger.debug(
                    f"Sample chunk: id={sample_chunk.get('chunk_id')[:8]}..., "
                    f"content_length={len(sample_chunk.get('content', ''))}, "
                    f"type={sample_chunk.get('chunk_type')}"
                )
            
            # For very large documents (500+ chunks), process in batches
            BATCH_SIZE = 500
            
            if total_chunks_in_sections > BATCH_SIZE:
                logger.info(f"Large document ({total_chunks_in_sections} chunks), processing in batches")
                
                # Create document node first
                driver = self.graph_repo._get_driver()
                session = driver.session(database=self.graph_repo.database)
                try:
                    doc_query = f"""
                    MERGE (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
                    SET d.{PROPERTY_KEYS['TITLE']} = $title,
                        d.{PROPERTY_KEYS['SOURCE']} = $source
                    """
                    if document_type:
                        doc_query += f", d.{PROPERTY_KEYS['DOCUMENT_TYPE']} = $document_type"
                    if metadata:
                        doc_query += f", d.{PROPERTY_KEYS['METADATA']} = $metadata"
                    
                    params = {"document_id": document_id, "title": title, "source": source}
                    if document_type:
                        params["document_type"] = document_type
                    if metadata:
                        params["metadata"] = metadata
                    
                    session.run(doc_query, params)
                    logger.info(f"Created document node: {document_id}")
                finally:
                    session.close()
                
                # Process sections in batches
                for section in sections:
                    section_chunks = section["chunks"]
                    if len(section_chunks) > BATCH_SIZE:
                        # Split large sections into sub-batches
                        for batch_idx in range(0, len(section_chunks), BATCH_SIZE):
                            batch_chunks = section_chunks[batch_idx:batch_idx + BATCH_SIZE]
                            batch_section = {
                                "section_id": f"{section['section_id']}_{batch_idx // BATCH_SIZE}",
                                "title": section['title'] if batch_idx == 0 else f"{section['title']} (cont.)",
                                "index": section['index'],
                                "chunks": batch_chunks,
                            }
                            self.graph_repo.create_document_graph_batch(
                                document_id=document_id,
                                title=title,
                                source=source,
                                sections=[batch_section],
                                document_type=document_type,
                                metadata=None,
                                skip_document_node=True,
                            )
                    else:
                        # Section fits in one batch
                        self.graph_repo.create_document_graph_batch(
                            document_id=document_id,
                            title=title,
                            source=source,
                            sections=[section],
                            document_type=document_type,
                            metadata=None,
                            skip_document_node=True,
                        )
                
                logger.info(f"Created graph: {len(sections)} sections, {total_chunks_in_sections} chunks")
            else:
                # Small document - process all sections at once
                self.graph_repo.create_document_graph_batch(
                    document_id=document_id,
                    title=title,
                    source=source,
                    sections=sections,
                    document_type=document_type,
                    metadata=metadata,
                )
                logger.info(f"Created graph: {len(sections)} sections, {total_chunks_in_sections} chunks")
        except Exception as e:
            logger.error(f"Failed to create document graph structure: {str(e)}", exc_info=True)
            raise
        
        # Create NEXT_CHUNK relationships for sequential context expansion
        try:
            chunk_pairs = []
            for section in sections:
                section_chunks = section.get("chunks", [])
                for i in range(len(section_chunks) - 1):
                    current_chunk = section_chunks[i]
                    next_chunk = section_chunks[i + 1]
                    chunk_pairs.append({
                        "from_chunk_id": current_chunk["chunk_id"],
                        "to_chunk_id": next_chunk["chunk_id"],
                    })
            
            if chunk_pairs:
                self.graph_repo.create_next_chunk_relationships_batch(chunk_pairs)
                logger.info(f"Created {len(chunk_pairs)} NEXT_CHUNK relationships for context expansion")
        except Exception as e:
            logger.warning(f"Failed to create NEXT_CHUNK relationships: {e}. Continuing without sequential links.")
        
        # Extract entities and create entity nodes and relationships
        try:
            logger.info(f"Extracting entities from {len(chunks)} chunks and {len(table_chunks)} tables...")
            entities = self._extract_entities_simple(chunks, table_chunks)
            
            if entities:
                logger.info(f"Extracted {len(entities)} unique entities, creating entity nodes...")
                self.graph_repo.create_entity_nodes_batch(entities)
                
                # Link entities to chunks (text and tables)
                logger.info(f"Linking entities to {len(chunks)} text chunks and {len(table_chunks)} table chunks...")
                RELATIONSHIP_BATCH_SIZE = 1000
                
                # Use first section ID as default (entities are linked to chunks, not sections directly)
                default_section_id = sections[0]["section_id"] if sections else "default"
                relationships = self._link_entities_to_chunks(
                    entities, 
                    chunks, 
                    chunk_ids,
                    table_chunks,      # Include table chunks
                    table_chunk_ids,   # Include table chunk IDs
                    default_section_id
                )
                
                if relationships:
                    logger.info(f"Creating {len(relationships)} entity-chunk relationships")
                    if len(relationships) > RELATIONSHIP_BATCH_SIZE:
                        for i in range(0, len(relationships), RELATIONSHIP_BATCH_SIZE):
                            batch = relationships[i:i + RELATIONSHIP_BATCH_SIZE]
                            self.graph_repo.create_relationships_batch(batch)
                    else:
                        self.graph_repo.create_relationships_batch(relationships)
                    
                    logger.info(f"✓ Created {len(entities)} entities and {len(relationships)} relationships")
                    
                    # Extract and create entity-to-entity relationships
                    try:
                        logger.info("Extracting entity relationships from co-occurrences...")
                        entity_relationships = self._extract_entity_relationships(entities, chunks)
                        
                        if entity_relationships:
                            logger.info(f"Creating {len(entity_relationships)} entity-entity relationships")
                            # Batch entity relationships
                            ENTITY_REL_BATCH_SIZE = 500
                            if len(entity_relationships) > ENTITY_REL_BATCH_SIZE:
                                for i in range(0, len(entity_relationships), ENTITY_REL_BATCH_SIZE):
                                    batch = entity_relationships[i:i + ENTITY_REL_BATCH_SIZE]
                                    self.graph_repo.create_entity_relationships_batch(batch)
                            else:
                                self.graph_repo.create_entity_relationships_batch(entity_relationships)
                            logger.info(f"✓ Created {len(entity_relationships)} entity relationships")
                    except Exception as rel_error:
                        logger.warning(f"Failed to create entity relationships: {rel_error}. Continuing...")
                else:
                    logger.info(f"✓ Created {len(entities)} entities (no chunk relationships)")
            else:
                logger.info("No entities extracted from document")
        except Exception as e:
            logger.warning(f"Entity extraction/creation failed: {e}. Continuing without entities.", exc_info=True)
        
        # Extract topics and create topic nodes
        try:
            logger.info("Extracting topics for cross-document ontology...")
            # Ensure entities is defined (might be undefined if entity extraction failed)
            entities_list = entities if 'entities' in locals() and entities else []
            topics = self._extract_topics_from_chunks(chunks, sections, entities_list)
            
            if topics:
                logger.info(f"Creating {len(topics)} topic nodes...")
                self.graph_repo.create_topic_nodes_batch(topics)
                
                # Link chunks to topics based on keyword matching
                topic_relationships = []
                from app.repositories.graph_schema import NODE_LABELS, PROPERTY_KEYS, RELATIONSHIP_TYPES
                
                for topic in topics:
                    topic_keywords = set(topic.get("keywords", []))
                    if not topic_keywords:
                        continue
                    
                    # Link sections whose titles contain topic keywords
                    for section in sections:
                        section_title = section.get("title", "").lower()
                        if any(kw in section_title for kw in topic_keywords):
                            # Link chunks in this section to the topic
                            for chunk_data in section.get("chunks", []):
                                topic_relationships.append({
                                    "from_label": NODE_LABELS['CHUNK'],
                                    "from_id_key": PROPERTY_KEYS['CHUNK_ID'],
                                    "from_id_value": chunk_data["chunk_id"],
                                    "to_label": NODE_LABELS['TOPIC'],
                                    "to_id_key": PROPERTY_KEYS['TOPIC_NORMALIZED_NAME'],
                                    "to_id_value": topic["normalized_name"],
                                    "relationship_type": RELATIONSHIP_TYPES['HAS_TOPIC'],
                                    "properties": {"relevance": 0.8}
                                })
                
                # Link entities to topics (only if entities exist)
                if entities_list:
                    for entity in entities_list:
                        entity_type = entity.get("entity_type", "")
                        # Map entity types to topic names
                        entity_to_topic = {
                            "ORG": "organizations and companies",
                            "PERSON": "people and individuals",
                            "GPE": "geographic locations",
                            "DATE": "timeline and events",
                            "MONEY": "financial information",
                            "PERCENT": "statistics and metrics",
                            "PRODUCT": "products and services",
                            "EVENT": "events and milestones",
                            "LAW": "legal and regulatory",
                        }
                        
                        topic_normalized = entity_to_topic.get(entity_type)
                        if topic_normalized:
                            topic_relationships.append({
                                "from_label": NODE_LABELS['ENTITY'],
                                "from_id_key": PROPERTY_KEYS['ENTITY_NORMALIZED_NAME'],
                                "from_id_value": entity["normalized_name"],
                                "to_label": NODE_LABELS['TOPIC'],
                                "to_id_key": PROPERTY_KEYS['TOPIC_NORMALIZED_NAME'],
                                "to_id_value": topic_normalized,
                                "relationship_type": RELATIONSHIP_TYPES['ASSOCIATED_WITH'],
                                "properties": {"relevance": 0.7}
                            })
                
                if topic_relationships:
                    logger.info(f"Creating {len(topic_relationships)} topic relationships")
                    TOPIC_REL_BATCH_SIZE = 1000
                    if len(topic_relationships) > TOPIC_REL_BATCH_SIZE:
                        for i in range(0, len(topic_relationships), TOPIC_REL_BATCH_SIZE):
                            batch = topic_relationships[i:i + TOPIC_REL_BATCH_SIZE]
                            self.graph_repo.create_relationships_batch(batch)
                    else:
                        self.graph_repo.create_relationships_batch(topic_relationships)
                    logger.info(f"✓ Created {len(topics)} topics and {len(topic_relationships)} topic relationships")
                else:
                    logger.info(f"✓ Created {len(topics)} topics (no relationships)")
            else:
                logger.info("No topics extracted from document")
        except Exception as e:
            logger.warning(f"Topic extraction/creation failed: {e}. Continuing without topics.", exc_info=True)
    
    def _generate_image_chunk_text(
        self,
        extracted_image: ExtractedImage,
        extracted_content: ExtractedContent,
    ) -> str:
        """
        Generate descriptive text for an image chunk.
        
        This helps the LLM understand what the image contains, especially for charts/graphs.
        
        Args:
            extracted_image: The extracted image object
            extracted_content: The extracted text content (for finding surrounding context)
        
        Returns:
            Descriptive text for the image chunk
        """
        # Start with OCR text if available
        if extracted_image.extracted_text:
            base_text = extracted_image.extracted_text
        else:
            # Generate a descriptive base text based on image type
            image_type = extracted_image.image_type or "photo"
            if image_type in ["chart", "graph", "diagram"]:
                base_text = f"Chart or graph showing data visualization"
            else:
                base_text = f"Image: {image_type}"
        
        # Try to find surrounding text context from the same page
        # This helps provide context about what the chart shows
        context_text = ""
        if extracted_image.page and extracted_content and extracted_content.text:
            # For PDFs, we can try to extract text around the image's page
            # This is a simple approach - in a more sophisticated system, we'd track
            # exact positions and extract nearby text
            try:
                # Split text by pages if available (this is a simplified approach)
                # In practice, you might want to use the page boundaries from the extractor
                if hasattr(extracted_content, 'page_texts') and extracted_content.page_texts:
                    page_text = extracted_content.page_texts.get(extracted_image.page, "")
                    if page_text:
                        # Take first 200 chars as context (usually contains title/description)
                        context_snippet = page_text[:200].strip()
                        if context_snippet:
                            context_text = f"\n\nContext from page {extracted_image.page}: {context_snippet}"
            except Exception as e:
                logger.debug(f"Could not extract page context for image: {e}")
        
        # Build final text
        if context_text:
            return f"{base_text}{context_text}"
        else:
            return base_text
