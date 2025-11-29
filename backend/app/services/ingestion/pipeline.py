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
from app.services.ingestion.image_extractor import ExtractedImage
from app.services.embedding.text_embedder import TextEmbedder, EmbeddingError
from app.services.embedding.image_embedder import ImageEmbedder
from app.services.storage import MinIOStorage
from app.services.storage.supabase_storage import SupabaseImageStorage
from app.services.vision import VisionProcessorFactory
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
            enable_deduplication=True,
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
                
                # Generate embeddings in batch
                text_embeddings = self.embedder.embed_batch(
                    texts=chunk_texts,
                    show_progress=True,
                )
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
                
                # Generate embeddings in batch (same model as text)
                table_embeddings = self.embedder.embed_batch(
                    texts=table_texts,
                    show_progress=True,
                )
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
                    
                    # Generate embeddings in batch
                    image_embeddings = self.image_embedder.embed_batch(
                        image_bytes_list=image_bytes_list,
                        show_progress=True,
                    )
                    
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
