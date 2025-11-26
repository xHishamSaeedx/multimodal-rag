"""
Document repository.

Handles all database operations for documents and chunks in Supabase (PostgreSQL).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4

from app.core.database import get_supabase_client, DatabaseError
from app.services.ingestion.chunker import Chunk
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class RepositoryError(BaseAppException):
    """Raised when repository operations fail."""
    pass


@dataclass
class Document:
    """
    Represents a document in the database.
    
    Attributes:
        id: Document UUID
        source_path: Path to raw document in MinIO
        filename: Original filename
        document_type: Document type (pdf, docx, txt, md)
        extracted_text: Full extracted text
        metadata: Additional metadata (JSONB)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    
    id: UUID
    source_path: str
    filename: str
    document_type: str
    extracted_text: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentRepository:
    """
    Repository for document and chunk database operations.
    
    Handles CRUD operations for:
    - documents table
    - chunks table
    """
    
    def __init__(self):
        """Initialize the document repository."""
        self.client = get_supabase_client()
        self.documents_table = "documents"
        self.chunks_table = "chunks"
    
    def create_document(
        self,
        source_path: str,
        filename: str,
        document_type: str,
        extracted_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """
        Create a new document record.
        
        Args:
            source_path: Path to raw document in MinIO (object_key)
            filename: Original filename
            document_type: Document type (pdf, docx, txt, md)
            extracted_text: Full extracted text content
            metadata: Optional metadata dictionary
        
        Returns:
            Document UUID
        
        Raises:
            RepositoryError: If creation fails
        """
        try:
            document_id = uuid4()
            now = datetime.utcnow().isoformat()
            
            document_data = {
                "id": str(document_id),
                "source_path": source_path,
                "filename": filename,
                "document_type": document_type,
                "extracted_text": extracted_text,
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now,
            }
            
            logger.debug(f"Creating document: {document_id} ({filename})")
            
            result = self.client.table(self.documents_table).insert(document_data).execute()
            
            if not result.data:
                raise RepositoryError(
                    "Failed to create document: No data returned",
                    {"document_id": str(document_id), "filename": filename},
                )
            
            logger.info(f"Created document: {document_id} ({filename})")
            return document_id
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error creating document: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to create document: {str(e)}",
                {"filename": filename, "error": str(e)},
            ) from e
    
    def get_document(self, document_id: UUID) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document UUID
        
        Returns:
            Document object or None if not found
        
        Raises:
            RepositoryError: If retrieval fails
        """
        try:
            result = (
                self.client.table(self.documents_table)
                .select("*")
                .eq("id", str(document_id))
                .execute()
            )
            
            if not result.data:
                return None
            
            doc_data = result.data[0]
            return self._document_from_dict(doc_data)
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error getting document: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to get document: {str(e)}",
                {"document_id": str(document_id), "error": str(e)},
            ) from e
    
    def get_document_by_source_path(self, source_path: str) -> Optional[Document]:
        """
        Get a document by source_path (object_key).
        
        Args:
            source_path: Source path (object_key) in MinIO
        
        Returns:
            Document object or None if not found
        
        Raises:
            RepositoryError: If retrieval fails
        """
        try:
            result = (
                self.client.table(self.documents_table)
                .select("*")
                .eq("source_path", source_path)
                .execute()
            )
            
            if not result.data:
                return None
            
            doc_data = result.data[0]
            return self._document_from_dict(doc_data)
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error getting document by source_path: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to get document by source_path: {str(e)}",
                {"source_path": source_path, "error": str(e)},
            ) from e
    
    def create_chunks(
        self,
        document_id: UUID,
        chunks: List[Chunk],
    ) -> List[UUID]:
        """
        Create chunk records for a document.
        
        Args:
            document_id: Parent document UUID
            chunks: List of Chunk objects to store
        
        Returns:
            List of created chunk UUIDs
        
        Raises:
            RepositoryError: If creation fails
        """
        try:
            if not chunks:
                logger.warning(f"No chunks to create for document: {document_id}")
                return []
            
            logger.debug(f"Creating {len(chunks)} chunks for document: {document_id}")
            
            # Prepare chunk data
            chunks_data = []
            for chunk in chunks:
                chunk_id = uuid4()
                chunk_data = {
                    "id": str(chunk_id),
                    "document_id": str(document_id),
                    "chunk_text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "metadata": {
                        **(chunk.metadata or {}),
                        "start_char_index": chunk.start_char_index,
                        "end_char_index": chunk.end_char_index,
                        "token_count": chunk.token_count,
                    },
                    "created_at": chunk.created_at.isoformat(),
                }
                chunks_data.append(chunk_data)
            
            # Insert chunks in batches (Supabase has limits)
            batch_size = 100
            created_ids = []
            
            for i in range(0, len(chunks_data), batch_size):
                batch = chunks_data[i:i + batch_size]
                result = self.client.table(self.chunks_table).insert(batch).execute()
                
                if result.data:
                    batch_ids = [UUID(item["id"]) for item in result.data]
                    created_ids.extend(batch_ids)
                    logger.debug(
                        f"Created batch of {len(batch_ids)} chunks "
                        f"({i + 1}-{min(i + batch_size, len(chunks_data))} of {len(chunks_data)})"
                    )
            
            logger.info(
                f"Created {len(created_ids)} chunks for document: {document_id}"
            )
            return created_ids
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error creating chunks: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to create chunks: {str(e)}",
                {"document_id": str(document_id), "chunk_count": len(chunks), "error": str(e)},
            ) from e
    
    def get_document_chunks(
        self,
        document_id: UUID,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document UUID
            limit: Optional limit on number of chunks
        
        Returns:
            List of chunk dictionaries
        
        Raises:
            RepositoryError: If retrieval fails
        """
        try:
            query = (
                self.client.table(self.chunks_table)
                .select("*")
                .eq("document_id", str(document_id))
                .order("chunk_index")
            )
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            
            return result.data or []
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error getting chunks: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to get chunks: {str(e)}",
                {"document_id": str(document_id), "error": str(e)},
            ) from e
    
    def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document and its chunks and tables.
        
        Note: This should cascade delete chunks if foreign key constraints are set up.
        
        Args:
            document_id: Document UUID
        
        Returns:
            True if deleted, False if not found
        
        Raises:
            RepositoryError: If deletion fails
        """
        try:
            # Delete tables first (associated with document)
            self.client.table("tables").delete().eq("document_id", str(document_id)).execute()
            logger.debug(f"Deleted tables for document: {document_id}")
            
            # Delete chunks (if cascade is not set up)
            self.client.table(self.chunks_table).delete().eq("document_id", str(document_id)).execute()
            
            # Delete document
            result = (
                self.client.table(self.documents_table)
                .delete()
                .eq("id", str(document_id))
                .execute()
            )
            
            deleted = bool(result.data)
            
            if deleted:
                logger.info(f"Deleted document: {document_id}")
            else:
                logger.warning(f"Document not found for deletion: {document_id}")
            
            return deleted
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to delete document: {str(e)}",
                {"document_id": str(document_id), "error": str(e)},
            ) from e
    
    def create_table(
        self,
        document_id: UUID,
        chunk_id: UUID,
        table_data: Dict[str, Any],
        table_markdown: str,
        table_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """
        Create a table record in the tables table.
        
        Args:
            document_id: Parent document UUID
            chunk_id: Associated chunk UUID
            table_data: Structured table data (JSON format)
            table_markdown: Markdown representation
            table_text: Flattened text representation
            metadata: Additional metadata (row_count, col_count, headers, etc.)
        
        Returns:
            Table UUID
        """
        try:
            table_id = uuid4()
            now = datetime.utcnow().isoformat()
            
            table_record = {
                "id": str(table_id),
                "document_id": str(document_id),
                "chunk_id": str(chunk_id),
                "table_data": table_data,
                "table_markdown": table_markdown,
                "table_text": table_text,
                "metadata": metadata or {},
                "created_at": now,
            }
            
            logger.debug(f"Creating table record: {table_id} for document: {document_id}")
            
            result = self.client.table("tables").insert(table_record).execute()
            
            if not result.data:
                raise RepositoryError(
                    "Failed to create table: No data returned",
                    {"table_id": str(table_id), "document_id": str(document_id)},
                )
            
            logger.info(f"Created table record: {table_id}")
            return table_id
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error creating table: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to create table: {str(e)}",
                {"document_id": str(document_id), "error": str(e)},
            ) from e
    
    def create_tables_batch(
        self,
        document_id: UUID,
        tables_data: List[Dict[str, Any]],
    ) -> List[UUID]:
        """
        Create multiple table records in batch.
        
        Args:
            document_id: Parent document UUID
            tables_data: List of table data dictionaries, each containing:
                - chunk_id: UUID
                - table_data: Dict (JSON format)
                - table_markdown: str
                - table_text: str
                - metadata: Dict (optional)
        
        Returns:
            List of created table UUIDs
        """
        try:
            if not tables_data:
                logger.warning(f"No tables to create for document: {document_id}")
                return []
            
            logger.debug(f"Creating {len(tables_data)} table(s) for document: {document_id}")
            
            now = datetime.utcnow().isoformat()
            tables_records = []
            
            for table_info in tables_data:
                table_id = uuid4()
                table_record = {
                    "id": str(table_id),
                    "document_id": str(document_id),
                    "chunk_id": str(table_info["chunk_id"]),
                    "table_data": table_info["table_data"],
                    "table_markdown": table_info["table_markdown"],
                    "table_text": table_info["table_text"],
                    "metadata": table_info.get("metadata", {}),
                    "created_at": now,
                }
                tables_records.append(table_record)
            
            # Insert in batches
            batch_size = 100
            created_ids = []
            
            for i in range(0, len(tables_records), batch_size):
                batch = tables_records[i:i + batch_size]
                result = self.client.table("tables").insert(batch).execute()
                
                if result.data:
                    batch_ids = [UUID(item["id"]) for item in result.data]
                    created_ids.extend(batch_ids)
                    logger.debug(
                        f"Created batch of {len(batch_ids)} tables "
                        f"({i + 1}-{min(i + batch_size, len(tables_records))} of {len(tables_records)})"
                    )
            
            logger.info(f"Created {len(created_ids)} table(s) for document: {document_id}")
            return created_ids
        
        except Exception as e:
            if isinstance(e, (RepositoryError, DatabaseError)):
                raise
            logger.error(f"Error creating tables batch: {str(e)}", exc_info=True)
            raise RepositoryError(
                f"Failed to create tables batch: {str(e)}",
                {"document_id": str(document_id), "table_count": len(tables_data), "error": str(e)},
            ) from e
    
    def _document_from_dict(self, data: Dict[str, Any]) -> Document:
        """Convert dictionary to Document object."""
        return Document(
            id=UUID(data["id"]),
            source_path=data["source_path"],
            filename=data["filename"],
            document_type=data["document_type"],
            extracted_text=data["extracted_text"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
        )
