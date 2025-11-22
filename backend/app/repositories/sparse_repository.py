"""
Sparse index repository.

Handles all BM25 operations in Elasticsearch (keyword search, indexing).
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import RequestError, NotFoundError
except ImportError:
    Elasticsearch = None
    RequestError = None
    NotFoundError = None

from app.core.database import get_elasticsearch_client, DatabaseError
from app.core.config import settings
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class SparseRepositoryError(BaseAppException):
    """Raised when sparse repository operations fail."""
    pass


class SparseRepository:
    """
    Repository for BM25 sparse search in Elasticsearch.
    
    Handles:
    - Indexing chunks with BM25 scoring
    - Searching chunks using BM25
    - Deleting chunks from index
    """
    
    def __init__(
        self,
        index_name: Optional[str] = None,
    ):
        """
        Initialize the sparse repository.
        
        Args:
            index_name: Elasticsearch index name (default: from config)
        """
        self.client = get_elasticsearch_client()
        self.index_name = index_name or settings.elasticsearch_index_name
        
        # Ensure index exists (will be created by init script, but check anyway)
        self._ensure_index_exists()
    
    def _ensure_index_exists(self) -> None:
        """Ensure the index exists, log warning if it doesn't."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                logger.warning(
                    f"Elasticsearch index '{self.index_name}' does not exist. "
                    f"Run: python scripts/init_elasticsearch.py"
                )
        except Exception as e:
            logger.warning(
                f"Could not verify index existence: {str(e)}. "
                f"Make sure Elasticsearch is running and index is initialized."
            )
    
    def index_chunks(
        self,
        chunk_ids: List[UUID],
        chunk_texts: List[str],
        document_ids: List[UUID],
        filenames: List[str],
        document_types: List[str],
        source_paths: List[str],
        metadata_list: List[Dict[str, Any]],
        created_at_list: Optional[List[datetime]] = None,
    ) -> bool:
        """
        Index chunks into Elasticsearch for BM25 search.
        
        Args:
            chunk_ids: List of chunk UUIDs
            chunk_texts: List of chunk text content (for BM25 search)
            document_ids: List of parent document UUIDs
            filenames: List of filenames
            document_types: List of document types (pdf, docx, etc.)
            source_paths: List of source paths (MinIO object keys)
            metadata_list: List of metadata dictionaries for each chunk
            created_at_list: Optional list of creation timestamps (default: current time)
        
        Returns:
            True if successful
        
        Raises:
            SparseRepositoryError: If indexing fails
        """
        try:
            if Elasticsearch is None:
                raise SparseRepositoryError(
                    "Elasticsearch client is not installed. "
                    "Install it with: pip install 'elasticsearch>=8.0.0,<9.0.0'",
                    {},
                )
            
            # Validate input lengths
            lengths = [
                len(chunk_ids),
                len(chunk_texts),
                len(document_ids),
                len(filenames),
                len(document_types),
                len(source_paths),
                len(metadata_list),
            ]
            
            if not all(length == lengths[0] for length in lengths):
                raise SparseRepositoryError(
                    "All input lists must have the same length",
                    {
                        "chunk_ids": len(chunk_ids),
                        "chunk_texts": len(chunk_texts),
                        "document_ids": len(document_ids),
                        "filenames": len(filenames),
                        "document_types": len(document_types),
                        "source_paths": len(source_paths),
                        "metadata_list": len(metadata_list),
                    },
                )
            
            if not chunk_ids:
                logger.warning("No chunks to index")
                return True
            
            # Prepare documents for bulk indexing
            actions = []
            now = datetime.utcnow()
            
            for i, (chunk_id, chunk_text, document_id, filename, doc_type, source_path, metadata) in enumerate(
                zip(chunk_ids, chunk_texts, document_ids, filenames, document_types, source_paths, metadata_list)
            ):
                # Get creation timestamp
                created_at = created_at_list[i] if created_at_list and i < len(created_at_list) else now
                
                # Prepare document for Elasticsearch
                doc = {
                    "chunk_id": str(chunk_id),
                    "document_id": str(document_id),
                    "chunk_text": chunk_text,
                    "filename": filename,
                    "document_type": doc_type,
                    "source_path": source_path,
                    "metadata": metadata or {},
                    "created_at": created_at.isoformat(),
                    "updated_at": now.isoformat(),
                }
                
                # Add bulk action (index operation)
                action = {
                    "_index": self.index_name,
                    "_id": str(chunk_id),  # Use chunk_id as document ID
                    "_source": doc,
                }
                actions.append(action)
            
            # Bulk index documents
            logger.debug(f"Indexing {len(actions)} chunks into Elasticsearch index '{self.index_name}'")
            
            from elasticsearch.helpers import bulk
            
            success_count, failed_items = bulk(
                self.client,
                actions,
                raise_on_error=False,  # Don't raise on individual failures
                request_timeout=settings.elasticsearch_timeout,
            )
            
            if failed_items:
                failed_count = len(failed_items)
                logger.warning(
                    f"Failed to index {failed_count} out of {len(actions)} chunks. "
                    f"Successfully indexed: {success_count}"
                )
                # Log first few failures for debugging
                for item in failed_items[:5]:
                    logger.debug(f"Failed item: {item}")
            
            logger.info(
                f"Successfully indexed {success_count} chunks into Elasticsearch index '{self.index_name}'"
            )
            
            # Refresh index to make documents searchable immediately
            try:
                self.client.indices.refresh(index=self.index_name)
            except Exception as refresh_error:
                logger.warning(f"Failed to refresh index (documents may not be immediately searchable): {refresh_error}")
            
            return True
        
        except Exception as e:
            if isinstance(e, SparseRepositoryError):
                raise
            logger.error(f"Error indexing chunks: {str(e)}", exc_info=True)
            raise SparseRepositoryError(
                f"Failed to index chunks: {str(e)}",
                {
                    "index_name": self.index_name,
                    "chunk_count": len(chunk_ids),
                    "error": str(e),
                },
            ) from e
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search chunks using BM25.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions (e.g., {"document_id": "...", "document_type": "pdf"})
        
        Returns:
            List of search results with score, chunk_id, and document data
            Each result contains:
            - chunk_id: UUID string
            - document_id: UUID string
            - score: BM25 relevance score
            - chunk_text: Chunk text content
            - filename: Document filename
            - document_type: Document type
            - source_path: Source path
            - metadata: Chunk metadata
        """
        try:
            if Elasticsearch is None:
                raise SparseRepositoryError(
                    "Elasticsearch client is not installed.",
                    {},
                )
            
            # Build query
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "chunk_text": {
                                        "query": query,
                                        "operator": "or",  # Match any term (OR logic)
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": limit,
                "_source": [
                    "chunk_id",
                    "document_id",
                    "chunk_text",
                    "filename",
                    "document_type",
                    "source_path",
                    "metadata",
                    "created_at",
                ],
            }
            
            # Add filters if provided
            if filter_conditions:
                filter_clauses = []
                
                for field, value in filter_conditions.items():
                    if field == "document_id":
                        filter_clauses.append({"term": {"document_id": str(value)}})
                    elif field == "document_type":
                        filter_clauses.append({"term": {"document_type": value}})
                    elif field == "filename":
                        filter_clauses.append({"term": {"filename.keyword": value}})
                    elif field == "source_path":
                        filter_clauses.append({"term": {"source_path": value}})
                    elif field.startswith("metadata."):
                        # Filter on metadata field (e.g., "metadata.tags")
                        metadata_field = field.replace("metadata.", "")
                        filter_clauses.append({"term": {f"metadata.{metadata_field}": value}})
                    else:
                        logger.warning(f"Unknown filter field: {field}, skipping")
                
                if filter_clauses:
                    query_body["query"]["bool"]["filter"] = filter_clauses
            
            logger.debug(
                f"Searching Elasticsearch index '{self.index_name}' "
                f"with query: '{query[:50]}...' (limit: {limit})"
            )
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query_body,
                request_timeout=settings.elasticsearch_timeout,
            )
            
            # Format results
            results = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit["_source"]
                results.append({
                    "chunk_id": source.get("chunk_id"),
                    "document_id": source.get("document_id"),
                    "score": hit.get("_score", 0.0),
                    "chunk_text": source.get("chunk_text", ""),
                    "filename": source.get("filename", ""),
                    "document_type": source.get("document_type", ""),
                    "source_path": source.get("source_path", ""),
                    "metadata": source.get("metadata", {}),
                    "created_at": source.get("created_at"),
                })
            
            logger.debug(f"Found {len(results)} results")
            return results
        
        except Exception as e:
            if isinstance(e, SparseRepositoryError):
                raise
            logger.error(f"Error searching chunks: {str(e)}", exc_info=True)
            raise SparseRepositoryError(
                f"Failed to search chunks: {str(e)}",
                {
                    "index_name": self.index_name,
                    "query": query[:100] if query else "",
                    "error": str(e),
                },
            ) from e
    
    def delete_chunks(self, chunk_ids: List[UUID]) -> bool:
        """
        Delete chunks from Elasticsearch index.
        
        Args:
            chunk_ids: List of chunk UUIDs to delete
        
        Returns:
            True if successful
        """
        try:
            if not chunk_ids:
                return True
            
            if Elasticsearch is None:
                raise SparseRepositoryError(
                    "Elasticsearch client is not installed.",
                    {},
                )
            
            # Delete documents by ID
            from elasticsearch.helpers import bulk
            
            actions = []
            for chunk_id in chunk_ids:
                actions.append({
                    "_op_type": "delete",
                    "_index": self.index_name,
                    "_id": str(chunk_id),
                })
            
            logger.debug(f"Deleting {len(actions)} chunks from Elasticsearch index '{self.index_name}'")
            
            success_count, failed_items = bulk(
                self.client,
                actions,
                raise_on_error=False,
                request_timeout=settings.elasticsearch_timeout,
            )
            
            if failed_items:
                failed_count = len(failed_items)
                logger.warning(
                    f"Failed to delete {failed_count} out of {len(actions)} chunks. "
                    f"Successfully deleted: {success_count}"
                )
            
            logger.info(f"Deleted {success_count} chunks from Elasticsearch index")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}", exc_info=True)
            raise SparseRepositoryError(
                f"Failed to delete chunks: {str(e)}",
                {"chunk_ids_count": len(chunk_ids), "error": str(e)},
            ) from e
    
    def delete_chunks_by_document(self, document_id: UUID) -> bool:
        """
        Delete all chunks for a document from Elasticsearch index.
        
        Args:
            document_id: Document UUID
        
        Returns:
            True if successful
        """
        try:
            if Elasticsearch is None:
                raise SparseRepositoryError(
                    "Elasticsearch client is not installed.",
                    {},
                )
            
            # Use delete_by_query to delete all chunks for a document
            query_body = {
                "query": {
                    "term": {
                        "document_id": str(document_id)
                    }
                }
            }
            
            logger.debug(
                f"Deleting all chunks for document {document_id} "
                f"from Elasticsearch index '{self.index_name}'"
            )
            
            response = self.client.delete_by_query(
                index=self.index_name,
                body=query_body,
                request_timeout=settings.elasticsearch_timeout,
            )
            
            deleted_count = response.get("deleted", 0)
            logger.info(
                f"Deleted {deleted_count} chunks for document {document_id} "
                f"from Elasticsearch index"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting chunks by document: {str(e)}", exc_info=True)
            raise SparseRepositoryError(
                f"Failed to delete chunks by document: {str(e)}",
                {"document_id": str(document_id), "error": str(e)},
            ) from e
