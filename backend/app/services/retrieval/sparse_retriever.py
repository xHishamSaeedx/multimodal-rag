"""
BM25 sparse retrieval service.

Retrieves chunks using Elasticsearch BM25 search.
"""

import asyncio
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.repositories.sparse_repository import SparseRepository, SparseRepositoryError
from app.utils.exceptions import BaseAppException
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SparseRetrieverError(BaseAppException):
    """Raised when sparse retrieval operations fail."""
    pass


class SparseRetriever:
    """
    BM25 sparse retrieval service.
    
    Retrieves chunks using Elasticsearch BM25 keyword search.
    
    Features:
    - Full-text search with BM25 scoring
    - Metadata filtering (document type, filename, etc.)
    - Configurable result limits
    """
    
    def __init__(
        self,
        sparse_repository: Optional[SparseRepository] = None,
    ):
        """
        Initialize the sparse retriever.
        
        Args:
            sparse_repository: Optional SparseRepository instance (creates new if not provided)
        """
        self.sparse_repo = sparse_repository or SparseRepository()
        logger.debug("sparse_retriever_initialized")
    
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using BM25 search (async).
        
        Args:
            query: Search query text
            limit: Maximum number of results to return (default: 10)
            filter_conditions: Optional filter conditions:
                - document_id: Filter by document UUID
                - document_type: Filter by document type (pdf, docx, etc.)
                - filename: Filter by exact filename
                - source_path: Filter by source path
                - metadata.*: Filter by metadata fields (e.g., "metadata.tags": "important")
        
        Returns:
            List of retrieved chunks, each containing:
            - chunk_id: UUID string
            - document_id: UUID string
            - score: BM25 relevance score
            - chunk_text: Chunk text content
            - filename: Document filename
            - document_type: Document type
            - source_path: Source path
            - metadata: Chunk metadata
            - created_at: Creation timestamp
        
        Raises:
            SparseRetrieverError: If retrieval fails
        """
        try:
            if not query or not query.strip():
                logger.warning("sparse_retrieval_empty_query")
                return []
            
            logger.debug(
                "sparse_retrieval_start",
                query_preview=query[:50] if len(query) > 50 else query,
                limit=limit,
                has_filters=filter_conditions is not None,
            )
            
            # Run synchronous repository search in thread pool to avoid blocking
            results = await asyncio.to_thread(
                self.sparse_repo.search,
                query=query,
                limit=limit,
                filter_conditions=filter_conditions,
            )
            
            logger.info(
                "bm25_search_completed",
                results_count=len(results),
                method="BM25",
            )
            return results
        
        except SparseRepositoryError as e:
            logger.error(
                "sparse_retrieval_error",
                error_type="SparseRepositoryError",
                error_message=str(e),
                query_preview=query[:100] if query else "",
                exc_info=True,
            )
            raise SparseRetrieverError(
                f"Failed to retrieve chunks: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(
                "sparse_retrieval_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:100] if query else "",
                exc_info=True,
            )
            raise SparseRetrieverError(
                f"Unexpected error during sparse retrieval: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
    
    async def retrieve_by_document(
        self,
        query: str,
        document_id: UUID,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific document using BM25 search.
        
        Convenience method that filters by document_id.
        
        Args:
            query: Search query text
            document_id: Document UUID to filter by
            limit: Maximum number of results to return
        
        Returns:
            List of retrieved chunks (same format as retrieve())
        """
        return await self.retrieve(
            query=query,
            limit=limit,
            filter_conditions={"document_id": document_id},
        )
    
    async def retrieve_by_type(
        self,
        query: str,
        document_type: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from documents of a specific type using BM25 search.
        
        Convenience method that filters by document_type.
        
        Args:
            query: Search query text
            document_type: Document type to filter by (pdf, docx, txt, md)
            limit: Maximum number of results to return
        
        Returns:
            List of retrieved chunks (same format as retrieve())
        """
        return await self.retrieve(
            query=query,
            limit=limit,
            filter_conditions={"document_type": document_type},
        )
