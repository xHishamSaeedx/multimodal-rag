"""
BM25 sparse retrieval service.

Retrieves chunks using Elasticsearch BM25 search.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.repositories.sparse_repository import SparseRepository, SparseRepositoryError
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


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
        logger.debug("Initialized SparseRetriever")
    
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using BM25 search.
        
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
                logger.warning("Empty query provided to sparse retriever")
                return []
            
            logger.debug(
                f"Retrieving chunks with BM25: query='{query[:50]}...', "
                f"limit={limit}, filters={filter_conditions}"
            )
            
            # Search using sparse repository
            results = self.sparse_repo.search(
                query=query,
                limit=limit,
                filter_conditions=filter_conditions,
            )
            
            logger.info(f"Retrieved {len(results)} chunks using BM25 search")
            return results
        
        except SparseRepositoryError as e:
            logger.error(f"Sparse repository error during retrieval: {str(e)}", exc_info=True)
            raise SparseRetrieverError(
                f"Failed to retrieve chunks: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during sparse retrieval: {str(e)}", exc_info=True)
            raise SparseRetrieverError(
                f"Unexpected error during sparse retrieval: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
    
    def retrieve_by_document(
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
        return self.retrieve(
            query=query,
            limit=limit,
            filter_conditions={"document_id": document_id},
        )
    
    def retrieve_by_type(
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
        return self.retrieve(
            query=query,
            limit=limit,
            filter_conditions={"document_type": document_type},
        )
