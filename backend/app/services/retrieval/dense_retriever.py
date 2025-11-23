"""
Dense vector retrieval service.

Retrieves chunks using Qdrant vector similarity search.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.repositories.vector_repository import VectorRepository, VectorRepositoryError
from app.services.embedding.text_embedder import TextEmbedder, EmbeddingError
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class DenseRetrieverError(BaseAppException):
    """Raised when dense retrieval operations fail."""
    pass


class DenseRetriever:
    """
    Dense vector retrieval service.
    
    Retrieves chunks using Qdrant vector similarity search.
    
    Features:
    - Semantic search using dense embeddings
    - Metadata filtering (document type, filename, etc.)
    - Configurable result limits
    - Automatic query embedding generation
    """
    
    def __init__(
        self,
        vector_repository: Optional[VectorRepository] = None,
        embedder: Optional[TextEmbedder] = None,
    ):
        """
        Initialize the dense retriever.
        
        Args:
            vector_repository: Optional VectorRepository instance (creates new if not provided)
            embedder: Optional TextEmbedder instance (creates new if not provided)
        """
        self.vector_repo = vector_repository or VectorRepository()
        self.embedder = embedder or TextEmbedder()
        
        # Verify embedding dimension matches vector repository
        if self.embedder.embedding_dim != self.vector_repo.vector_size:
            raise DenseRetrieverError(
                f"Embedding dimension mismatch: embedder produces {self.embedder.embedding_dim} dimensions, "
                f"but vector repository expects {self.vector_repo.vector_size} dimensions. "
                f"Ensure the embedding model matches the Qdrant collection configuration.",
                {
                    "embedder_dim": self.embedder.embedding_dim,
                    "vector_repo_dim": self.vector_repo.vector_size,
                },
            )
        
        logger.debug(
            f"Initialized DenseRetriever: model={self.embedder.model_name}, "
            f"dim={self.embedder.embedding_dim}"
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate query embedding (separate from retrieval timing).
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector as list of floats
        """
        query_text = query.strip()
        
        # For e5-base-v2, we need "query: " prefix instead of "passage: "
        if "e5" in self.embedder.model_name.lower():
            query_text_with_prefix = f"query: {query_text}"
            embedding = self.embedder.model.encode(
                query_text_with_prefix,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.tolist()
        else:
            return self.embedder.embed_text(query_text)
    
    async def retrieve_with_embedding(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using a pre-computed query embedding (async).
        
        This method does NOT include embedding generation time in retrieval timing.
        
        Args:
            query_embedding: Pre-computed query embedding vector
            limit: Maximum number of results to return (default: 10)
            filter_conditions: Optional filter conditions
            
        Returns:
            List of retrieved chunks (same format as retrieve())
        """
        try:
            # Search vectors in Qdrant (run in thread pool to avoid blocking)
            search_start = time.time()
            raw_results = await asyncio.to_thread(
                self.vector_repo.search_vectors,
                query_vector=query_embedding,
                limit=limit,
                filter_conditions=filter_conditions,
            )
            search_time = time.time() - search_start
            logger.debug(f"Qdrant vector search completed in {search_time:.3f}s")
            
            # Format results to match SparseRetriever format
            results = []
            for result in raw_results:
                payload = result.get("payload", {})
                
                # Extract common fields from payload
                formatted_result = {
                    "chunk_id": payload.get("chunk_id", result.get("id", "")),
                    "document_id": payload.get("document_id", ""),
                    "score": result.get("score", 0.0),  # Cosine similarity score
                    "chunk_text": payload.get("text", ""),
                    "filename": payload.get("filename", ""),
                    "document_type": payload.get("document_type", ""),
                    "source_path": payload.get("source_path", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_type": payload.get("chunk_type", "text"),
                }
                
                # Add all other payload fields as metadata
                metadata = {}
                for key, value in payload.items():
                    if key not in [
                        "chunk_id",
                        "document_id",
                        "text",
                        "filename",
                        "document_type",
                        "source_path",
                        "chunk_index",
                        "chunk_type",
                    ]:
                        metadata[key] = value
                
                formatted_result["metadata"] = metadata
                results.append(formatted_result)
            
            logger.info(f"Retrieved {len(results)} chunks using vector similarity search")
            return results
        
        except VectorRepositoryError as e:
            logger.error(f"Vector repository error during retrieval: {str(e)}", exc_info=True)
            raise DenseRetrieverError(
                f"Failed to retrieve chunks: {str(e)}",
                {"error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during dense retrieval: {str(e)}", exc_info=True)
            raise DenseRetrieverError(
                f"Unexpected error during dense retrieval: {str(e)}",
                {"error": str(e)},
            ) from e
    
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using vector similarity search (async).
        
        Args:
            query: Search query text (will be embedded automatically)
            limit: Maximum number of results to return (default: 10)
            filter_conditions: Optional filter conditions:
                - document_id: Filter by document UUID
                - document_type: Filter by document type (pdf, docx, etc.)
                - filename: Filter by exact filename
                - source_path: Filter by source path
                - Any other payload field from Qdrant
        
        Returns:
            List of retrieved chunks, each containing:
            - chunk_id: UUID string
            - document_id: UUID string
            - score: Cosine similarity score (higher is more similar)
            - chunk_text: Chunk text content (from payload)
            - filename: Document filename (from payload)
            - document_type: Document type (from payload)
            - source_path: Source path (from payload)
            - metadata: Chunk metadata (from payload)
            - Additional fields from payload
        
        Raises:
            DenseRetrieverError: If retrieval fails
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided to dense retriever")
                return []
            
            logger.debug(
                f"Retrieving chunks with vector search: query='{query[:50]}...', "
                f"limit={limit}, filters={filter_conditions}"
            )
            
            # Generate query embedding (this is preprocessing, not retrieval)
            # Run in thread pool since embedding model might block
            query_embedding = await asyncio.to_thread(
                self.generate_query_embedding,
                query
            )
            
            # Retrieve using pre-computed embedding
            return await self.retrieve_with_embedding(
                query_embedding=query_embedding,
                limit=limit,
                filter_conditions=filter_conditions,
            )
        
        except EmbeddingError as e:
            logger.error(f"Embedding error during dense retrieval: {str(e)}", exc_info=True)
            raise DenseRetrieverError(
                f"Failed to generate query embedding: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
        except VectorRepositoryError as e:
            logger.error(f"Vector repository error during retrieval: {str(e)}", exc_info=True)
            raise DenseRetrieverError(
                f"Failed to retrieve chunks: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during dense retrieval: {str(e)}", exc_info=True)
            raise DenseRetrieverError(
                f"Unexpected error during dense retrieval: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
    
    async def retrieve_by_document(
        self,
        query: str,
        document_id: UUID,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific document using vector similarity search.
        
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
        Retrieve chunks from documents of a specific type using vector similarity search.
        
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
