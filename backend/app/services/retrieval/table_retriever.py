"""
Table retrieval service.

Retrieves table chunks from Qdrant table_chunks collection.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any

from app.repositories.vector_repository import VectorRepository, VectorRepositoryError
from app.services.embedding.text_embedder import TextEmbedder, EmbeddingError
from app.utils.exceptions import BaseAppException
from app.utils.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class TableRetrieverError(BaseAppException):
    """Raised when table retrieval operations fail."""
    pass


class TableRetriever:
    """
    Table chunk retrieval service.
    
    Retrieves table chunks from Qdrant table_chunks collection.
    
    Features:
    - Semantic search using dense embeddings
    - Metadata filtering (document type, filename, etc.)
    - Configurable result limits
    - Uses same embedding model as text chunks
    """
    
    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
    ):
        """
        Initialize the table retriever.
        
        Args:
            embedder: Optional TextEmbedder instance (creates new if not provided)
        """
        # Create vector repository for table_chunks collection
        self.vector_repo = VectorRepository(
            collection_name="table_chunks",
            vector_size=settings.embedding_dimension,  # Same dimension as text chunks
        )
        self.embedder = embedder or TextEmbedder()
        
        # Verify embedding dimension matches
        if self.embedder.embedding_dim != self.vector_repo.vector_size:
            raise TableRetrieverError(
                f"Embedding dimension mismatch: embedder produces {self.embedder.embedding_dim} dimensions, "
                f"but table_chunks collection expects {self.vector_repo.vector_size} dimensions.",
                {
                    "embedder_dim": self.embedder.embedding_dim,
                    "vector_repo_dim": self.vector_repo.vector_size,
                },
            )
        
        logger.debug(
            "table_retriever_initialized",
            model=self.embedder.model_name,
            embedding_dim=self.embedder.embedding_dim,
            collection="table_chunks",
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate query embedding for table search.
        
        Args:
            query: Search query text
        
        Returns:
            Query embedding vector
        """
        return self.embedder.generate_query_embedding(query)
    
    async def retrieve_with_embedding(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve table chunks using a pre-computed query embedding (async).
        
        Args:
            query_embedding: Pre-computed query embedding vector
            limit: Maximum number of results to return (default: 10)
            filter_conditions: Optional filter conditions
        
        Returns:
            List of retrieved table chunks with table_markdown included
        """
        try:
            # Search vectors in Qdrant table_chunks collection
            search_start = time.time()
            raw_results = await asyncio.to_thread(
                self.vector_repo.search_vectors,
                query_vector=query_embedding,
                limit=limit,
                filter_conditions=filter_conditions,
            )
            search_time = time.time() - search_start
            
            logger.debug(
                "table_chunks_vector_search_completed",
                duration_seconds=round(search_time, 3),
                results_count=len(raw_results),
            )
            
            # Format results
            results = []
            for result in raw_results:
                payload = result.get("payload", {})
                
                # Extract fields from payload
                formatted_result = {
                    "chunk_id": payload.get("chunk_id", result.get("id", "")),
                    "document_id": payload.get("document_id", ""),
                    "score": result.get("score", 0.0),
                    "chunk_text": payload.get("text", ""),  # Flattened text
                    "table_markdown": payload.get("table_markdown", ""),  # Markdown format
                    "table_data": payload.get("table_data", {}),  # JSON format
                    "filename": payload.get("filename", ""),
                    "document_type": payload.get("document_type", ""),
                    "source_path": payload.get("source_path", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_type": "table",  # Always table for this retriever
                    "embedding_type": "table",
                }
                
                # Add all other payload fields as metadata
                metadata = {}
                for key, value in payload.items():
                    if key not in [
                        "chunk_id",
                        "document_id",
                        "text",
                        "table_markdown",
                        "table_data",
                        "filename",
                        "document_type",
                        "source_path",
                        "chunk_index",
                        "chunk_type",
                        "embedding_type",
                    ]:
                        metadata[key] = value
                
                formatted_result["metadata"] = metadata
                results.append(formatted_result)
            
            logger.info(
                "table_chunks_search_completed",
                results_count=len(results),
                method="vector_similarity",
            )
            return results
        
        except VectorRepositoryError as e:
            logger.error(
                "table_retrieval_error",
                error_type="VectorRepositoryError",
                error_message=str(e),
                exc_info=True,
            )
            raise TableRetrieverError(
                f"Failed to retrieve table chunks: {str(e)}",
                {"error": str(e)},
            ) from e
        except Exception as e:
            logger.error(
                "table_retrieval_error",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            raise TableRetrieverError(
                f"Unexpected error during table retrieval: {str(e)}",
                {"error": str(e)},
            ) from e

