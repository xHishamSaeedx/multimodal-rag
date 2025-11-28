"""
Image retrieval service.

Retrieves image chunks from Qdrant image_chunks collection.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any

from app.repositories.vector_repository import VectorRepository, VectorRepositoryError
from app.services.embedding.image_embedder import ImageEmbedder, EmbeddingError
from app.utils.exceptions import BaseAppException
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ImageRetrieverError(BaseAppException):
    """Raised when image retrieval operations fail."""
    pass


class ImageRetriever:
    """
    Image chunk retrieval service.
    
    Retrieves image chunks from Qdrant image_chunks collection.
    
    Features:
    - Semantic search using dense embeddings (768 dimensions for CLIP large)
    - Text-to-image search: encodes text queries into image embedding space (unified CLIP model)
    - Metadata filtering (document type, filename, etc.)
    - Configurable result limits
    """
    
    def __init__(
        self,
        embedder: Optional[ImageEmbedder] = None,
    ):
        """
        Initialize the image retriever.
        
        Args:
            embedder: Optional ImageEmbedder instance (creates new if not provided)
        """
        # Create vector repository for image_chunks collection
        self.vector_repo = VectorRepository(
            collection_name="image_chunks",
            vector_size=768,  # CLIP large produces 768 dimensions (matches text embeddings)
        )
        self.embedder = embedder or ImageEmbedder(model_type="clip")
        
        # Verify embedding dimension is set
        if self.embedder.embedding_dim is None:
            raise ImageRetrieverError(
                f"ImageEmbedder failed to initialize properly. embedding_dim is None. "
                f"Check logs for model loading errors.",
                {
                    "model_type": self.embedder.model_type,
                    "model_name": self.embedder.model_name,
                },
            )
        
        # Verify embedding dimension matches
        if self.embedder.embedding_dim != self.vector_repo.vector_size:
            raise ImageRetrieverError(
                f"Embedding dimension mismatch: embedder produces {self.embedder.embedding_dim} dimensions, "
                f"but image_chunks collection expects {self.vector_repo.vector_size} dimensions.",
                {
                    "embedder_dim": self.embedder.embedding_dim,
                    "vector_repo_dim": self.vector_repo.vector_size,
                },
            )
        
        logger.debug(
            "image_retriever_initialized",
            model=self.embedder.model_name,
            embedding_dim=self.embedder.embedding_dim,
            collection="image_chunks",
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate query embedding for image search.
        
        Uses ImageEmbedder's embed_text_query() method to encode text queries
        into the image embedding space (1024 dimensions).
        
        Args:
            query: Search query text
        
        Returns:
            Query embedding vector (1024 dimensions)
        """
        return self.embedder.embed_text_query(query)
    
    async def retrieve_with_embedding(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve image chunks using a pre-computed query embedding (async).
        
        Args:
            query_embedding: Pre-computed query embedding vector (1024 dimensions)
            limit: Maximum number of results to return (default: 10)
            filter_conditions: Optional filter conditions
        
        Returns:
            List of retrieved image chunks with image_path and caption included
        """
        try:
            # Search vectors in Qdrant image_chunks collection
            search_start = time.time()
            raw_results = await asyncio.to_thread(
                self.vector_repo.search_vectors,
                query_vector=query_embedding,
                limit=limit,
                filter_conditions=filter_conditions,
            )
            search_time = time.time() - search_start
            
            logger.debug(
                "image_chunks_vector_search_completed",
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
                    "chunk_text": payload.get("text", ""),  # Image description or caption
                    "image_path": payload.get("image_path", ""),  # Supabase storage path
                    "caption": payload.get("caption", ""),  # Image caption
                    "image_type": payload.get("image_type", "photo"),  # diagram, chart, photo, etc.
                    "filename": payload.get("filename", ""),
                    "document_type": payload.get("document_type", ""),
                    "source_path": payload.get("source_path", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_type": "image",  # Always image for this retriever
                    "embedding_type": "image",
                }
                
                # Add all other payload fields as metadata
                metadata = {}
                for key, value in payload.items():
                    if key not in [
                        "chunk_id",
                        "document_id",
                        "text",
                        "image_path",
                        "caption",
                        "image_type",
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
                "image_chunks_search_completed",
                results_count=len(results),
                method="vector_similarity",
            )
            return results
        
        except VectorRepositoryError as e:
            logger.error(
                "image_retrieval_error",
                error_type="VectorRepositoryError",
                error_message=str(e),
                exc_info=True,
            )
            raise ImageRetrieverError(
                f"Failed to retrieve image chunks: {str(e)}",
                {"error": str(e)},
            ) from e
        except Exception as e:
            logger.error(
                "image_retrieval_error",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            raise ImageRetrieverError(
                f"Unexpected error during image retrieval: {str(e)}",
                {"error": str(e)},
            ) from e

