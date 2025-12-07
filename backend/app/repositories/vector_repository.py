"""
Vector repository.

Handles all vector operations in Qdrant (vector search, storage).
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

try:
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        HnswConfigDiff,
        SearchParams,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
    )
except ImportError:
    Distance = None
    VectorParams = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    HnswConfigDiff = None
    SearchParams = None
    ScalarQuantization = None
    ScalarQuantizationConfig = None
    ScalarType = None

from app.core.database import get_qdrant_client, DatabaseError
from app.core.config import settings
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class VectorRepositoryError(BaseAppException):
    """Raised when vector repository operations fail."""
    pass


class VectorRepository:
    """
    Repository for vector storage and retrieval in Qdrant.
    
    Handles:
    - Storing embeddings with chunk metadata
    - Vector similarity search
    - Collection management
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        """
        Initialize the vector repository.
        
        Args:
            collection_name: Qdrant collection name (default: from config)
            vector_size: Vector dimension size (default: from config)
        """
        self.client = get_qdrant_client()
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size or settings.qdrant_vector_size
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create if it doesn't. Check dimension match if exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                
                if VectorParams is None or Distance is None:
                    raise VectorRepositoryError(
                        "Qdrant models not available. Check qdrant-client installation.",
                        {},
                    )
                
                # Optimize HNSW parameters for speed (especially for image_chunks)
                # Lower m and ef_construct = faster search and indexing
                hnsw_config = None
                quantization_config = None
                
                if HnswConfigDiff is not None:
                    if self.collection_name == "image_chunks":
                        # Aggressively optimized for speed: very low m and ef_construct for fastest image search
                        hnsw_config = HnswConfigDiff(
                            m=4,              # Very low m (default 16) = minimal connections = fastest traversal
                            ef_construct=64,   # Very low ef_construct (default 200) = fastest index building
                        )
                        
                        # Enable scalar quantization for image_chunks (20-40ms speed improvement)
                        # Converts float32 vectors to int8, reducing memory by 75% and speeding up distance calculations
                        if ScalarQuantization is not None and ScalarQuantizationConfig is not None and ScalarType is not None:
                            quantization_config = ScalarQuantization(
                                scalar=ScalarQuantizationConfig(
                                    type=ScalarType.INT8,  # 8-bit integers (vs 32-bit floats)
                                    quantile=0.99,         # Exclude extreme 1% values for better accuracy
                                    always_ram=True,       # Keep quantized vectors in RAM for fastest access
                                )
                            )
                            logger.info(
                                f"Using aggressive HNSW + scalar quantization for image_chunks: "
                                f"m=4, ef_construct=64, INT8 quantization (ultra-fast, target <50ms)"
                            )
                        else:
                            logger.info(
                                f"Using aggressive HNSW config for image_chunks: m=4, ef_construct=64 (ultra-fast, target <50ms)"
                            )
                    else:
                        # Balanced settings for text/table chunks (still faster than defaults)
                        hnsw_config = HnswConfigDiff(
                            m=12,             # Moderate m for balance between speed and accuracy
                            ef_construct=128, # Moderate ef_construct
                        )
                        logger.debug(
                            f"Using balanced HNSW config: m=12, ef_construct=128"
                        )
                
                vectors_config_params = {
                    "size": self.vector_size,
                    "distance": Distance.COSINE,  # Cosine similarity for normalized embeddings
                }
                if hnsw_config is not None:
                    vectors_config_params["hnsw_config"] = hnsw_config
                
                # Build collection creation parameters
                collection_params = {
                    "collection_name": self.collection_name,
                    "vectors_config": VectorParams(**vectors_config_params),
                }
                
                # Add quantization config if available (for image_chunks)
                if quantization_config is not None:
                    collection_params["quantization_config"] = quantization_config
                
                self.client.create_collection(**collection_params)
                logger.info(
                    f"Created collection '{self.collection_name}' "
                    f"with vector size {self.vector_size}"
                    f"{' (HNSW optimized)' if hnsw_config else ''}"
                )
            else:
                # Collection exists - verify dimension matches
                collection_info = self.client.get_collection(self.collection_name)
                existing_vector_size = collection_info.config.params.vectors.size
                
                if existing_vector_size != self.vector_size:
                    # Dimension mismatch - this means the collection was created with a different
                    # model dimension. We need to match what the model actually produces.
                    logger.error(
                        f"❌ DIMENSION MISMATCH: Collection '{self.collection_name}' was created with "
                        f"{existing_vector_size} dimensions, but current embedding model produces "
                        f"{self.vector_size} dimensions.\n"
                        f"\n"
                        f"To fix this:\n"
                        f"1. Delete the existing collection and let it be recreated with correct dimension, OR\n"
                        f"2. Run: python backend/scripts/qdrant/init_qdrant.py --recreate --vector-size {self.vector_size}\n"
                        f"\n"
                        f"Current embedding model dimension: {self.vector_size}"
                    )
                    raise VectorRepositoryError(
                        f"Dimension mismatch: Collection has {existing_vector_size} dimensions, "
                        f"but embedding model produces {self.vector_size} dimensions. "
                        f"Recreate collection with: python backend/scripts/qdrant/init_qdrant.py --recreate --vector-size {self.vector_size}",
                        {
                            "collection_name": self.collection_name,
                            "existing_vector_size": existing_vector_size,
                            "expected_vector_size": self.vector_size,
                        },
                    )
                else:
                    logger.debug(
                        f"Collection '{self.collection_name}' already exists "
                        f"with correct vector size {self.vector_size}"
                    )
        except Exception as e:
            if isinstance(e, VectorRepositoryError):
                raise
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise VectorRepositoryError(
                f"Failed to ensure collection exists: {str(e)}",
                {"collection_name": self.collection_name, "error": str(e)},
            ) from e
    
    def store_table_vectors(
        self,
        chunk_ids: List[UUID],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> bool:
        """
        Store table vectors in Qdrant table_chunks collection.
        
        Args:
            chunk_ids: List of chunk UUIDs (used as point IDs)
            embeddings: List of embedding vectors
            payloads: List of payload dictionaries (metadata for each vector)
        
        Returns:
            True if successful
        
        Raises:
            VectorRepositoryError: If storage fails
        """
        # Use the same logic as store_vectors but with table_chunks collection
        original_collection = self.collection_name
        try:
            # Temporarily switch to table_chunks collection
            self.collection_name = "table_chunks"
            self._ensure_collection_exists()
            
            # Use the same storage logic
            return self.store_vectors(chunk_ids, embeddings, payloads)
        finally:
            # Restore original collection name
            self.collection_name = original_collection
    
    def store_vectors(
        self,
        chunk_ids: List[UUID],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> bool:
        """
        Store vectors in Qdrant.
        
        Args:
            chunk_ids: List of chunk UUIDs (used as point IDs)
            embeddings: List of embedding vectors
            payloads: List of payload dictionaries (metadata for each vector)
        
        Returns:
            True if successful
        
        Raises:
            VectorRepositoryError: If storage fails
        """
        try:
            if len(chunk_ids) != len(embeddings) or len(chunk_ids) != len(payloads):
                raise VectorRepositoryError(
                    "chunk_ids, embeddings, and payloads must have the same length",
                    {
                        "chunk_ids_count": len(chunk_ids),
                        "embeddings_count": len(embeddings),
                        "payloads_count": len(payloads),
                    },
                )
            
            if not chunk_ids:
                logger.warning("No vectors to store")
                return True
            
            # Validate vector dimensions
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self.vector_size:
                    raise VectorRepositoryError(
                        f"Vector dimension mismatch: expected {self.vector_size}, "
                        f"got {len(embedding)} at index {i}",
                        {"index": i, "expected": self.vector_size, "got": len(embedding)},
                    )
            
            # Convert UUIDs to strings for Qdrant point IDs
            points = []
            for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads):
                # Qdrant uses string IDs, convert UUID to string
                point_id = str(chunk_id)
                
                # Convert chunk_id in payload to string as well
                payload_with_string_id = {**payload, "chunk_id": str(payload.get("chunk_id", chunk_id))}
                if "document_id" in payload_with_string_id:
                    payload_with_string_id["document_id"] = str(payload_with_string_id["document_id"])
                
                if PointStruct is None:
                    raise VectorRepositoryError(
                        "Qdrant PointStruct not available. Check qdrant-client installation.",
                        {},
                    )
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload_with_string_id,
                )
                points.append(point)
            
            logger.debug(
                f"Storing {len(points)} vectors in collection '{self.collection_name}'"
            )
            
            # Upsert points (insert or update)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            logger.info(
                f"Successfully stored {len(points)} vectors in collection '{self.collection_name}'"
            )
            return True
        
        except Exception as e:
            if isinstance(e, VectorRepositoryError):
                raise
            logger.error(f"Error storing vectors: {str(e)}", exc_info=True)
            raise VectorRepositoryError(
                f"Failed to store vectors: {str(e)}",
                {
                    "collection_name": self.collection_name,
                    "vector_count": len(chunk_ids),
                    "error": str(e),
                },
            ) from e
    
    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        ef: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_conditions: Optional filter conditions (e.g., {"document_id": "..."})
            ef: Optional HNSW ef parameter (candidate list size during search).
                Lower values = faster search but potentially lower recall.
                Default: None (uses collection default, typically 100).
                Recommended: 16-32 for image_chunks (speed-optimized), 32-64 for others.
        
        Returns:
            List of search results with score, id, and payload
        """
        try:
            if len(query_vector) != self.vector_size:
                raise VectorRepositoryError(
                    f"Query vector dimension mismatch: expected {self.vector_size}, "
                    f"got {len(query_vector)}",
                    {"expected": self.vector_size, "got": len(query_vector)},
                )
            
            # Build filter if conditions provided
            qdrant_filter = None
            if filter_conditions:
                if FieldCondition is None or MatchValue is None or Filter is None:
                    raise VectorRepositoryError(
                        "Qdrant filter models not available.",
                        {},
                    )
                
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=str(value)),
                        )
                    )
                
                qdrant_filter = Filter(must=conditions)
            
            # Optimize ef parameter for speed (especially for image_chunks)
            # Lower ef = faster search but potentially lower recall
            if ef is None:
                if self.collection_name == "image_chunks":
                    ef = 4   # Ultra-aggressive optimization for fastest image search (target <50ms)
                else:
                    ef = 32  # Balanced for text/table search
            
            logger.debug(
                f"Searching vectors in collection '{self.collection_name}' "
                f"(limit: {limit}, ef: {ef})"
            )
            
            # Perform search using query_points (new API in qdrant-client 1.16+)
            # query_points replaces the deprecated search() method
            # Use SearchParams to set ef for HNSW search optimization
            query_params = {}
            if SearchParams is not None:
                query_params["search_params"] = SearchParams(hnsw_ef=ef)
            
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                **query_params,
            )
            
            # Extract results from QueryResponse object
            results = query_result.points
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                })
            
            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results
        
        except Exception as e:
            if isinstance(e, VectorRepositoryError):
                raise
            logger.error(f"Error searching vectors: {str(e)}", exc_info=True)
            raise VectorRepositoryError(
                f"Failed to search vectors: {str(e)}",
                {"collection_name": self.collection_name, "error": str(e)},
            ) from e
    
    def delete_vectors(self, chunk_ids: List[UUID]) -> bool:
        """
        Delete vectors by chunk IDs.
        
        Args:
            chunk_ids: List of chunk UUIDs to delete
        
        Returns:
            True if successful
        """
        try:
            if not chunk_ids:
                return True
            
            point_ids = [str(chunk_id) for chunk_id in chunk_ids]
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )
            
            logger.info(f"Deleted {len(point_ids)} vectors from collection")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}", exc_info=True)
            raise VectorRepositoryError(
                f"Failed to delete vectors: {str(e)}",
                {"chunk_ids_count": len(chunk_ids), "error": str(e)},
            ) from e
    
    def delete_vectors_by_document(self, document_id: UUID) -> bool:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: Document UUID
        
        Returns:
            True if successful
        """
        try:
            if Filter is None or FieldCondition is None or MatchValue is None:
                raise VectorRepositoryError(
                    "Qdrant filter models not available.",
                    {},
                )
            
            # Build filter for document_id
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    )
                ]
            )
            
            # Use scroll to get all points, then delete
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=10000,  # Large limit to get all points
            )
            
            if scroll_result[0]:  # Points found
                point_ids = [point.id for point in scroll_result[0]]
                if point_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids,
                    )
                    logger.info(
                        f"Deleted {len(point_ids)} vectors for document {document_id}"
                    )
            
            return True
        
        except Exception as e:
            if isinstance(e, VectorRepositoryError):
                raise
            logger.error(f"Error deleting vectors by document: {str(e)}", exc_info=True)
            raise VectorRepositoryError(
                f"Failed to delete vectors by document: {str(e)}",
                {"document_id": str(document_id), "error": str(e)},
            ) from e
    
    def delete_table_vectors_by_document(self, document_id: UUID) -> bool:
        """
        Delete all table vectors for a document from table_chunks collection.
        
        Args:
            document_id: Document UUID
        
        Returns:
            True if successful (or if collection doesn't exist)
        """
        # Use the same logic as delete_vectors_by_document but with table_chunks collection
        original_collection = self.collection_name
        try:
            # Temporarily switch to table_chunks collection
            self.collection_name = "table_chunks"
            
            # Check if collection exists before trying to delete
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Collection doesn't exist, nothing to delete
                logger.debug(f"Table chunks collection '{self.collection_name}' does not exist, nothing to delete")
                return True
            
            # Use the same deletion logic
            return self.delete_vectors_by_document(document_id)
        finally:
            # Restore original collection name
            self.collection_name = original_collection
    
    def delete_image_vectors_by_document(self, document_id: UUID) -> bool:
        """
        Delete all image vectors for a document from image_chunks collection.
        
        Args:
            document_id: Document UUID
        
        Returns:
            True if successful (or if collection doesn't exist)
        """
        # Use the same logic as delete_vectors_by_document but with image_chunks collection
        original_collection = self.collection_name
        try:
            # Temporarily switch to image_chunks collection
            self.collection_name = "image_chunks"
            
            # Check if collection exists before trying to delete
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Collection doesn't exist, nothing to delete
                logger.debug(f"Image chunks collection '{self.collection_name}' does not exist, nothing to delete")
                return True
            
            # Use the same deletion logic
            return self.delete_vectors_by_document(document_id)
        finally:
            # Restore original collection name
            self.collection_name = original_collection