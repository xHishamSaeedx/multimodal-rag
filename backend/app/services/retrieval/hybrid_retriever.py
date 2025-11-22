"""
Hybrid retrieval service.

Combines BM25 (sparse) and vector (dense) retrieval results.
Handles merging, deduplication, and score normalization.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.services.retrieval.sparse_retriever import SparseRetriever, SparseRetrieverError
from app.services.retrieval.dense_retriever import DenseRetriever, DenseRetrieverError
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class HybridRetrieverError(BaseAppException):
    """Raised when hybrid retrieval operations fail."""
    pass


class HybridRetriever:
    """
    Hybrid retrieval service that combines BM25 (sparse) and vector (dense) search.
    
    Features:
    - Parallel retrieval from both indexes
    - Result merging and deduplication
    - Score normalization (optional)
    - Configurable retrieval limits
    - Metadata filtering support
    """
    
    def __init__(
        self,
        sparse_retriever: Optional[SparseRetriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        normalize_scores: bool = True,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            sparse_retriever: Optional SparseRetriever instance (creates new if not provided)
            dense_retriever: Optional DenseRetriever instance (creates new if not provided)
            normalize_scores: Whether to normalize scores before merging (default: True)
        """
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.normalize_scores = normalize_scores
        
        logger.debug("Initialized HybridRetriever")
    
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        sparse_limit: Optional[int] = None,
        dense_limit: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using hybrid search (BM25 + vector similarity).
        
        Process:
        1. Parallel retrieval from both sparse and dense indexes
        2. Merge results and deduplicate by chunk_id
        3. Normalize scores (optional)
        4. Combine scores and rank
        5. Return top-N results
        
        Args:
            query: Search query text
            limit: Maximum number of final results to return (default: 10)
            sparse_limit: Maximum results from BM25 search (default: limit * 2)
            dense_limit: Maximum results from vector search (default: limit * 2)
            filter_conditions: Optional filter conditions (applied to both retrievers):
                - document_id: Filter by document UUID
                - document_type: Filter by document type (pdf, docx, etc.)
                - filename: Filter by exact filename
                - source_path: Filter by source path
                - metadata.*: Filter by metadata fields
        
        Returns:
            List of retrieved chunks, each containing:
            - chunk_id: UUID string
            - document_id: UUID string
            - score: Combined/hybrid relevance score
            - sparse_score: Original BM25 score (if available)
            - dense_score: Original vector similarity score (if available)
            - chunk_text: Chunk text content
            - filename: Document filename
            - document_type: Document type
            - source_path: Source path
            - metadata: Chunk metadata
        
        Raises:
            HybridRetrieverError: If retrieval fails
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided to hybrid retriever")
                return []
            
            # Set default limits if not provided
            if sparse_limit is None:
                sparse_limit = limit * 2  # Retrieve more from each index for better merging
            if dense_limit is None:
                dense_limit = limit * 2
            
            logger.debug(
                f"Hybrid retrieval: query='{query[:50]}...', "
                f"limit={limit}, sparse_limit={sparse_limit}, dense_limit={dense_limit}, "
                f"filters={filter_conditions}"
            )
            
            # Step 1: Parallel retrieval from both indexes
            logger.debug("Performing parallel retrieval from sparse and dense indexes...")
            
            sparse_results = []
            dense_results = []
            
            try:
                sparse_results = self.sparse_retriever.retrieve(
                    query=query,
                    limit=sparse_limit,
                    filter_conditions=filter_conditions,
                )
                logger.debug(f"Retrieved {len(sparse_results)} results from BM25 search")
            except SparseRetrieverError as e:
                logger.warning(f"BM25 retrieval failed: {str(e)}, continuing with vector search only")
            
            try:
                dense_results = self.dense_retriever.retrieve(
                    query=query,
                    limit=dense_limit,
                    filter_conditions=filter_conditions,
                )
                logger.debug(f"Retrieved {len(dense_results)} results from vector search")
            except DenseRetrieverError as e:
                logger.warning(f"Vector retrieval failed: {str(e)}, continuing with BM25 search only")
            
            if not sparse_results and not dense_results:
                logger.warning("No results from either sparse or dense retrieval")
                return []
            
            # Step 2: Merge and deduplicate results
            logger.debug("Merging and deduplicating results...")
            merged_results = self._merge_and_deduplicate(
                sparse_results=sparse_results,
                dense_results=dense_results,
            )
            
            # Step 3: Normalize scores if enabled
            if self.normalize_scores:
                merged_results = self._normalize_scores(merged_results)
            
            # Step 4: Combine scores and rank
            merged_results = self._combine_scores(merged_results)
            
            # Step 5: Sort by combined score and return top-N
            merged_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            final_results = merged_results[:limit]
            
            logger.info(
                f"Hybrid retrieval complete: {len(final_results)} results "
                f"(from {len(sparse_results)} sparse + {len(dense_results)} dense)"
            )
            
            return final_results
        
        except Exception as e:
            logger.error(f"Unexpected error during hybrid retrieval: {str(e)}", exc_info=True)
            raise HybridRetrieverError(
                f"Failed to perform hybrid retrieval: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
    
    def _merge_and_deduplicate(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge results from sparse and dense retrievers, deduplicating by chunk_id.
        
        Args:
            sparse_results: Results from BM25 search
            dense_results: Results from vector search
        
        Returns:
            List of merged results (one per unique chunk_id)
        """
        # Use dictionary to deduplicate by chunk_id
        merged = {}
        
        # Process sparse results
        for result in sparse_results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result.get("document_id", ""),
                    "chunk_text": result.get("chunk_text", ""),
                    "filename": result.get("filename", ""),
                    "document_type": result.get("document_type", ""),
                    "source_path": result.get("source_path", ""),
                    "metadata": result.get("metadata", {}),
                    "sparse_score": result.get("score", 0.0),
                    "dense_score": None,  # Will be filled if found in dense results
                }
            else:
                # Update sparse score if this result has a higher score
                existing_score = merged[chunk_id].get("sparse_score", 0.0)
                new_score = result.get("score", 0.0)
                if new_score > existing_score:
                    merged[chunk_id]["sparse_score"] = new_score
        
        # Process dense results
        for result in dense_results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in merged:
                # New chunk, add it
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result.get("document_id", ""),
                    "chunk_text": result.get("chunk_text", ""),
                    "filename": result.get("filename", ""),
                    "document_type": result.get("document_type", ""),
                    "source_path": result.get("source_path", ""),
                    "metadata": result.get("metadata", {}),
                    "sparse_score": None,  # Not found in sparse results
                    "dense_score": result.get("score", 0.0),
                }
            else:
                # Existing chunk, update dense score
                existing_score = merged[chunk_id].get("dense_score")
                new_score = result.get("score", 0.0)
                if existing_score is None or new_score > existing_score:
                    merged[chunk_id]["dense_score"] = new_score
        
        return list(merged.values())
    
    def _normalize_scores(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize scores to [0, 1] range for fair combination.
        
        Uses min-max normalization for both sparse and dense scores separately.
        
        Args:
            results: List of results with sparse_score and/or dense_score
        
        Returns:
            List of results with normalized scores
        """
        if not results:
            return results
        
        # Extract scores
        sparse_scores = [
            r.get("sparse_score") for r in results
            if r.get("sparse_score") is not None
        ]
        dense_scores = [
            r.get("dense_score") for r in results
            if r.get("dense_score") is not None
        ]
        
        # Calculate min/max for normalization
        sparse_min = min(sparse_scores) if sparse_scores else 0.0
        sparse_max = max(sparse_scores) if sparse_scores else 1.0
        sparse_range = sparse_max - sparse_min if sparse_max > sparse_min else 1.0
        
        dense_min = min(dense_scores) if dense_scores else 0.0
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
        
        # Normalize scores
        for result in results:
            sparse_score = result.get("sparse_score")
            if sparse_score is not None:
                result["sparse_score_normalized"] = (sparse_score - sparse_min) / sparse_range
            else:
                result["sparse_score_normalized"] = 0.0
            
            dense_score = result.get("dense_score")
            if dense_score is not None:
                result["dense_score_normalized"] = (dense_score - dense_min) / dense_range
            else:
                result["dense_score_normalized"] = 0.0
        
        return results
    
    def _combine_scores(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine sparse and dense scores into a single hybrid score.
        
        Uses weighted average:
        - If both scores available: (sparse_score * 0.4) + (dense_score * 0.6)
        - If only one available: use that score
        
        Args:
            results: List of results with sparse_score and/or dense_score (normalized if enabled)
        
        Returns:
            List of results with combined "score" field
        """
        for result in results:
            sparse_score = result.get("sparse_score_normalized") if self.normalize_scores else result.get("sparse_score")
            dense_score = result.get("dense_score_normalized") if self.normalize_scores else result.get("dense_score")
            
            if sparse_score is not None and dense_score is not None:
                # Both scores available: weighted average (40% sparse, 60% dense)
                # Dense scores typically more reliable for semantic similarity
                combined_score = (sparse_score * 0.4) + (dense_score * 0.6)
            elif sparse_score is not None:
                # Only sparse score available
                combined_score = sparse_score
            elif dense_score is not None:
                # Only dense score available
                combined_score = dense_score
            else:
                # No scores available (shouldn't happen, but handle gracefully)
                combined_score = 0.0
            
            result["score"] = combined_score
        
        return results
    
    def retrieve_by_document(
        self,
        query: str,
        document_id: UUID,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific document using hybrid search.
        
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
        Retrieve chunks from documents of a specific type using hybrid search.
        
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
