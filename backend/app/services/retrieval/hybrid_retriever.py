"""
Hybrid retrieval service.

Combines BM25 (sparse) and vector (dense) retrieval results.
Handles merging, deduplication, and score normalization.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.services.retrieval.sparse_retriever import SparseRetriever, SparseRetrieverError
from app.services.retrieval.dense_retriever import DenseRetriever, DenseRetrieverError
from app.services.retrieval.table_retriever import TableRetriever, TableRetrieverError
from app.services.retrieval.image_retriever import ImageRetriever, ImageRetrieverError
from app.utils.exceptions import BaseAppException
from app.utils.logging import get_logger
from app.utils.metrics import (
    retrieval_duration_seconds,
    chunks_retrieved_total,
    chunks_retrieved_per_query,
    text_embedding_duration_seconds,
    hybrid_merge_duration_seconds,
)

logger = get_logger(__name__)


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
        table_retriever: Optional[TableRetriever] = None,
        image_retriever: Optional[ImageRetriever] = None,
        normalize_scores: bool = True,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            sparse_retriever: Optional SparseRetriever instance (creates new if not provided)
            dense_retriever: Optional DenseRetriever instance (creates new if not provided)
            table_retriever: Optional TableRetriever instance (creates new if not provided)
            image_retriever: Optional ImageRetriever instance (creates new if not provided)
            normalize_scores: Whether to normalize scores before merging (default: True)
        """
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.table_retriever = table_retriever or TableRetriever()
        self.image_retriever = image_retriever or ImageRetriever()
        self.normalize_scores = normalize_scores
        
        logger.debug("hybrid_retriever_initialized", includes_table_retrieval=True, includes_image_retrieval=True)
    
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        sparse_limit: Optional[int] = None,
        dense_limit: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        enable_sparse: bool = True,
        enable_dense: bool = True,
        enable_table: bool = True,
        enable_image: bool = True,
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
                logger.warning("hybrid_retrieval_empty_query")
                return []
            
            # Set default limits if not provided
            if sparse_limit is None:
                sparse_limit = limit * 2  # Retrieve more from each index for better merging
            if dense_limit is None:
                dense_limit = limit * 2
            
            total_start = time.time()
            
            logger.debug(
                "hybrid_retrieval_start",
                query_preview=query[:50] if len(query) > 50 else query,
                limit=limit,
                sparse_limit=sparse_limit,
                dense_limit=dense_limit,
                has_filters=filter_conditions is not None,
            )
            
            # Step 0: Generate query embedding (preprocessing, not part of retrieval timing)
            embedding_start = time.time()
            # Run embedding generation in thread pool since it might block
            query_embedding = await asyncio.to_thread(
                self.dense_retriever.generate_query_embedding,
                query
            )
            embedding_time = time.time() - embedding_start
            # Record embedding time metric
            text_embedding_duration_seconds.observe(embedding_time)
            logger.debug(
                "query_embedding_generated",
                duration_seconds=round(embedding_time, 3),
                embedding_dim=len(query_embedding),
            )
            
            # Step 1: Parallel retrieval from all indexes (retrieval timing starts here)
            retrieval_start = time.time()
            enabled_retrievers = []
            if enable_sparse:
                enabled_retrievers.append("bm25")
            if enable_dense:
                enabled_retrievers.append("text_chunks")
            if enable_table:
                enabled_retrievers.append("table_chunks")
            if enable_image:
                enabled_retrievers.append("image_chunks")
            
            logger.info("hybrid_retrieval_parallel_start", method="async_await", collections=enabled_retrievers)
            
            # Define async retrieval functions for parallel execution
            async def retrieve_sparse():
                if not enable_sparse:
                    return [], 0.0, None
                start = time.time()
                try:
                    logger.debug("bm25_search_start", method="async_parallel")
                    results = await self.sparse_retriever.retrieve(
                        query=query,
                        limit=sparse_limit,
                        filter_conditions=filter_conditions,
                    )
                    elapsed = time.time() - start
                    # Count chunk types
                    text_count = sum(1 for r in results if r.get("chunk_type", "text") == "text")
                    table_count = sum(1 for r in results if r.get("chunk_type") == "table")
                    image_count = sum(1 for r in results if r.get("chunk_type") == "image")
                    # Record sparse retrieval metrics
                    retrieval_duration_seconds.labels(retrieval_type="sparse").observe(elapsed)
                    chunks_retrieved_total.labels(retrieval_type="sparse").inc(len(results))
                    if len(results) > 0:
                        chunks_retrieved_per_query.labels(retrieval_type="sparse").observe(len(results))
                    logger.info(
                        "bm25_search_completed",
                        duration_seconds=round(elapsed, 3),
                        results_count=len(results),
                        text_chunks=text_count,
                        table_chunks=table_count,
                        image_chunks=image_count,
                    )
                    return results, elapsed, None
                except SparseRetrieverError as e:
                    elapsed = time.time() - start
                    logger.warning(
                        "bm25_search_failed",
                        duration_seconds=round(elapsed, 3),
                        error_message=str(e),
                    )
                    return [], elapsed, e
            
            async def retrieve_dense():
                if not enable_dense:
                    return [], 0.0, None
                start = time.time()
                try:
                    logger.debug("vector_search_start", collection="text_chunks", method="async_parallel")
                    # Use pre-computed embedding (doesn't include embedding time)
                    results = await self.dense_retriever.retrieve_with_embedding(
                        query_embedding=query_embedding,
                        limit=dense_limit,
                        filter_conditions=filter_conditions,
                    )
                    elapsed = time.time() - start
                    # Count chunk types
                    text_count = sum(1 for r in results if r.get("chunk_type", "text") == "text")
                    table_count = sum(1 for r in results if r.get("chunk_type") == "table")
                    image_count = sum(1 for r in results if r.get("chunk_type") == "image")
                    # Record dense retrieval metrics
                    retrieval_duration_seconds.labels(retrieval_type="dense").observe(elapsed)
                    chunks_retrieved_total.labels(retrieval_type="dense").inc(len(results))
                    if len(results) > 0:
                        chunks_retrieved_per_query.labels(retrieval_type="dense").observe(len(results))
                    logger.info(
                        "vector_search_completed",
                        collection="text_chunks",
                        duration_seconds=round(elapsed, 3),
                        results_count=len(results),
                        text_chunks=text_count,
                        table_chunks=table_count,
                        image_chunks=image_count,
                    )
                    return results, elapsed, None
                except DenseRetrieverError as e:
                    elapsed = time.time() - start
                    logger.warning(
                        "vector_search_failed",
                        collection="text_chunks",
                        duration_seconds=round(elapsed, 3),
                        error_message=str(e),
                    )
                    return [], elapsed, e
            
            async def retrieve_tables():
                if not enable_table:
                    return [], 0.0, None
                start = time.time()
                try:
                    logger.debug("table_search_start", collection="table_chunks", method="async_parallel")
                    # Use pre-computed embedding (same as text chunks)
                    results = await self.table_retriever.retrieve_with_embedding(
                        query_embedding=query_embedding,
                        limit=dense_limit,  # Use same limit as dense text search
                        filter_conditions=filter_conditions,
                    )
                    elapsed = time.time() - start
                    # Record table retrieval metrics
                    retrieval_duration_seconds.labels(retrieval_type="table").observe(elapsed)
                    chunks_retrieved_total.labels(retrieval_type="table").inc(len(results))
                    if len(results) > 0:
                        chunks_retrieved_per_query.labels(retrieval_type="table").observe(len(results))
                    logger.info(
                        "table_search_completed",
                        collection="table_chunks",
                        duration_seconds=round(elapsed, 3),
                        results_count=len(results),
                    )
                    return results, elapsed, None
                except TableRetrieverError as e:
                    elapsed = time.time() - start
                    logger.warning(
                        "table_search_failed",
                        collection="table_chunks",
                        duration_seconds=round(elapsed, 3),
                        error_message=str(e),
                    )
                    return [], elapsed, e
            
            async def retrieve_images():
                if not enable_image:
                    return [], 0.0, None
                start = time.time()
                try:
                    logger.debug("image_search_start", collection="image_chunks", method="async_parallel")
                    # Generate image query embedding (1024 dimensions, different from text embedding)
                    image_query_embedding = await asyncio.to_thread(
                        self.image_retriever.generate_query_embedding,
                        query
                    )
                    # Use image-specific embedding (1024 dim)
                    results = await self.image_retriever.retrieve_with_embedding(
                        query_embedding=image_query_embedding,
                        limit=dense_limit,  # Use same limit as dense text search
                        filter_conditions=filter_conditions,
                    )
                    elapsed = time.time() - start
                    # Record image retrieval metrics
                    retrieval_duration_seconds.labels(retrieval_type="image").observe(elapsed)
                    chunks_retrieved_total.labels(retrieval_type="image").inc(len(results))
                    if len(results) > 0:
                        chunks_retrieved_per_query.labels(retrieval_type="image").observe(len(results))
                    logger.info(
                        "image_search_completed",
                        collection="image_chunks",
                        duration_seconds=round(elapsed, 3),
                        results_count=len(results),
                    )
                    return results, elapsed, None
                except ImageRetrieverError as e:
                    elapsed = time.time() - start
                    logger.warning(
                        "image_search_failed",
                        collection="image_chunks",
                        duration_seconds=round(elapsed, 3),
                        error_message=str(e),
                    )
                    return [], elapsed, e
            
            # Execute all retrievers in parallel using asyncio.gather()
            logger.debug("parallel_search_execution", method="asyncio_gather", collections=enabled_retrievers)
            (sparse_results, sparse_time, sparse_error), (dense_results, dense_time, dense_error), (table_results, table_time, table_error), (image_results, image_time, image_error) = await asyncio.gather(
                retrieve_sparse(),
                retrieve_dense(),
                retrieve_tables(),
                retrieve_images(),
            )
            
            if sparse_error:
                logger.warning("hybrid_retrieval_partial_failure", failed="bm25", fallback="vector_and_table_and_image")
            if dense_error:
                logger.warning("hybrid_retrieval_partial_failure", failed="text_chunks", fallback="table_and_image_and_bm25")
            if table_error:
                logger.warning("hybrid_retrieval_partial_failure", failed="table_chunks", fallback="text_and_image_and_bm25")
            if image_error:
                logger.warning("hybrid_retrieval_partial_failure", failed="image_chunks", fallback="text_and_table_and_bm25")
            
            retrieval_elapsed = time.time() - retrieval_start
            sequential_time = sparse_time + dense_time + table_time + image_time
            parallel_savings = max(0, sequential_time - retrieval_elapsed)
            efficiency = (parallel_savings / sequential_time * 100) if sequential_time > 0 else 0
            
            # Count chunk types from all sources
            all_results = sparse_results + dense_results + table_results + image_results
            text_count = sum(1 for r in all_results if r.get("chunk_type", "text") == "text")
            table_count = sum(1 for r in all_results if r.get("chunk_type") == "table")
            image_count = sum(1 for r in all_results if r.get("chunk_type") == "image")
            
            logger.info(
                "parallel_retrieval_completed",
                duration_seconds=round(retrieval_elapsed, 3),
                bm25_duration_seconds=round(sparse_time, 3),
                text_vector_duration_seconds=round(dense_time, 3),
                table_vector_duration_seconds=round(table_time, 3),
                image_vector_duration_seconds=round(image_time, 3),
                sequential_duration_seconds=round(sequential_time, 3),
                time_saved_seconds=round(parallel_savings, 3),
                efficiency_percent=round(efficiency, 1),
                total_chunks_found=len(all_results),
                text_chunks_found=text_count,
                table_chunks_found=table_count,
                image_chunks_found=image_count,
            )
            
            if not sparse_results and not dense_results and not table_results and not image_results:
                logger.warning("hybrid_retrieval_no_results")
                return []
            
            # Step 2: Merge and deduplicate results from all sources
            logger.debug("hybrid_retrieval_merge_start", sources=["bm25", "text_chunks", "table_chunks", "image_chunks"])
            merged_results = self._merge_and_deduplicate(
                sparse_results=sparse_results,
                dense_results=dense_results,
                table_results=table_results,
                image_results=image_results,
            )
            
            # Step 3: Normalize scores if enabled
            if self.normalize_scores:
                merged_results = self._normalize_scores(merged_results)
            
            # Step 4: Combine scores and rank
            merged_results = self._combine_scores(merged_results)
            
            # Step 5: Sort by combined score and return top-N
            merge_start = time.time()
            merged_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            final_results = merged_results[:limit]
            merge_time = time.time() - merge_start
            # Record hybrid merge duration
            hybrid_merge_duration_seconds.observe(merge_time)
            
            total_time = time.time() - total_start
            
            # Count final chunk types
            final_text_count = sum(1 for r in final_results if r.get("chunk_type", "text") == "text")
            final_table_count = sum(1 for r in final_results if r.get("chunk_type") == "table")
            final_image_count = sum(1 for r in final_results if r.get("chunk_type") == "image")
            
            logger.info(
                "hybrid_retrieval_complete",
                results_count=len(final_results),
                text_chunks=final_text_count,
                table_chunks=final_table_count,
                image_chunks=final_image_count,
                embedding_duration_seconds=round(embedding_time, 3),
                retrieval_duration_seconds=round(retrieval_elapsed, 3),
                merge_duration_seconds=round(merge_time, 3),
                total_duration_seconds=round(total_time, 3),
                sparse_results_count=len(sparse_results),
                dense_text_results_count=len(dense_results),
                dense_table_results_count=len(table_results),
                dense_image_results_count=len(image_results),
            )
            
            return final_results
        
        except Exception as e:
            logger.error(
                "hybrid_retrieval_error",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            raise HybridRetrieverError(
                f"Failed to perform hybrid retrieval: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
    
    def _merge_and_deduplicate(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        table_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge results from sparse, dense text, table, and image retrievers, deduplicating by chunk_id.
        
        Args:
            sparse_results: Results from BM25 search
            dense_results: Results from text_chunks vector search
            table_results: Results from table_chunks vector search
            image_results: Results from image_chunks vector search
        
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
                    "table_markdown": result.get("table_markdown"),  # May be present in BM25 results
                    "filename": result.get("filename", ""),
                    "document_type": result.get("document_type", ""),
                    "source_path": result.get("source_path", ""),
                    "chunk_type": result.get("chunk_type", "text"),
                    "metadata": result.get("metadata", {}),
                    "sparse_score": result.get("score", 0.0),
                    "dense_score": None,  # Will be filled if found in dense results
                    "table_score": None,  # Will be filled if found in table results
                    "image_score": None,  # Will be filled if found in image results
                }
            else:
                # Update sparse score if this result has a higher score
                existing_score = merged[chunk_id].get("sparse_score", 0.0)
                new_score = result.get("score", 0.0)
                if new_score > existing_score:
                    merged[chunk_id]["sparse_score"] = new_score
                # Update table_markdown if available
                if result.get("table_markdown") and not merged[chunk_id].get("table_markdown"):
                    merged[chunk_id]["table_markdown"] = result.get("table_markdown")
        
        # Process dense text results
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
                    "table_markdown": result.get("table_markdown"),  # May be present for table chunks
                    "filename": result.get("filename", ""),
                    "document_type": result.get("document_type", ""),
                    "source_path": result.get("source_path", ""),
                    "chunk_type": result.get("chunk_type", "text"),
                    "metadata": result.get("metadata", {}),
                    "sparse_score": None,  # Not found in sparse results
                    "dense_score": result.get("score", 0.0),
                    "table_score": None,  # Not from table collection
                }
            else:
                # Existing chunk, update dense score
                existing_score = merged[chunk_id].get("dense_score")
                new_score = result.get("score", 0.0)
                if existing_score is None or new_score > existing_score:
                    merged[chunk_id]["dense_score"] = new_score
                # Update table_markdown if available
                if result.get("table_markdown") and not merged[chunk_id].get("table_markdown"):
                    merged[chunk_id]["table_markdown"] = result.get("table_markdown")
        
        # Process table results
        for result in table_results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in merged:
                # New table chunk, add it
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result.get("document_id", ""),
                    "chunk_text": result.get("chunk_text", ""),  # Flattened text
                    "table_markdown": result.get("table_markdown", ""),  # Markdown format
                    "table_data": result.get("table_data", {}),  # JSON format
                    "filename": result.get("filename", ""),
                    "document_type": result.get("document_type", ""),
                    "source_path": result.get("source_path", ""),
                    "chunk_type": "table",
                    "metadata": result.get("metadata", {}),
                    "sparse_score": None,  # May be found in sparse results
                    "dense_score": None,  # Not from text_chunks collection
                    "table_score": result.get("score", 0.0),
                    "image_score": None,  # Not from image collection
                }
            else:
                # Existing chunk, update table score
                existing_score = merged[chunk_id].get("table_score")
                new_score = result.get("score", 0.0)
                if existing_score is None or new_score > existing_score:
                    merged[chunk_id]["table_score"] = new_score
                # Update table-specific fields if available
                if result.get("table_markdown") and not merged[chunk_id].get("table_markdown"):
                    merged[chunk_id]["table_markdown"] = result.get("table_markdown")
                if result.get("table_data") and not merged[chunk_id].get("table_data"):
                    merged[chunk_id]["table_data"] = result.get("table_data")
                # Mark as table chunk
                merged[chunk_id]["chunk_type"] = "table"
        
        # Process image results
        for result in image_results:
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in merged:
                # New image chunk, add it
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result.get("document_id", ""),
                    "chunk_text": result.get("chunk_text", ""),  # Image description or caption
                    "image_path": result.get("image_path", ""),  # Supabase storage path
                    "caption": result.get("caption", ""),  # Image caption
                    "image_type": result.get("image_type", "photo"),  # diagram, chart, photo, etc.
                    "filename": result.get("filename", ""),
                    "document_type": result.get("document_type", ""),
                    "source_path": result.get("source_path", ""),
                    "chunk_type": "image",
                    "metadata": result.get("metadata", {}),
                    "sparse_score": None,  # May be found in sparse results
                    "dense_score": None,  # Not from text_chunks collection
                    "table_score": None,  # Not from table collection
                    "image_score": result.get("score", 0.0),
                }
            else:
                # Existing chunk, update image score
                existing_score = merged[chunk_id].get("image_score")
                new_score = result.get("score", 0.0)
                if existing_score is None or new_score > existing_score:
                    merged[chunk_id]["image_score"] = new_score
                # Update image-specific fields if available
                if result.get("image_path") and not merged[chunk_id].get("image_path"):
                    merged[chunk_id]["image_path"] = result.get("image_path")
                if result.get("caption") and not merged[chunk_id].get("caption"):
                    merged[chunk_id]["caption"] = result.get("caption")
                if result.get("image_type") and not merged[chunk_id].get("image_type"):
                    merged[chunk_id]["image_type"] = result.get("image_type")
                # Mark as image chunk
                merged[chunk_id]["chunk_type"] = "image"
        
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
        table_scores = [
            r.get("table_score") for r in results
            if r.get("table_score") is not None
        ]
        image_scores = [
            r.get("image_score") for r in results
            if r.get("image_score") is not None
        ]
        
        # Calculate min/max for normalization
        sparse_min = min(sparse_scores) if sparse_scores else 0.0
        sparse_max = max(sparse_scores) if sparse_scores else 1.0
        sparse_range = sparse_max - sparse_min if sparse_max > sparse_min else 1.0
        
        dense_min = min(dense_scores) if dense_scores else 0.0
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
        
        table_min = min(table_scores) if table_scores else 0.0
        table_max = max(table_scores) if table_scores else 1.0
        table_range = table_max - table_min if table_max > table_min else 1.0
        
        image_min = min(image_scores) if image_scores else 0.0
        image_max = max(image_scores) if image_scores else 1.0
        image_range = image_max - image_min if image_max > image_min else 1.0
        
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
            
            table_score = result.get("table_score")
            if table_score is not None:
                result["table_score_normalized"] = (table_score - table_min) / table_range
            else:
                result["table_score_normalized"] = 0.0
            
            image_score = result.get("image_score")
            if image_score is not None:
                result["image_score_normalized"] = (image_score - image_min) / image_range
            else:
                result["image_score_normalized"] = 0.0
        
        return results
    
    def _combine_scores(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine sparse, dense text, table, and image scores into a single hybrid score.
        
        Uses weighted average:
        - If all four scores available: (sparse * 0.2) + (dense_text * 0.3) + (table * 0.2) + (image * 0.3)
        - If three scores available: weighted based on available scores
        - If two scores available: weighted average
        - If only one available: use that score
        
        Args:
            results: List of results with sparse_score, dense_score, table_score, and/or image_score (normalized if enabled)
        
        Returns:
            List of results with combined "score" field
        """
        for result in results:
            sparse_score = result.get("sparse_score_normalized") if self.normalize_scores else result.get("sparse_score")
            dense_score = result.get("dense_score_normalized") if self.normalize_scores else result.get("dense_score")
            table_score = result.get("table_score_normalized") if self.normalize_scores else result.get("table_score")
            image_score = result.get("image_score_normalized") if self.normalize_scores else result.get("image_score")
            
            # Count available scores
            available_scores = [s for s in [sparse_score, dense_score, table_score, image_score] if s is not None]
            
            # Combine scores based on what's available
            if len(available_scores) == 4:
                # All four scores: weighted average
                combined_score = (sparse_score * 0.2) + (dense_score * 0.3) + (table_score * 0.2) + (image_score * 0.3)
            elif sparse_score is not None and dense_score is not None and table_score is not None:
                # Three scores (sparse + dense + table): weighted average
                combined_score = (sparse_score * 0.3) + (dense_score * 0.4) + (table_score * 0.3)
            elif sparse_score is not None and dense_score is not None and image_score is not None:
                # Three scores (sparse + dense + image): weighted average
                combined_score = (sparse_score * 0.25) + (dense_score * 0.35) + (image_score * 0.4)
            elif sparse_score is not None and table_score is not None and image_score is not None:
                # Three scores (sparse + table + image): weighted average
                combined_score = (sparse_score * 0.3) + (table_score * 0.3) + (image_score * 0.4)
            elif dense_score is not None and table_score is not None and image_score is not None:
                # Three scores (dense + table + image): weighted average
                combined_score = (dense_score * 0.35) + (table_score * 0.3) + (image_score * 0.35)
            elif sparse_score is not None and dense_score is not None:
                # Sparse + dense text: weighted average (40% sparse, 60% dense)
                combined_score = (sparse_score * 0.4) + (dense_score * 0.6)
            elif sparse_score is not None and table_score is not None:
                # Sparse + table: weighted average (40% sparse, 60% table)
                combined_score = (sparse_score * 0.4) + (table_score * 0.6)
            elif sparse_score is not None and image_score is not None:
                # Sparse + image: weighted average (40% sparse, 60% image)
                combined_score = (sparse_score * 0.4) + (image_score * 0.6)
            elif dense_score is not None and table_score is not None:
                # Dense text + table: equal weight
                combined_score = (dense_score * 0.5) + (table_score * 0.5)
            elif dense_score is not None and image_score is not None:
                # Dense text + image: equal weight
                combined_score = (dense_score * 0.5) + (image_score * 0.5)
            elif table_score is not None and image_score is not None:
                # Table + image: equal weight
                combined_score = (table_score * 0.5) + (image_score * 0.5)
            elif sparse_score is not None:
                # Only sparse score available
                combined_score = sparse_score
            elif dense_score is not None:
                # Only dense text score available
                combined_score = dense_score
            elif table_score is not None:
                # Only table score available
                combined_score = table_score
            elif image_score is not None:
                # Only image score available
                combined_score = image_score
            else:
                # No scores available (shouldn't happen, but handle gracefully)
                combined_score = 0.0
            
            result["score"] = combined_score
        
        return results
    
    async def retrieve_by_document(
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
        Retrieve chunks from documents of a specific type using hybrid search.
        
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
