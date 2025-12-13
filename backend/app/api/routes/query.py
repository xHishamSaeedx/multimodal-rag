"""
Query endpoint.

POST /api/v1/query - Submit query and get answer
"""
import time
from fastapi import APIRouter, HTTPException, status, Request

from app.api.schemas import QueryRequest, QueryResponse, SourceInfo, ErrorResponse
from app.services.retrieval import HybridRetriever, HybridRetrieverError
from app.services.generation import AnswerGenerator, AnswerGeneratorError
from app.utils.logging import get_logger
from app.utils.metrics import (
    queries_processed_total,
    query_processing_duration_seconds,
    retrieval_duration_seconds,
    chunks_retrieved_total,
    chunks_retrieved_per_query,
    answer_generation_duration_seconds,
    answer_generation_ttft_seconds,
    answers_generated_total,
    tokens_used_total,
    retrieval_relevance_score,
    average_retrieval_relevance_per_query,
    top_k_retrieval_relevance,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def query(
    request: QueryRequest,
    http_request: Request,
) -> QueryResponse:
    """
    Submit a query and get an answer from the RAG system.
    
    This endpoint:
    1. Retrieves relevant chunks using hybrid search (BM25 + vector)
    2. Generates an answer using Groq LLM
    3. Extracts and formats citations
    4. Returns the answer with sources
    
    Args:
        request: QueryRequest containing:
            - query: The question/query text
            - limit: Number of chunks to retrieve (default: 10)
            - include_sources: Whether to include source citations (default: True)
            - filter_conditions: Optional filters (document_id, document_type, etc.)
    
    Returns:
        QueryResponse containing:
            - answer: Generated answer text
            - sources: List of source citations
            - chunks_used: List of chunk IDs used
            - model: Model used for generation
            - tokens_used: Token usage information
            - retrieval_stats: Retrieval statistics
    
    Raises:
        HTTPException: If query is empty or processing fails
    """
    query_text = request.query.strip()
    
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Query cannot be empty",
                "details": {},
            },
        )
    
    # Ensure at least one retriever is enabled
    enable_sparse = request.enable_sparse if request.enable_sparse is not None else True
    enable_dense = request.enable_dense if request.enable_dense is not None else True
    enable_table = request.enable_table if request.enable_table is not None else True
    enable_image = request.enable_image if request.enable_image is not None else True
    enable_graph = request.enable_graph if request.enable_graph is not None else True
    
    if not (enable_sparse or enable_dense or enable_table or enable_image or enable_graph):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "At least one retriever must be enabled",
                "details": {},
            },
        )
    
    query_start_time = time.time()
    try:
        logger.info(
            "query_processing_start",
            query_preview=query_text[:100] if len(query_text) > 100 else query_text,
            query_length=len(query_text),
            limit=request.limit,
            include_sources=request.include_sources,
        )
        
        # Step 1: Retrieve relevant chunks using hybrid search
        # Use pre-initialized HybridRetriever from app state
        hybrid_retriever = getattr(http_request.app.state, "hybrid_retriever", None)
        if hybrid_retriever is None:
            # Fallback: create new instance if not pre-initialized
            logger.warning(
                "retriever_not_preinitialized",
                message="HybridRetriever not pre-initialized, creating new instance (slower)",
            )
            hybrid_retriever = HybridRetriever()
        
        logger.debug(
            "retrieval_start",
            limit=request.limit,
            filter_conditions=request.filter_conditions,
            enable_sparse=request.enable_sparse,
            enable_dense=request.enable_dense,
            enable_table=request.enable_table,
            enable_image=request.enable_image,
        )
        retrieval_start_time = time.time()
        retrieved_chunks = await hybrid_retriever.retrieve(
            query=query_text,
            limit=request.limit,
            filter_conditions=request.filter_conditions,
            enable_sparse=enable_sparse,
            enable_dense=enable_dense,
            enable_table=enable_table,
            enable_image=enable_image,
            enable_graph=enable_graph,
        )
        retrieval_end_time = time.time()
        retrieval_duration = retrieval_end_time - retrieval_start_time
        
        # Record retrieval metrics
        retrieval_duration_seconds.labels(retrieval_type="hybrid").observe(retrieval_duration)
        chunks_retrieved_total.labels(retrieval_type="hybrid").inc(len(retrieved_chunks))
        chunks_retrieved_per_query.labels(retrieval_type="hybrid").observe(len(retrieved_chunks))

        # Record relevance metrics per retrieval type
        if retrieved_chunks:
            # Extract relevance scores from retrieved chunks for different retrieval types
            retrieval_type_scores = {
                'dense': [],
                'sparse': [],
                'table': [],
                'image': [],
                'graph': [],
                'hybrid': []  # Overall combined score
            }

            for chunk in retrieved_chunks:
                # Get individual retrieval scores if available
                dense_score = chunk.get('dense_score')
                sparse_score = chunk.get('sparse_score')
                table_score = chunk.get('table_score')
                image_score = chunk.get('image_score')
                graph_score = chunk.get('graph_score')
                hybrid_score = chunk.get('score', 0.0)  # Combined score

                # Normalize and collect scores for each retrieval type
                for retrieval_type, score in [
                    ('dense', dense_score),
                    ('sparse', sparse_score),
                    ('table', table_score),
                    ('image', image_score),
                    ('graph', graph_score),
                    ('hybrid', hybrid_score)
                ]:
                    if score is not None:
                        # Ensure score is between 0 and 1
                        normalized_score = min(max(float(score), 0.0), 1.0)
                        retrieval_type_scores[retrieval_type].append(normalized_score)
                        # Record individual chunk relevance score per type
                        retrieval_relevance_score.labels(retrieval_type=retrieval_type).observe(normalized_score)

            # Record average relevance per query for each retrieval type
            for retrieval_type, scores in retrieval_type_scores.items():
                if scores:
                    avg_relevance = sum(scores) / len(scores)
                    average_retrieval_relevance_per_query.labels(retrieval_type=retrieval_type).observe(avg_relevance)

                    # Record top-k relevance scores (for the top 5 chunks)
                    top_k = min(5, len(scores))
                    if top_k > 0:
                        top_scores = sorted(scores, reverse=True)[:top_k]
                        for i, score in enumerate(top_scores, 1):
                            top_k_retrieval_relevance.labels(retrieval_type=retrieval_type, k=str(i)).observe(score)
        
        # Note: retrieval_duration includes embedding time, but logs will show them separately
        logger.info(
            "retrieval_completed",
            chunks_retrieved=len(retrieved_chunks),
            duration_seconds=round(retrieval_duration, 3),
        )
        
        if not retrieved_chunks:
            logger.warning("retrieval_empty", message="No chunks retrieved for query")
            # Record metrics for empty retrieval
            query_processing_duration_seconds.observe(time.time() - query_start_time)
            queries_processed_total.labels(status="success").inc()
            return QueryResponse(
                success=True,
                query=query_text,
                answer="I don't have enough information to answer this question. No relevant documents were found in the knowledge base.",
                sources=[],
                chunks_used=[],
                model="none",
                tokens_used=None,
                retrieval_stats={
                    "chunks_found": 0,
                    "retrieval_method": "hybrid",
                },
            )
        
        # Step 2: Generate answer using Groq LLM
        logger.debug("answer_generation_start", chunks_count=len(retrieved_chunks))
        # Use pre-initialized AnswerGenerator from app state
        answer_generator = getattr(http_request.app.state, "answer_generator", None)
        if answer_generator is None:
            # Fallback: create new instance if not pre-initialized
            logger.warning(
                "answer_generator_not_preinitialized",
                message="AnswerGenerator not pre-initialized, creating new instance (slower)",
            )
            answer_generator = AnswerGenerator()
        
        llm_start_time = time.time()
        generation_result = answer_generator.generate_answer(
            query=query_text,
            chunks=retrieved_chunks,
            include_sources=request.include_sources,
        )
        llm_end_time = time.time()
        llm_duration = llm_end_time - llm_start_time
        
        # Record generation metrics
        model_name = generation_result.get('model', 'unknown')
        answer_generation_duration_seconds.labels(model=model_name).observe(llm_duration)
        answers_generated_total.labels(model=model_name, status="success").inc()
        
        ttft = generation_result.get('ttft', None)
        if ttft is not None:
            answer_generation_ttft_seconds.labels(model=model_name).observe(ttft)
        
        # Record token usage if available
        tokens_info = generation_result.get('tokens_used')
        if tokens_info:
            if isinstance(tokens_info, dict):
                input_tokens = tokens_info.get('input_tokens', 0)
                output_tokens = tokens_info.get('output_tokens', 0)
                total_tokens = tokens_info.get('total_tokens', input_tokens + output_tokens)
                
                if input_tokens > 0:
                    tokens_used_total.labels(model=model_name, type="input").inc(input_tokens)
                if output_tokens > 0:
                    tokens_used_total.labels(model=model_name, type="output").inc(output_tokens)
                if total_tokens > 0:
                    tokens_used_total.labels(model=model_name, type="total").inc(total_tokens)
        
        logger.info(
            "answer_generation_completed",
            answer_length=len(generation_result['answer']),
            sources_count=len(generation_result.get('sources', [])),
            llm_duration_seconds=round(llm_duration, 3),
            ttft_seconds=round(ttft, 3) if ttft is not None else None,
            model=model_name,
        )
        
        # Step 3: Format sources
        sources = []
        if request.include_sources and generation_result.get("sources"):
            for source in generation_result["sources"]:
                sources.append(SourceInfo(
                    chunk_id=source["chunk_id"],
                    document_id=source["document_id"],
                    filename=source["filename"],
                    chunk_index=source["chunk_index"],
                    chunk_text=source["chunk_text"],
                    full_chunk_text=source["full_chunk_text"],
                    citation=source["citation"],
                    metadata=source.get("metadata"),
                    image_path=source.get("image_path"),
                    image_url=source.get("image_url"),
                ))
        
        # Step 4: Build response
        # Record total query processing duration
        total_duration = time.time() - query_start_time
        query_processing_duration_seconds.observe(total_duration)
        queries_processed_total.labels(status="success").inc()
        
        return QueryResponse(
            success=True,
            query=query_text,
            answer=generation_result["answer"],
            sources=sources,
            chunks_used=generation_result.get("chunks_used", []),
            model=generation_result.get("model", "unknown"),
            tokens_used=generation_result.get("tokens_used"),
            retrieval_stats={
                "chunks_found": len(retrieved_chunks),
                "chunks_used": len(generation_result.get("chunks_used", [])),
                "retrieval_method": "hybrid",
            },
        )
    
    except HybridRetrieverError as e:
        # Record error metrics
        query_processing_duration_seconds.observe(time.time() - query_start_time)
        queries_processed_total.labels(status="retrieval_error").inc()
        
        logger.error(
            "query_retrieval_error",
            error_type="HybridRetrieverError",
            error_message=str(e),
            query_preview=query_text[:100] if query_text else "",
            details=e.details if hasattr(e, 'details') else {},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Retrieval failed: {str(e)}",
                "details": e.details if hasattr(e, 'details') else {"query": query_text[:100]},
            },
        )
    
    except AnswerGeneratorError as e:
        # Record error metrics
        query_processing_duration_seconds.observe(time.time() - query_start_time)
        queries_processed_total.labels(status="generation_error").inc()
        
        logger.error(
            "query_generation_error",
            error_type="AnswerGeneratorError",
            error_message=str(e),
            query_preview=query_text[:100] if query_text else "",
            details=e.details if hasattr(e, 'details') else {},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Answer generation failed: {str(e)}",
                "details": e.details if hasattr(e, 'details') else {"query": query_text[:100]},
            },
        )
    
    except Exception as e:
        # Record error metrics
        query_processing_duration_seconds.observe(time.time() - query_start_time)
        queries_processed_total.labels(status="error").inc()
        
        logger.error(
            "query_unexpected_error",
            error_type=type(e).__name__,
            error_message=str(e),
            query_preview=query_text[:100] if query_text else "",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Unexpected error during query processing: {str(e)}",
                "details": {
                    "query": query_text[:100] if query_text else "",
                    "error_type": type(e).__name__,
                },
            },
        )
