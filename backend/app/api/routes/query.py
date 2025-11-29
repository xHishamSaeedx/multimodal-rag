"""
Query endpoint.

POST /api/v1/query - Submit query and get answer
"""
import time
import traceback
from fastapi import APIRouter, HTTPException, status, Request

from app.api.schemas import QueryRequest, QueryResponse, SourceInfo, ErrorResponse
from app.services.retrieval import HybridRetriever, HybridRetrieverError
from app.services.generation import AnswerGenerator, AnswerGeneratorError
from app.utils.exceptions import BaseAppException
from app.utils.logging import get_logger

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
        )
        retrieval_start_time = time.time()
        retrieved_chunks = await hybrid_retriever.retrieve(
            query=query_text,
            limit=request.limit,
            filter_conditions=request.filter_conditions,
        )
        retrieval_end_time = time.time()
        retrieval_duration = retrieval_end_time - retrieval_start_time
        
        # Note: retrieval_duration includes embedding time, but logs will show them separately
        logger.info(
            "retrieval_completed",
            chunks_retrieved=len(retrieved_chunks),
            duration_seconds=round(retrieval_duration, 3),
        )
        
        if not retrieved_chunks:
            logger.warning("retrieval_empty", message="No chunks retrieved for query")
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
        
        ttft = generation_result.get('ttft', None)
        logger.info(
            "answer_generation_completed",
            answer_length=len(generation_result['answer']),
            sources_count=len(generation_result.get('sources', [])),
            llm_duration_seconds=round(llm_duration, 3),
            ttft_seconds=round(ttft, 3) if ttft is not None else None,
            model=generation_result.get('model', 'unknown'),
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
