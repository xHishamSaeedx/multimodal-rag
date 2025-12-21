"""
Health check endpoints.

GET /api/v1/health - Health check for API and dependencies
GET /metrics - Prometheus metrics endpoint
"""
from fastapi import APIRouter, Response
from datetime import datetime
from typing import Dict, Any
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(prefix="/health", tags=["health"])
metrics_router = APIRouter(tags=["metrics"])


@router.get("", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns the current status of the API and its dependencies.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "multimodal-rag-api",
        "version": "1.0.0"
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint.
    
    Checks if the service is ready to accept traffic.
    """
    # TODO: Add actual dependency checks (database, vector store, etc.)
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint.

    Checks if the service is alive.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/models")
async def get_current_models() -> Dict[str, Any]:
    """
    Get current models configuration.

    Returns the currently configured models for different components
    (text embeddings, image embeddings, LLMs, etc.) along with basic info.
    """
    from app.core.config import settings

    # Get current model configurations
    models_config = {
        "text_embedding": {
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
            "device": settings.embedding_device,
        },
        "image_embedding": {
            "model_type": settings.image_embedding_model_type,
            "model_name": settings.image_embedding_model_name,
        },
        "llm": {
            "provider": settings.vision_llm_provider,
            "model": settings.vision_llm_model,
        },
        "captioning": {
            "model": settings.captioning_model,
        },
        "vision_processing": {
            "mode": settings.vision_processing_mode,
        },
        "retrieval_types": [
            "sparse",  # BM25/Elasticsearch
            "dense",   # Vector similarity
            "image",   # CLIP/SigLIP
            "table",   # Structured data
            "graph"    # Knowledge graph (if enabled)
        ] if settings.neo4j_enabled else [
            "sparse",
            "dense",
            "image",
            "table"
        ]
    }

    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "models": models_config
    }


@metrics_router.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/sample-pdf")
async def get_sample_pdf():
    """
    Download the sample PDF used for ingestion performance testing.
    
    Returns the tech_sector_report.pdf file for users to see what sample was used.
    """
    from fastapi.responses import FileResponse
    import os
    
    # Get the path to the PDF file
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "tech_sector_report.pdf")
    pdf_path = os.path.abspath(pdf_path)
    
    if not os.path.exists(pdf_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Sample PDF not found")
    
    return FileResponse(
        path=pdf_path,
        filename="tech_sector_report.pdf",
        media_type="application/pdf"
    )
