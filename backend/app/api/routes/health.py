"""
Health check endpoints.

GET /api/v1/health - Health check for API and dependencies
"""
from fastapi import APIRouter
from datetime import datetime
from typing import Dict, Any

router = APIRouter(prefix="/health", tags=["health"])


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
