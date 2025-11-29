"""
FastAPI dependencies.

This module contains dependency injection functions for database clients,
configuration, and other shared resources.

Note: These functions are currently unused as routes access app.state directly.
They are kept for potential future use with FastAPI's Depends() system.
"""
from typing import Optional
from fastapi import Request

from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.embedding.text_embedder import TextEmbedder
from app.repositories.vector_repository import VectorRepository
from app.repositories.sparse_repository import SparseRepository


def get_hybrid_retriever(request: Request) -> HybridRetriever:
    """
    Get the pre-initialized HybridRetriever from app state.
    
    Falls back to creating a new instance if not pre-initialized.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HybridRetriever instance
    """
    hybrid_retriever = getattr(request.app.state, "hybrid_retriever", None)
    if hybrid_retriever is None:
        # Fallback: create new instance if not pre-initialized
        from app.services.retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever()
    return hybrid_retriever


def get_text_embedder(request: Request) -> Optional[TextEmbedder]:
    """
    Get the pre-initialized TextEmbedder from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        TextEmbedder instance or None if not available
    """
    return getattr(request.app.state, "text_embedder", None)


def get_vector_repository(request: Request) -> Optional[VectorRepository]:
    """
    Get the pre-initialized VectorRepository from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        VectorRepository instance or None if not available
    """
    return getattr(request.app.state, "vector_repository", None)


def get_sparse_repository(request: Request) -> Optional[SparseRepository]:
    """
    Get the pre-initialized SparseRepository from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        SparseRepository instance or None if not available
    """
    return getattr(request.app.state, "sparse_repository", None)

