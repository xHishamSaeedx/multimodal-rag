"""
Database connections and clients.

This module initializes connections to:
- Supabase (PostgreSQL)
- Qdrant (Vector DB)
- Elasticsearch (BM25 Index)
"""

import logging
from typing import Optional

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

try:
    from qdrant_client import QdrantClient as QdrantClientLib
except ImportError:
    QdrantClientLib = None

from app.core.config import settings
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class DatabaseError(BaseAppException):
    """Raised when database operations fail."""
    pass


# Global Supabase client instance
_supabase_client: Optional[Client] = None

# Global Qdrant client instance
_qdrant_client: Optional[QdrantClientLib] = None


def get_supabase_client() -> Client:
    """
    Get or create Supabase client instance.
    
    Returns:
        Supabase client instance
    
    Raises:
        DatabaseError: If Supabase is not configured or client creation fails
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
    
    if create_client is None:
        raise DatabaseError(
            "Supabase client is not installed. Install it with: pip install supabase",
            {},
        )
    
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise DatabaseError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env",
            {},
        )
    
    try:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,  # Use service role for server-side operations
        )
        logger.info(f"Initialized Supabase client for: {settings.supabase_url}")
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {str(e)}")
        raise DatabaseError(
            f"Failed to create Supabase client: {str(e)}",
            {"supabase_url": settings.supabase_url},
        ) from e


def reset_supabase_client() -> None:
    """Reset the global Supabase client (useful for testing)."""
    global _supabase_client
    _supabase_client = None


def get_qdrant_client() -> QdrantClientLib:
    """
    Get or create Qdrant client instance.
    
    Returns:
        Qdrant client instance
    
    Raises:
        DatabaseError: If Qdrant client creation fails
    """
    global _qdrant_client
    
    if _qdrant_client is not None:
        return _qdrant_client
    
    if QdrantClientLib is None:
        raise DatabaseError(
            "Qdrant client is not installed. Install it with: pip install qdrant-client",
            {},
        )
    
    try:
        _qdrant_client = QdrantClientLib(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            timeout=settings.qdrant_timeout,
        )
        logger.info(
            f"Initialized Qdrant client: {settings.qdrant_host}:{settings.qdrant_port}"
        )
        return _qdrant_client
    except Exception as e:
        logger.error(f"Failed to create Qdrant client: {str(e)}")
        raise DatabaseError(
            f"Failed to create Qdrant client: {str(e)}",
            {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "error": str(e),
            },
        ) from e


def reset_qdrant_client() -> None:
    """Reset the global Qdrant client (useful for testing)."""
    global _qdrant_client
    _qdrant_client = None
