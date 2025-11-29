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

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

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

# Global Elasticsearch client instance
_elasticsearch_client: Optional[Elasticsearch] = None


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
        # Use gRPC for better performance (lower latency than HTTP)
        # gRPC is preferred for vector operations, HTTP is fallback for compatibility
        _qdrant_client = QdrantClientLib(
            host=settings.qdrant_host,
            port=settings.qdrant_port,  # HTTP port (fallback)
            grpc_port=settings.qdrant_grpc_port,  # gRPC port (preferred)
            timeout=settings.qdrant_timeout,
            prefer_grpc=True,  # Prefer gRPC for better performance
        )
        logger.info(
            f"Initialized Qdrant client (gRPC preferred): {settings.qdrant_host}:{settings.qdrant_grpc_port} (gRPC), {settings.qdrant_port} (HTTP fallback)"
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


def get_elasticsearch_client() -> Elasticsearch:
    """
    Get or create Elasticsearch client instance.
    
    Returns:
        Elasticsearch client instance
    
    Raises:
        DatabaseError: If Elasticsearch client creation fails
    """
    global _elasticsearch_client
    
    if _elasticsearch_client is not None:
        return _elasticsearch_client
    
    if Elasticsearch is None:
        raise DatabaseError(
            "Elasticsearch client is not installed. Install it with: pip install 'elasticsearch>=8.0.0,<9.0.0'",
            {},
        )
    
    try:
        # Parse URL to extract host and port
        from urllib.parse import urlparse
        parsed = urlparse(settings.elasticsearch_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 9200
        
        # Create Elasticsearch client
        _elasticsearch_client = Elasticsearch(
            hosts=[{"host": host, "port": port, "scheme": "http"}],
            request_timeout=settings.elasticsearch_timeout,
            max_retries=3,
            retry_on_timeout=True,
        )
        
        # Test connection
        if not _elasticsearch_client.ping():
            raise DatabaseError(
                "Failed to connect to Elasticsearch: ping() returned False",
                {"url": settings.elasticsearch_url},
            )
        
        logger.info(
            f"Initialized Elasticsearch client: {settings.elasticsearch_url}"
        )
        return _elasticsearch_client
    except Exception as e:
        logger.error(f"Failed to create Elasticsearch client: {str(e)}")
        raise DatabaseError(
            f"Failed to create Elasticsearch client: {str(e)}",
            {
                "url": settings.elasticsearch_url,
                "error": str(e),
            },
        ) from e


def reset_elasticsearch_client() -> None:
    """Reset the global Elasticsearch client (useful for testing)."""
    global _elasticsearch_client
    _elasticsearch_client = None
