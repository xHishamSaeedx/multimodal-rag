"""
Neo4j database connection and management.

This module provides functions to connect to Neo4j and manage graph database operations.
"""
from neo4j import GraphDatabase, Driver
from typing import Optional
import time

from app.core.config import settings
from app.core.database import DatabaseError
from app.utils.metrics import (
    neo4j_connection_errors_total,
    neo4j_connection_duration_seconds,
    neo4j_active_connections,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Global Neo4j driver instance
_neo4j_driver: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    """
    Get or create Neo4j driver instance.

    Returns:
        Neo4j driver instance

    Raises:
        DatabaseError: If Neo4j is not configured or driver creation fails
    """
    global _neo4j_driver

    if _neo4j_driver is not None:
        return _neo4j_driver

    if not settings.neo4j_enabled:
        raise DatabaseError(
            "Neo4j is disabled. Set NEO4J_ENABLED=true to enable.",
            {},
        )

    connection_start = time.time()
    try:
        _neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
            connection_timeout=settings.neo4j_timeout,
        )

        # Verify connection
        _neo4j_driver.verify_connectivity()
        
        # Record metrics
        connection_duration = time.time() - connection_start
        neo4j_connection_duration_seconds.observe(connection_duration)
        neo4j_active_connections.inc()

        logger.info(f"Initialized Neo4j driver for: {settings.neo4j_uri}")
        return _neo4j_driver
    except Exception as e:
        neo4j_connection_errors_total.inc()
        logger.error(f"Failed to create Neo4j driver: {str(e)}")
        raise DatabaseError(
            f"Failed to create Neo4j driver: {str(e)}",
            {"neo4j_uri": settings.neo4j_uri},
        ) from e


def close_neo4j_driver() -> None:
    """Close the Neo4j driver connection."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None
        neo4j_active_connections.dec()
        logger.info("Closed Neo4j driver connection")


def reset_neo4j_driver() -> None:
    """Reset the global Neo4j driver (useful for testing)."""
    close_neo4j_driver()

