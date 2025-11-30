"""
Simple Neo4j connection helper for POC.

This module provides basic Neo4j connection functionality for testing
the graph schema independently before integration into the backend.
"""
from neo4j import GraphDatabase, Driver
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

# Global Neo4j driver instance
_neo4j_driver: Optional[Driver] = None


def get_neo4j_driver(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None
) -> Driver:
    """
    Get or create Neo4j driver instance.
    
    Args:
        uri: Neo4j URI (defaults to NEO4J_URI env var or bolt://localhost:7687)
        user: Neo4j username (defaults to NEO4J_USER env var or 'neo4j')
        password: Neo4j password (defaults to NEO4J_PASSWORD env var or 'neo4jpassword')
        database: Database name (defaults to NEO4J_DATABASE env var or 'neo4j')
    
    Returns:
        Neo4j driver instance
    
    Raises:
        Exception: If Neo4j connection fails
    """
    global _neo4j_driver
    
    if _neo4j_driver is not None:
        return _neo4j_driver
    
    # Get values from environment or use defaults
    # Default password matches docker-compose.yml: NEO4J_AUTH=neo4j/neo4jpassword
    neo4j_uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = user or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = password or os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    neo4j_database = database or os.getenv("NEO4J_DATABASE", "neo4j")
    
    try:
        _neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            max_connection_pool_size=50,
            connection_timeout=30,
        )
        
        # Verify connection
        _neo4j_driver.verify_connectivity()
        
        logger.info(f"Initialized Neo4j driver for: {neo4j_uri}")
        logger.info(f"Using database: {neo4j_database}")
        return _neo4j_driver
    except Exception as e:
        logger.error(f"Failed to create Neo4j driver: {str(e)}")
        raise Exception(f"Failed to create Neo4j driver: {str(e)}") from e


def close_neo4j_driver() -> None:
    """Close the Neo4j driver connection."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None
        logger.info("Closed Neo4j driver connection")


def reset_neo4j_driver() -> None:
    """Reset the global Neo4j driver (useful for testing)."""
    close_neo4j_driver()


def get_database_name() -> str:
    """Get the database name from environment or default."""
    return os.getenv("NEO4J_DATABASE", "neo4j")

