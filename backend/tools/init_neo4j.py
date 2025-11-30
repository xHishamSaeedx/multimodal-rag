"""
Initialize Neo4j database with constraints and indexes.

Run this script once after setting up Neo4j to create the necessary
constraints and indexes for optimal performance.
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.neo4j_database import get_neo4j_driver, close_neo4j_driver
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_constraints(driver):
    """Create uniqueness constraints on node IDs."""
    constraints = [
        # Document constraints
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",

        # Section constraints
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",

        # Chunk constraints
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",

        # Entity constraints
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",

        # Media constraints
        "CREATE CONSTRAINT media_id IF NOT EXISTS FOR (m:Media) REQUIRE m.media_id IS UNIQUE",
    ]

    session = driver.session(database=settings.neo4j_database)
    try:
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"Created constraint: {constraint.split()[2]}")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {str(e)}")
    finally:
        session.close()


def create_indexes(driver):
    """Create indexes for common query patterns."""
    indexes = [
        # Entity type index (for filtering by entity type)
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",

        # Entity name index (for text search on entity names)
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.entity_name)",

        # Media type index
        "CREATE INDEX media_type IF NOT EXISTS FOR (m:Media) ON (m.media_type)",

        # Document title index
        "CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
    ]

    session = driver.session(database=settings.neo4j_database)
    try:
        for index in indexes:
            try:
                session.run(index)
                logger.info(f"Created index: {index.split()[2]}")
            except Exception as e:
                logger.warning(f"Index may already exist: {str(e)}")
    finally:
        session.close()


def main():
    """Initialize Neo4j database."""
    logger.info("Initializing Neo4j database...")
    logger.info(f"Connecting to: {settings.neo4j_uri}")

    try:
        driver = get_neo4j_driver()

        logger.info("Creating constraints...")
        create_constraints(driver)

        logger.info("Creating indexes...")
        create_indexes(driver)

        logger.info("Neo4j initialization complete!")

    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {str(e)}")
        sys.exit(1)
    finally:
        close_neo4j_driver()


if __name__ == "__main__":
    main()

