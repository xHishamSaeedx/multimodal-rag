"""
Initialize Neo4j database with constraints and indexes.

Run this script once after setting up Neo4j to create the necessary
constraints and indexes for optimal performance.

This is a POC version that can be tested independently before integration
into the main backend.

Usage:
    python init_neo4j.py
    
Environment Variables (optional):
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (default: neo4jpassword - matches docker-compose.yml)
    NEO4J_DATABASE: Database name (default: neo4j)
"""
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from graph_schema import NODE_LABELS, PROPERTY_KEYS
from neo4j_connection import get_neo4j_driver, close_neo4j_driver, get_database_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_constraints(driver):
    """Create uniqueness constraints on node IDs."""
    constraints = [
        # Document constraints
        f"CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:{NODE_LABELS['DOCUMENT']}) "
        f"REQUIRE d.{PROPERTY_KEYS['DOCUMENT_ID']} IS UNIQUE",
        
        # Section constraints
        f"CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:{NODE_LABELS['SECTION']}) "
        f"REQUIRE s.{PROPERTY_KEYS['SECTION_ID']} IS UNIQUE",
        
        # Chunk constraints
        f"CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:{NODE_LABELS['CHUNK']}) "
        f"REQUIRE c.{PROPERTY_KEYS['CHUNK_ID']} IS UNIQUE",
        
        # Entity constraints
        f"CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:{NODE_LABELS['ENTITY']}) "
        f"REQUIRE e.{PROPERTY_KEYS['ENTITY_ID']} IS UNIQUE",
        
        # Media constraints
        f"CREATE CONSTRAINT media_id IF NOT EXISTS FOR (m:{NODE_LABELS['MEDIA']}) "
        f"REQUIRE m.{PROPERTY_KEYS['MEDIA_ID']} IS UNIQUE",
    ]
    
    database = get_database_name()
    with driver.session(database=database) as session:
        for constraint in constraints:
            try:
                session.run(constraint)
                # Extract constraint name for logging
                constraint_name = constraint.split()[2]
                logger.info(f"Created constraint: {constraint_name}")
            except Exception as e:
                error_msg = str(e)
                # Check if constraint already exists (this is okay)
                if "already exists" in error_msg.lower() or "equivalent" in error_msg.lower():
                    logger.info(f"Constraint already exists: {constraint.split()[2]}")
                else:
                    logger.warning(f"Constraint creation issue: {error_msg}")


def create_indexes(driver):
    """Create indexes for common query patterns."""
    indexes = [
        # Entity type index (for filtering by entity type)
        f"CREATE INDEX entity_type IF NOT EXISTS FOR (e:{NODE_LABELS['ENTITY']}) "
        f"ON (e.{PROPERTY_KEYS['ENTITY_TYPE']})",
        
        # Entity name index (for text search on entity names)
        f"CREATE INDEX entity_name IF NOT EXISTS FOR (e:{NODE_LABELS['ENTITY']}) "
        f"ON (e.{PROPERTY_KEYS['ENTITY_NAME']})",
        
        # Media type index
        f"CREATE INDEX media_type IF NOT EXISTS FOR (m:{NODE_LABELS['MEDIA']}) "
        f"ON (m.{PROPERTY_KEYS['MEDIA_TYPE']})",
        
        # Document title index
        f"CREATE INDEX document_title IF NOT EXISTS FOR (d:{NODE_LABELS['DOCUMENT']}) "
        f"ON (d.{PROPERTY_KEYS['TITLE']})",
        
        # Section index for faster section lookups
        f"CREATE INDEX section_index IF NOT EXISTS FOR (s:{NODE_LABELS['SECTION']}) "
        f"ON (s.{PROPERTY_KEYS['SECTION_INDEX']})",
        
        # Chunk index for faster chunk lookups
        f"CREATE INDEX chunk_index IF NOT EXISTS FOR (c:{NODE_LABELS['CHUNK']}) "
        f"ON (c.{PROPERTY_KEYS['CHUNK_INDEX']})",
    ]
    
    database = get_database_name()
    with driver.session(database=database) as session:
        for index in indexes:
            try:
                session.run(index)
                # Extract index name for logging
                index_name = index.split()[2]
                logger.info(f"Created index: {index_name}")
            except Exception as e:
                error_msg = str(e)
                # Check if index already exists (this is okay)
                if "already exists" in error_msg.lower() or "equivalent" in error_msg.lower():
                    logger.info(f"Index already exists: {index.split()[2]}")
                else:
                    logger.warning(f"Index creation issue: {error_msg}")


def verify_setup(driver):
    """Verify that constraints and indexes were created successfully."""
    database = get_database_name()
    with driver.session(database=database) as session:
        # Check constraints
        constraints_result = session.run("SHOW CONSTRAINTS")
        constraint_count = 0
        logger.info("\n[Constraints]")
        for record in constraints_result:
            constraint_count += 1
            logger.info(f"  - {record.get('name', 'N/A')}: {record.get('type', 'N/A')}")
        
        # Check indexes
        indexes_result = session.run("SHOW INDEXES")
        index_count = 0
        logger.info("\n[Indexes]")
        for record in indexes_result:
            # Only show our indexes (filter out internal ones)
            index_name = record.get('name', '')
            if any(keyword in index_name.lower() for keyword in ['document', 'section', 'chunk', 'entity', 'media']):
                index_count += 1
                logger.info(f"  - {index_name}: {record.get('type', 'N/A')}")
        
        logger.info(f"\n[Summary] Created {constraint_count} constraints and {index_count} indexes")


def main():
    """Initialize Neo4j database."""
    logger.info("=" * 60)
    logger.info("Initializing Neo4j database for Graph POC")
    logger.info("=" * 60)
    
    import os
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    logger.info(f"Connecting to: {neo4j_uri}")
    logger.info(f"Database: {get_database_name()}")
    
    try:
        driver = get_neo4j_driver()
        
        logger.info("\n[Step 1/3] Creating constraints...")
        create_constraints(driver)
        
        logger.info("\n[Step 2/3] Creating indexes...")
        create_indexes(driver)
        
        logger.info("\n[Step 3/3] Verifying setup...")
        verify_setup(driver)
        
        logger.info("\n" + "=" * 60)
        logger.info("[SUCCESS] Neo4j initialization complete!")
        logger.info("=" * 60)
        logger.info("\nYou can now use the graph schema to create nodes and relationships.")
        logger.info("Next steps:")
        logger.info("  1. Test creating document nodes")
        logger.info("  2. Test creating sections and chunks")
        logger.info("  3. Test entity extraction and linking")
        logger.info("  4. Test querying the graph")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n[ERROR] Failed to initialize Neo4j: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Make sure Neo4j is running (docker-compose up neo4j)")
        logger.error("  2. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables")
        logger.error("  3. Verify Neo4j is accessible at the specified URI")
        return 1
    finally:
        close_neo4j_driver()


if __name__ == "__main__":
    sys.exit(main())

