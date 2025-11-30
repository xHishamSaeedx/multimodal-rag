"""
Quick script to clean out Neo4j database - deletes all nodes and relationships.

Usage:
    python scripts/cleanup_neo4j.py
    python scripts/cleanup_neo4j.py --confirm  # Skips confirmation prompt
"""
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.core.neo4j_database import get_neo4j_driver
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def cleanup_neo4j(confirm: bool = False):
    """
    Delete all nodes and relationships from Neo4j database.
    
    Args:
        confirm: If True, skip confirmation prompt
    """
    if not confirm:
        response = input("⚠️  WARNING: This will delete ALL nodes and relationships from Neo4j. Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    print(f"Connecting to Neo4j: {settings.neo4j_uri} (database: {settings.neo4j_database})")
    
    driver = get_neo4j_driver()
    session = driver.session(database=settings.neo4j_database)
    
    try:
        # Delete all relationships first, then nodes
        # Using DETACH DELETE to handle orphaned nodes
        query = "MATCH (n) DETACH DELETE n"
        
        print("Deleting all nodes and relationships...")
        result = session.run(query)
        summary = result.consume()
        
        print(f"✅ Successfully deleted all nodes and relationships")
        print(f"   - Nodes deleted: {summary.counters.nodes_deleted}")
        print(f"   - Relationships deleted: {summary.counters.relationships_deleted}")
        
    except Exception as e:
        print(f"❌ Error cleaning Neo4j: {e}")
        logger.error(f"Failed to clean Neo4j: {e}", exc_info=True)
        raise
    finally:
        session.close()
        driver.close()
        print("Connection closed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean out Neo4j database")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    try:
        cleanup_neo4j(confirm=args.confirm)
        print("\n✅ Neo4j cleanup complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Neo4j cleanup failed: {e}")
        sys.exit(1)

