#!/usr/bin/env python3
"""
Cleanup orphaned entities from Neo4j knowledge graph.

This script removes:
1. Entities with no relationships at all (completely disconnected)
2. Entities not connected to any chunks (no MENTIONS or ABOUT relationships)

This is useful for cleaning up after bugs where entities were created
but never properly linked to document chunks.

Usage:
    python backend/scripts/neo4j/cleanup_orphaned_entities.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.repositories.graph_repository import GraphRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run the orphaned entity cleanup."""
    print("=" * 70)
    print("Cleaning up orphaned entities from Neo4j...")
    print("=" * 70)
    
    try:
        # Initialize graph repository
        graph_repo = GraphRepository()
        
        # Run cleanup
        counts = graph_repo.cleanup_orphaned_entities()
        
        print("\n✅ Cleanup completed successfully!")
        print(f"   - Disconnected entities deleted: {counts['disconnected_entities_deleted']}")
        print(f"   - Unlinked entities deleted: {counts['unlinked_entities_deleted']}")
        print(f"   - Total entities deleted: {counts['total_entities_deleted']}")
        
        if counts['total_entities_deleted'] == 0:
            print("\n✓ No orphaned entities found. Database is clean!")
        else:
            print(f"\n✓ Successfully cleaned up {counts['total_entities_deleted']} orphaned entities")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

