# Neo4j Knowledge Graph Setup Guide

This guide outlines the step-by-step setup process for integrating Neo4j into the multimodal RAG system as a knowledge graph database. The graph will follow the semantic, hierarchical design described in `graph.md`.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Add Neo4j to Docker Compose](#step-1-add-neo4j-to-docker-compose)
4. [Step 2: Install Python Neo4j Driver](#step-2-install-python-neo4j-driver)
5. [Step 3: Update Configuration](#step-3-update-configuration)
6. [Step 4: Create Database Connection Module](#step-4-create-database-connection-module)
7. [Step 5: Set Up Environment Variables](#step-5-set-up-environment-variables)
8. [Step 6: Define Graph Schema](#step-6-define-graph-schema)
9. [Step 7: Initialize Graph Constraints and Indexes](#step-7-initialize-graph-constraints-and-indexes)
10. [Step 8: Test Connection](#step-8-test-connection)
11. [Next Steps](#next-steps)

## Overview

The Neo4j knowledge graph will store:

- **Nodes:**

  - `Document`: Single node per document
  - `Section`: Logical document segments (chapter headings, sections)
  - `Chunk`: Text chunks that serve as retrieval granularity
  - `Entity`: Meaningful entities (people, organizations, concepts, locations, metrics, domain-specific terms)
  - `Media`: Images, tables, charts, diagrams

- **Relationships:**
  - `HAS_SECTION`: Document → Section
  - `HAS_CHUNK`: Section → Chunk
  - `MENTIONS`: Chunk → Entity
  - `ABOUT`: Section → Entity (top entities for each section)
  - `RELATED_TO`: Entity → Entity (strong semantic relationships)
  - `HAS_MEDIA`: Chunk → Media
  - `DESCRIBES`: Media → Entity

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ environment
- Access to the project repository
- Understanding of the existing architecture (Qdrant, Elasticsearch, MinIO)

## Step 1: Add Neo4j to Docker Compose

Add the Neo4j service to `docker-compose.yml`:

```yaml
# Neo4j - Knowledge Graph Database
neo4j:
  image: neo4j:5.15-community
  container_name: multimodal-rag-neo4j
  environment:
    # Authentication
    - NEO4J_AUTH=neo4j/neo4j-password # Change password in production!
    # Memory settings (adjust based on your system)
    - NEO4J_server_memory_heap_initial__size=512m
    - NEO4J_server_memory_heap_max__size=2G
    - NEO4J_server_memory_pagecache_size=1G
    # Enable plugins (APOC is useful for graph algorithms)
    - NEO4J_PLUGINS=["apoc"]
    # Allow HTTP and Bolt connections
    - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    - NEO4J_dbms_security_procedures_allowlist=apoc.*
  ports:
    - "7474:7474" # HTTP
    - "7687:7687" # Bolt (default protocol)
  volumes:
    - neo4j_data:/data
    - neo4j_logs:/logs
    - neo4j_import:/var/lib/neo4j/import
    - neo4j_plugins:/plugins
  healthcheck:
    test:
      [
        "CMD-SHELL",
        "wget --no-verbose --tries=1 --spider http://localhost:7474 || exit 1",
      ]
    interval: 30s
    timeout: 10s
    retries: 5
  networks:
    - rag-network
  restart: unless-stopped
  labels:
    - "logging=promtail"
```

Add the volume to the `volumes` section:

```yaml
volumes:
  # ... existing volumes ...
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
```

**Note:** Change `neo4j-password` to a secure password in production. Store it in your `.env` file.

## Step 2: Install Python Neo4j Driver

Add the Neo4j Python driver to `backend/requirements.txt`:

```txt
# Graph database
neo4j>=5.15.0  # Neo4j Python driver (matches server version 5.15)
```

Then install it:

```bash
cd backend
pip install -r requirements.txt
```

## Step 3: Update Configuration

Add Neo4j settings to `backend/app/core/config.py` in the `Settings` class:

```python
# Neo4j (Knowledge Graph) Settings
neo4j_uri: str = "bolt://localhost:7687"  # Use 'bolt://neo4j:7687' when backend runs in Docker
neo4j_user: str = "neo4j"
neo4j_password: str = "neo4j-password"  # Should be loaded from .env
neo4j_database: str = "neo4j"  # Default database (use 'neo4j' or specific database name)
neo4j_timeout: int = 30  # Connection timeout in seconds
neo4j_max_connection_pool_size: int = 50  # Connection pool size
neo4j_enabled: bool = True  # Feature flag to enable/disable Neo4j
```

## Step 4: Create Database Connection Module

Create `backend/app/core/neo4j_database.py` to handle Neo4j connections (similar to existing `database.py`):

```python
"""
Neo4j database connection and management.

This module provides functions to connect to Neo4j and manage graph database operations.
"""
from neo4j import GraphDatabase, Driver
from typing import Optional
import logging

from backend.app.core.config import settings
from backend.app.utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)

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

    try:
        _neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
            connection_timeout=settings.neo4j_timeout,
        )

        # Verify connection
        _neo4j_driver.verify_connectivity()

        logger.info(f"Initialized Neo4j driver for: {settings.neo4j_uri}")
        return _neo4j_driver
    except Exception as e:
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
        logger.info("Closed Neo4j driver connection")


def reset_neo4j_driver() -> None:
    """Reset the global Neo4j driver (useful for testing)."""
    close_neo4j_driver()
```

## Step 5: Set Up Environment Variables

Add Neo4j configuration to your `.env` file in the `backend/` directory:

```env
# Neo4j Knowledge Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j-password  # Change this to a secure password!
NEO4J_DATABASE=neo4j
NEO4J_TIMEOUT=30
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_ENABLED=true
```

**Security Note:**

- Never commit `.env` files to git (should already be in `.gitignore`)
- Use strong passwords in production
- Consider using environment variable injection in Docker for sensitive data

## Step 6: Define Graph Schema

Create `backend/app/repositories/graph_schema.py` to define the graph schema structure:

```python
"""
Neo4j graph schema definitions.

This module contains Cypher queries and schema definitions for the knowledge graph.
"""

# Node labels
NODE_LABELS = {
    "DOCUMENT": "Document",
    "SECTION": "Section",
    "CHUNK": "Chunk",
    "ENTITY": "Entity",
    "MEDIA": "Media",
}

# Relationship types
RELATIONSHIP_TYPES = {
    "HAS_SECTION": "HAS_SECTION",
    "HAS_CHUNK": "HAS_CHUNK",
    "MENTIONS": "MENTIONS",
    "ABOUT": "ABOUT",
    "RELATED_TO": "RELATED_TO",
    "HAS_MEDIA": "HAS_MEDIA",
    "DESCRIBES": "DESCRIBES",
}

# Property keys (standardized)
PROPERTY_KEYS = {
    # Document
    "DOCUMENT_ID": "document_id",
    "TITLE": "title",
    "SOURCE": "source",
    "METADATA": "metadata",

    # Section
    "SECTION_ID": "section_id",
    "SECTION_TITLE": "section_title",
    "SECTION_INDEX": "section_index",

    # Chunk
    "CHUNK_ID": "chunk_id",
    "CONTENT": "content",
    "CHUNK_INDEX": "chunk_index",

    # Entity
    "ENTITY_ID": "entity_id",
    "ENTITY_TYPE": "entity_type",  # PERSON, ORGANIZATION, CONCEPT, LOCATION, METRIC, etc.
    "ENTITY_NAME": "entity_name",
    "ENTITY_VALUE": "entity_value",  # For metrics

    # Media
    "MEDIA_ID": "media_id",
    "MEDIA_TYPE": "media_type",  # IMAGE, TABLE, CHART, etc.
    "MEDIA_URL": "media_url",
    "CAPTION": "caption",
}
```

## Step 7: Initialize Graph Constraints and Indexes

Create `backend/tools/init_neo4j.py` script to set up constraints and indexes:

```python
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

from backend.app.core.neo4j_database import get_neo4j_driver, close_neo4j_driver
from backend.app.core.config import settings
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

    with driver.session(database=settings.neo4j_database) as session:
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"Created constraint: {constraint.split()[2]}")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {str(e)}")


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

    with driver.session(database=settings.neo4j_database) as session:
        for index in indexes:
            try:
                session.run(index)
                logger.info(f"Created index: {index.split()[2]}")
            except Exception as e:
                logger.warning(f"Index may already exist: {str(e)}")


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
```

Run the initialization script:

```bash
cd backend
python tools/init_neo4j.py
```

## Step 8: Test Connection

Create a simple test script or add a health check endpoint. You can test the connection using the Neo4j Browser or a Python script:

**Using Neo4j Browser:**

1. Start Docker services: `docker-compose up -d neo4j`
2. Open browser: `http://localhost:7474`
3. Login with credentials: `neo4j` / `neo4j-password` (or your configured password)
4. Run a test query: `MATCH (n) RETURN count(n) as node_count`

**Using Python:**
Create a test script `backend/tools/test_neo4j.py`:

```python
"""Test Neo4j connection and basic operations."""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from backend.app.core.neo4j_database import get_neo4j_driver, close_neo4j_driver
from backend.app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_connection():
    """Test Neo4j connection and run a simple query."""
    try:
        driver = get_neo4j_driver()

        with driver.session(database=settings.neo4j_database) as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            logger.info(f"✅ Connection successful! Test value: {record['test']}")

            # Check node counts
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY label
            """)

            logger.info("Node counts:")
            for record in node_counts:
                logger.info(f"  {record['label']}: {record['count']}")

        logger.info("✅ Neo4j is ready to use!")
        return True

    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        return False
    finally:
        close_neo4j_driver()


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
```

Run the test:

```bash
cd backend
python tools/test_neo4j.py
```

## Implementation Best Practices

Before implementing graph building and retrieval, follow these best practices to ensure fast graph construction and efficient querying:

### 1. Store References, Not Duplicate Content

**Problem**: Storing full chunk content in Neo4j duplicates data already stored in Supabase/Qdrant, slowing down graph building and wasting storage.

**Solution**: Store only reference identifiers and metadata in graph nodes:

- **Chunk nodes** should contain:

  - `chunk_id` (primary key, links to existing chunk in Supabase/Qdrant)
  - `chunk_index` (position in document)
  - Optional: `metadata` (lightweight key-value pairs)
  - **Do NOT store**: Full text content (retrieve from existing stores when needed)

- **Document nodes** should contain:
  - `document_id` (primary key)
  - `title`, `source`, `created_at`, `document_type`
  - Lightweight metadata only

**Benefits**:

- Faster graph building (no large text writes)
- Lower storage usage
- Single source of truth (content remains in Supabase/Qdrant)
- Easier to keep graph in sync with content changes

**Example Implementation**:

```python
# ✅ GOOD: Store only reference
chunk_node = {
    "chunk_id": "chunk_123",
    "chunk_index": 5,
    "metadata": {"page": 10}
}

# ❌ BAD: Don't duplicate full content
chunk_node = {
    "chunk_id": "chunk_123",
    "content": "Very long text content here...",  # Don't do this!
    "chunk_index": 5
}
```

When retrieving chunks for RAG, fetch content from Supabase/Qdrant using the `chunk_id` reference.

### 2. Batch Operations for Graph Building

**Problem**: Creating nodes and relationships one-by-one is extremely slow for large documents (hundreds or thousands of chunks).

**Solution**: Use Cypher batch operations with `UNWIND` to create multiple nodes/relationships in a single transaction.

**Pattern 1: Batch Create Nodes**:

```cypher
UNWIND $chunks AS chunk
MERGE (c:Chunk {chunk_id: chunk.chunk_id})
SET c.chunk_index = chunk.chunk_index,
    c.metadata = chunk.metadata
```

**Pattern 2: Batch Create Relationships**:

```cypher
UNWIND $chunk_links AS link
MATCH (s:Section {section_id: link.section_id})
MATCH (c:Chunk {chunk_id: link.chunk_id})
MERGE (s)-[:HAS_CHUNK {
    order: link.order,
    created_at: datetime()
}]->(c)
```

**Pattern 3: Batch Create Entire Document Structure**:

```cypher
UNWIND $document_structure AS doc
MERGE (d:Document {document_id: doc.document_id})
SET d.title = doc.title,
    d.source = doc.source

WITH d, doc
UNWIND doc.sections AS section
MERGE (s:Section {section_id: section.section_id})
SET s.section_title = section.title,
    s.section_index = section.index
MERGE (d)-[:HAS_SECTION {order: section.index}]->(s)

WITH s, section
UNWIND section.chunks AS chunk
MERGE (c:Chunk {chunk_id: chunk.chunk_id})
SET c.chunk_index = chunk.chunk_index
MERGE (s)-[:HAS_CHUNK {order: chunk.chunk_index}]->(c)
```

**Implementation Example**:

```python
def create_document_graph_batch(driver, document_data):
    """Create entire document graph structure in a single transaction."""
    query = """
    UNWIND $data AS doc
    MERGE (d:Document {document_id: doc.document_id})
    SET d.title = doc.title,
        d.source = doc.source
    WITH d, doc
    UNWIND doc.sections AS section
    MERGE (s:Section {section_id: section.section_id})
    SET s.section_title = section.title,
        s.section_index = section.index
    MERGE (d)-[:HAS_SECTION {order: section.index}]->(s)
    WITH s, section
    UNWIND section.chunks AS chunk
    MERGE (c:Chunk {chunk_id: chunk.chunk_id})
    SET c.chunk_index = chunk.chunk_index
    MERGE (s)-[:HAS_CHUNK {order: chunk.chunk_index}]->(c)
    """

    with driver.session() as session:
        session.run(query, data=[document_data])
```

**Performance Gain**: Batch operations are 10-100x faster than individual creates. For a document with 1000 chunks:

- One-by-one: ~60+ seconds
- Batch operation: ~1-3 seconds

### 3. Relationship Properties for Ranking/Weight

**Problem**: Relationships without properties can't express importance, order, or context, limiting retrieval quality.

**Solution**: Add properties to relationships that enable better ranking and filtering during retrieval.

**Properties to Add**:

1. **Order/Position Properties**:

   ```cypher
   (Section)-[:HAS_CHUNK {order: 5, chunk_index: 42}]->(Chunk)
   ```

   - `order`: Position within section
   - `chunk_index`: Global position in document
   - Use for: Ordering chunks correctly, maintaining document flow

2. **Importance/Weight Properties**:

   ```cypher
   (Section)-[:ABOUT {importance: 0.85, frequency: 3}]->(Entity)
   (Chunk)-[:MENTIONS {count: 2, context: "title"}]->(Entity)
   ```

   - `importance`: Relevance score (0.0-1.0)
   - `frequency`: How often entity appears
   - `context`: Where entity appears (title, body, metadata)
   - Use for: Ranking entities, filtering by importance

3. **Temporal/Version Properties**:

   ```cypher
   (Document)-[:HAS_SECTION {created_at: datetime(), version: 1}]->(Section)
   ```

   - `created_at`: When relationship was created
   - `version`: Document version number
   - Use for: Time-based queries, version tracking

4. **Metadata Properties**:
   ```cypher
   (Chunk)-[:HAS_MEDIA {
       media_type: "table",
       position: "after",
       relevance: 0.9
   }]->(Media)
   ```
   - `media_type`: Type of media (image, table, chart)
   - `position`: Where media appears relative to chunk
   - `relevance`: How relevant media is to chunk content
   - Use for: Filtering media by type, ranking media relevance

**Example Queries Using Relationship Properties**:

```cypher
// Get chunks ordered by position
MATCH (d:Document {document_id: $doc_id})-[:HAS_SECTION]->(s)-[r:HAS_CHUNK]->(c)
RETURN c
ORDER BY s.section_index, r.order
LIMIT 10

// Get top entities by importance
MATCH (s:Section)-[r:ABOUT]->(e:Entity)
WHERE r.importance > 0.7
RETURN e, r.importance
ORDER BY r.importance DESC
LIMIT 5

// Get media with high relevance to chunks
MATCH (c:Chunk)-[r:HAS_MEDIA]->(m:Media)
WHERE r.relevance > 0.8 AND r.media_type = "table"
RETURN m, r.relevance
ORDER BY r.relevance DESC
```

**Implementation Example**:

```python
def create_relationship_with_properties(
    driver,
    from_node_label: str,
    from_node_id_key: str,
    from_node_id_value: str,
    to_node_label: str,
    to_node_id_key: str,
    to_node_id_value: str,
    relationship_type: str,
    properties: dict
):
    """Create relationship with properties in a single operation."""
    query = f"""
    MATCH (from:{from_node_label} {{{from_node_id_key}: $from_id}})
    MATCH (to:{to_node_label} {{{to_node_id_key}: $to_id}})
    MERGE (from)-[r:{relationship_type}]->(to)
    SET r += $properties
    """

    with driver.session() as session:
        session.run(query, {
            "from_id": from_node_id_value,
            "to_id": to_node_id_value,
            "properties": properties
        })
```

**Benefits**:

- Better ranking and filtering during retrieval
- Preserves document structure and ordering
- Enables more sophisticated query patterns
- Supports relevance scoring for hybrid retrieval

## Next Steps

After completing the setup:

1. **Create Graph Repository**: Implement `backend/app/repositories/graph_repository.py` with methods for:

   - Creating document/section/chunk/entity/media nodes
   - Creating relationships
   - Querying the graph
   - Graph-based retrieval methods

2. **Integrate with Ingestion Pipeline**: Modify `backend/app/services/ingestion/pipeline.py` to:

   - Create graph nodes during document ingestion
   - Extract entities using NER or LLM-based extraction
   - Build relationships between nodes

3. **Add Graph Retrieval**: Create `backend/app/services/retrieval/graph_retriever.py` for:

   - Multi-hop queries
   - Entity-based retrieval
   - Contextual graph traversal

4. **Update API Routes**: Add graph-related endpoints if needed

5. **Add Monitoring**: Include Neo4j metrics in your observability stack

6. **Documentation**: Update API documentation with graph query capabilities

## Troubleshooting

### Connection Issues

- **Error: "Unable to connect"**

  - Check if Neo4j container is running: `docker ps | grep neo4j`
  - Verify port mappings: `7474` (HTTP) and `7687` (Bolt)
  - Check firewall settings
  - Verify `NEO4J_URI` matches your setup (`bolt://localhost:7687` for local, `bolt://neo4j:7687` for Docker network)

- **Authentication Failed**
  - Verify `NEO4J_USER` and `NEO4J_PASSWORD` match Docker Compose settings
  - Check if password was changed in Neo4j Browser (first login forces password change)

### Performance Issues

- **Slow Queries**

  - Ensure indexes are created (run `init_neo4j.py`)
  - Monitor memory settings in Docker Compose
  - Use query profiling in Neo4j Browser (`PROFILE` or `EXPLAIN`)

- **Out of Memory**
  - Increase `NEO4J_server_memory_heap_max__size` in docker-compose.yml
  - Adjust `NEO4J_server_memory_pagecache_size` based on available RAM

### Database Not Found

- **Error: "Database does not exist"**
  - Neo4j Community Edition uses a single database named `neo4j` by default
  - Set `NEO4J_DATABASE=neo4j` in your `.env` file
  - Enterprise Edition supports multiple databases

## References

- [Neo4j Python Driver Documentation](https://neo4j.com/docs/python-manual/current/)
- [Neo4j Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Docker Hub](https://hub.docker.com/_/neo4j)
- [Neo4j Best Practices](https://neo4j.com/developer/kb/understanding-nodes-relationships-and-properties/)

## Service URLs

After setup, Neo4j will be accessible at:

- **Neo4j Browser (Web UI)**: `http://localhost:7474`
- **Bolt Protocol**: `bolt://localhost:7687` (for application connections)
- **HTTP API**: `http://localhost:7474` (for REST endpoints)

Default credentials (change in production):

- Username: `neo4j`
- Password: `neo4j-password` (or your configured password)
