# Graph POC (Proof of Concept)

This folder contains a proof-of-concept implementation for the Neo4j knowledge graph schema before integrating it into the main backend.

## Purpose

This POC allows you to:
- Test the graph schema design independently
- Validate the schema structure
- Initialize Neo4j constraints and indexes
- **Ingest documents and build knowledge graphs**
- Experiment with graph queries
- Ensure the design works before moving to production

## Files

- `graph_schema.py`: Defines the complete graph schema including:
  - Node labels (Document, Section, Chunk, Entity, Media)
  - Relationship types (HAS_SECTION, HAS_CHUNK, MENTIONS, etc.)
  - Property keys (standardized property names)
  - Entity types, Media types, Chunk types enums
  - Schema validation helpers
  - Documentation

- `neo4j_connection.py`: Simple Neo4j connection helper for POC
  - Standalone connection management
  - Environment variable support
  - No dependency on backend config

- `init_neo4j.py`: Initialize Neo4j database (Step 7)
  - Creates uniqueness constraints on all node IDs
  - Creates indexes for common query patterns
  - Verifies setup after initialization

- `graph_builder.py`: Graph builder utilities implementing best practices:
  - Store references, not duplicate content
  - Batch operations for fast graph building
  - Relationship properties for ranking/weight
  - Helper functions for all node and relationship types

- `document_processor.py`: Document processing for ingestion
  - Extracts text from PDFs and text files
  - Identifies sections and chunks
  - Extracts entities (simple pattern-based for POC)
  - Prepares data for graph storage

- `ingest_document.py`: Main ingestion script
  - Processes documents from file paths
  - Stores complete knowledge graph in Neo4j
  - Links entities to chunks and sections

- `test_schema.py`: Test script to validate schema structure
- `test_best_practices.py`: Test script to verify best practices implementation
- `test_init.py`: Test script to verify initialization structure

## Schema Design

Following the semantic, hierarchical design from `graph.md`:

- **Vertical Structure**: Document -> Section -> Chunk (hierarchical)
- **Horizontal Meaning**: Chunk -> Entity (semantic connections)
- **Sparse Relationships**: Only strong semantic connections between entities
- **Multimodal Support**: Chunk -> Media -> Entity

## Setup

### Prerequisites

1. Neo4j must be running (via Docker Compose or standalone)
2. Python 3.10+ with required packages

### Install Dependencies

```bash
cd graph_poc
pip install -r requirements.txt
```

### Environment Variables (Optional)

You can set these environment variables, or the script will use defaults:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jpassword
export NEO4J_DATABASE=neo4j
```

**Note:** The default password `neo4jpassword` matches the docker-compose.yml configuration.

### Initialize Neo4j Database

Run the initialization script to create constraints and indexes:

```bash
python init_neo4j.py
```

This will:
1. Connect to Neo4j
2. Create uniqueness constraints on all node ID properties
3. Create indexes for common query patterns
4. Verify the setup

### Test Schema

Test the schema definitions:

```bash
python test_schema.py
```

### Test Best Practices

Verify that best practices are implemented:

```bash
python test_best_practices.py
```

## Document Ingestion

### Ingest a Document

Process a document and store it in the knowledge graph:

```bash
python ingest_document.py <file_path>
```

**Examples:**
```bash
# Ingest a PDF
python ingest_document.py document.pdf

# Ingest a text file
python ingest_document.py document.txt
```

**What it does:**
1. Extracts text from the document
2. Identifies sections (by headings)
3. Chunks the text into manageable pieces
4. Extracts entities (organizations, metrics, etc.)
5. Creates the complete graph structure in Neo4j:
   - Document node
   - Section nodes
   - Chunk nodes
   - Entity nodes
   - Relationships (HAS_SECTION, HAS_CHUNK, MENTIONS, ABOUT)

**Output:**
- Document ID (for querying)
- Statistics (sections, chunks, entities created)

## Usage Examples

### Using Schema Constants

```python
from graph_schema import NODE_LABELS, RELATIONSHIP_TYPES, validate_entity_type

# Use constants in queries
query = f"CREATE (d:{NODE_LABELS['DOCUMENT']} {{document_id: $doc_id}})"
relationship = RELATIONSHIP_TYPES['HAS_SECTION']

# Validate entity types
if validate_entity_type("PERSON"):
    print("Valid entity type")
```

### Connecting to Neo4j

```python
from neo4j_connection import get_neo4j_driver, close_neo4j_driver

# Connect (uses environment variables or defaults)
driver = get_neo4j_driver()

# Use the driver
with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as count")
    print(result.single()['count'])

# Close connection
close_neo4j_driver()
```

### Using Best Practices Helpers

```python
from graph_builder import (
    create_document_node_data,
    create_chunk_node_data,
    create_document_graph_batch,
    create_mentions_relationship,
)
from neo4j_connection import get_neo4j_driver, close_neo4j_driver

# Get driver
driver = get_neo4j_driver()

# Create document graph in batch (10-100x faster than one-by-one)
document_data = {
    "document_id": "doc_123",
    "title": "My Document",
    "source": "/path/to/doc.pdf",
    "sections": [
        {
            "section_id": "sec_1",
            "title": "Introduction",
            "index": 1,
            "chunks": [
                {
                    "chunk_id": "chunk_1",
                    "chunk_index": 1,
                    "chunk_type": "text"
                }
            ]
        }
    ]
}
create_document_graph_batch(driver, document_data)

# Create relationship with properties for ranking
create_mentions_relationship(
    driver,
    chunk_id="chunk_1",
    entity_id="ent_1",
    frequency=3,
    importance=0.85,
    context="body"
)

close_neo4j_driver()
```

### Ingesting Documents Programmatically

```python
from ingest_document import ingest_document

# Ingest a document
document_id = ingest_document("path/to/document.pdf")
print(f"Document ingested with ID: {document_id}")
```

## Best Practices Implementation

This POC implements all three best practices from the Neo4j setup guide:

### 1. Store References, Not Duplicate Content
- ✅ Document nodes store only `document_id`, `title`, `source`, `metadata`
- ✅ Chunk nodes store only `chunk_id` reference (content remains in Supabase/Qdrant)
- ✅ Helper functions: `create_document_node_data()`, `create_chunk_node_data()`

### 2. Batch Operations for Graph Building
- ✅ Batch document creation: `create_document_graph_batch()` (uses UNWIND pattern)
- ✅ Batch chunk creation: `create_chunks_batch()`
- ✅ Batch relationship creation: `create_relationships_batch()`
- ✅ **Performance**: 10-100x faster than one-by-one operations

### 3. Relationship Properties for Ranking/Weight
- ✅ MENTIONS: `importance`, `frequency`, `context`
- ✅ ABOUT: `importance`, `frequency`
- ✅ HAS_MEDIA: `relevance`, `position`, `media_type`
- ✅ HAS_SECTION/HAS_CHUNK: `order`, `chunk_index`
- ✅ Helper functions: `create_mentions_relationship()`, `create_about_relationship()`, etc.

## Next Steps

After validating this POC:
1. Move `graph_schema.py` to `backend/app/repositories/graph_schema.py`
2. Move `neo4j_connection.py` logic to `backend/app/core/neo4j_database.py`
3. Move `init_neo4j.py` to `backend/tools/init_neo4j.py`
4. Integrate `graph_builder.py` into backend services
5. Integrate `document_processor.py` with existing ingestion pipeline
6. Add graph querying capabilities
7. Test document processing and graph storage
8. Test graph querying
