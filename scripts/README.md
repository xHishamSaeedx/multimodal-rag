# Setup Scripts

This directory contains initialization and setup scripts for the multimodal RAG system.

## Table Extraction Script

### Extract Tables from PDF and DOCX Files

A standalone script to extract tables from PDF and DOCX files for testing and development.

**Prerequisites**:

```bash
# For PDF table extraction (required)
pip install camelot-py[cv]

# Optional: tabula-py as fallback for difficult PDFs
pip install tabula-py

# For DOCX table extraction (already in requirements.txt)
pip install python-docx
```

**Usage**:

```bash
# Extract tables from a PDF file (tries multiple methods automatically)
python scripts/extract_tables.py path/to/file.pdf

# Extract tables from a DOCX file
python scripts/extract_tables.py path/to/file.docx

# Save output to a JSON file
python scripts/extract_tables.py path/to/file.pdf --output tables.json

# Output in markdown format
python scripts/extract_tables.py path/to/file.pdf --format markdown

# Save markdown output to file
python scripts/extract_tables.py path/to/file.pdf --format markdown --output tables.md

# Force a specific extraction method
python scripts/extract_tables.py path/to/file.pdf --method camelot-stream
python scripts/extract_tables.py path/to/file.pdf --method tabula
```

**Options**:

- `file_path`: Path to the PDF or DOCX file (required)
- `--output`, `-o`: Output file path (optional, prints to stdout if not specified)
- `--format`, `-f`: Output format - `json` (default) or `markdown`
- `--method`, `-m`: Extraction method for PDFs - `auto` (tries all methods), `camelot-lattice`, `camelot-stream`, or `tabula` (default: `auto`)

**Extraction Methods**:

- **auto** (default): Tries multiple methods in order:
  1. `camelot-lattice` - Best for tables with visible borders/lines
  2. `camelot-stream` - Best for tables without visible borders
  3. `tabula` - Fallback method using tabula-py
- **camelot-lattice**: Uses camelot with lattice detection (requires visible table borders)
- **camelot-stream**: Uses camelot with stream detection (works for tables without borders)
- **tabula**: Uses tabula-py library (alternative extraction method)

**Note**: The script automatically filters out empty tables and cleans up extracted data.

**Output Formats**:

- **JSON**: Structured data with table information, headers, rows, and metadata
- **Markdown**: Human-readable markdown tables with headers and separators

**Example Output (JSON)**:

```json
{
  "file_path": "document.pdf",
  "file_type": ".pdf",
  "table_count": 2,
  "tables": [
    {
      "table_index": 1,
      "page": 1,
      "accuracy": 95.5,
      "data": [["Header1", "Header2"], ["Value1", "Value2"]],
      "headers": ["Header1", "Header2"],
      "rows": [["Value1", "Value2"]]
    }
  ]
}
```

**Note**: This is a standalone testing/development script. For production use, table extraction will be integrated into the backend service.

## Qdrant Setup

### Prerequisites

1. **Start Qdrant service**:

   ```bash
   docker-compose up -d qdrant
   ```

2. **Verify Qdrant is running**:

   ```bash
   curl http://localhost:6333/health
   ```

   Or visit: http://localhost:6333/dashboard

3. **Install Python dependencies**:
   ```bash
   pip install qdrant-client
   ```

### Initialize Qdrant Collections

The initialization script supports creating collections for Phase 1 (text-only) and Phase 2 (multimodal) RAG systems.

#### Command Options

- `--url`: Qdrant server URL (default: http://localhost:6333)
- `--collection`: Collection name - `text_chunks`, `table_chunks`, or `image_chunks`
- `--vector-size`: Vector dimension - 384, 512, or 768 (default depends on collection type)
- `--recreate`: Delete existing collection and create new one
- `--multimodal`: Create all Phase 2 multimodal collections (`table_chunks`, `image_chunks`)
- `--all`: Create all collections (Phase 1 + Phase 2: `text_chunks`, `table_chunks`, `image_chunks`)
- `--image-vector-size`: Vector dimension for `image_chunks` - 512 (CLIP base, default) or 768 (SigLIP)
- `--verify`: Verify existing collections configuration (no changes made)

#### Phase 1: Text-Only RAG (Initial Setup)

**Create text_chunks collection**:

```bash
# Default setup (768 dimensions for e5-base-v2/all-mpnet-base-v2 - best quality, recommended)
# Supports sub-second retrieval (< 200-300ms)
python scripts/init_qdrant.py

# Or explicitly specify collection
python scripts/init_qdrant.py --collection text_chunks

# For 384 dimensions (all-MiniLM-L6-v2 - faster embedding generation)
python scripts/init_qdrant.py --collection text_chunks --vector-size 384

# Recreate collection (if you need to reset)
python scripts/init_qdrant.py --collection text_chunks --recreate

# Custom Qdrant URL
python scripts/init_qdrant.py --collection text_chunks --url http://localhost:6333
```

#### Phase 2: Multimodal RAG (Tables & Images)

**Create all Phase 2 collections** (recommended):

```bash
# Create table_chunks and image_chunks collections
# Uses 768 dim for tables, 512 dim for images (CLIP base)
python scripts/init_qdrant.py --multimodal

# Use SigLIP (768 dimensions) for images instead
python scripts/init_qdrant.py --multimodal --image-vector-size 768
```

**Create individual Phase 2 collections**:

```bash
# Create table_chunks collection (768 dimensions)
python scripts/init_qdrant.py --collection table_chunks --vector-size 768

# Create image_chunks collection with CLIP base (512 dimensions)
python scripts/init_qdrant.py --collection image_chunks --vector-size 512

# Create image_chunks collection with SigLIP (768 dimensions)
python scripts/init_qdrant.py --collection image_chunks --vector-size 768
```

**Create all collections (Phase 1 + Phase 2)**:

```bash
# Create text_chunks, table_chunks, and image_chunks
# Uses 768 dim for text/tables, 512 dim for images (CLIP base)
python scripts/init_qdrant.py --all

# Customize vector sizes
python scripts/init_qdrant.py --all --vector-size 768 --image-vector-size 512

# Recreate all collections
python scripts/init_qdrant.py --all --recreate
```

#### Verify Collections

**Check all collections are properly configured**:

```bash
# Verify all collections exist and have correct dimensions
python scripts/init_qdrant.py --verify
```

This will show:
- Which collections exist
- Their vector dimensions
- Their status and point counts
- Any configuration issues

#### Common Scenarios

**Scenario 1: Starting fresh (Phase 1 + Phase 2)**
```bash
# Create all collections at once
python scripts/init_qdrant.py --all
```

**Scenario 2: Adding Phase 2 to existing Phase 1 setup**
```bash
# Only create Phase 2 collections (text_chunks already exists)
python scripts/init_qdrant.py --multimodal
```

**Scenario 3: Creating individual collections**
```bash
# Create each collection separately
python scripts/init_qdrant.py --collection text_chunks
python scripts/init_qdrant.py --collection table_chunks
python scripts/init_qdrant.py --collection image_chunks --vector-size 512
```

**Scenario 4: Resetting a collection**
```bash
# Delete and recreate a specific collection
python scripts/init_qdrant.py --collection text_chunks --recreate
```

**Scenario 5: Checking setup before starting**
```bash
# Verify everything is configured correctly
python scripts/init_qdrant.py --verify
```

### Environment Variables

You can also use environment variables:

```bash
export QDRANT_URL=http://localhost:6333
export QDRANT_VECTOR_SIZE=384
python scripts/init_qdrant.py
```

### Collection Configurations

#### Phase 1: text_chunks Collection

- **Vector size**: 768 dimensions (default, for e5-base-v2/all-mpnet-base-v2) or 384 (for all-MiniLM-L6-v2)
- **Distance metric**: Cosine similarity (best for text embeddings)
- **Performance**: 768 dimensions supports sub-second retrieval (< 200-300ms total)
- **Payload fields** (stored with each vector):
  - `chunk_id`: UUID of the chunk
  - `document_id`: UUID of the parent document
  - `text`: The actual chunk text
  - `metadata`: JSON object with additional metadata (page_number, section, etc.)

#### Phase 2: table_chunks Collection

- **Vector size**: 768 dimensions (matching text embeddings)
- **Distance metric**: Cosine similarity
- **Payload fields**:
  - `chunk_id`: UUID of the chunk
  - `document_id`: UUID of the parent document
  - `table_data`: Structured table data (JSONB)
  - `table_markdown`: Markdown representation of the table
  - `metadata`: JSON object with table metadata (row_count, col_count, headers, etc.)

#### Phase 2: image_chunks Collection

- **Vector size**: 512 dimensions (CLIP base, default) or 768 dimensions (SigLIP)
- **Distance metric**: Cosine similarity
- **Payload fields**:
  - `chunk_id`: UUID of the chunk
  - `document_id`: UUID of the parent document
  - `image_path`: Path to the stored image file
  - `caption`: Optional image caption
  - `image_type`: Type of image (diagram, chart, photo, screenshot)
  - `metadata`: JSON object with image metadata (dimensions, format, page_number, etc.)

### Performance Notes

**With 768 dimensions (e5-base-v2):**

- Query embedding: 40-100ms (CPU) / 8-20ms (GPU)
- Vector search: 10-50ms (Qdrant is very efficient even with 768 dim)
- BM25 search: 10-30ms (parallel with vector search)
- Total retrieval: **~65-190ms** ✅ Well under 1 second

The additional dimension (384 → 768) adds minimal overhead to vector search (~10-20ms) but provides significantly better semantic understanding and retrieval quality.

### Verify Collections

**Using the script** (recommended):

```bash
# Verify all collections
python scripts/init_qdrant.py --verify
```

**Using Python**:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Check text_chunks
collection_info = client.get_collection("text_chunks")
print(f"text_chunks: {collection_info.config.params.vectors.size} dimensions")

# Check table_chunks (Phase 2)
collection_info = client.get_collection("table_chunks")
print(f"table_chunks: {collection_info.config.params.vectors.size} dimensions")

# Check image_chunks (Phase 2)
collection_info = client.get_collection("image_chunks")
print(f"image_chunks: {collection_info.config.params.vectors.size} dimensions")
```

**Using Qdrant Dashboard**:

Visit http://localhost:6333/dashboard to view all collections, their configurations, and point counts.

---

## Elasticsearch Setup

### Prerequisites

1. **Start Elasticsearch service**:

   ```bash
   docker-compose up -d elasticsearch
   ```

   Note: Elasticsearch may take 30-60 seconds to fully start on first run.

2. **Verify Elasticsearch is running**:

   ```bash
   curl http://localhost:9200/_cluster/health
   ```

   Should return: `{"status":"green"}` or `{"status":"yellow"}` for single-node setup.

   Or visit: http://localhost:9200

3. **Install Python dependencies**:

   ```bash
   pip install "elasticsearch>=8.0.0,<9.0.0"
   ```

   **Important**: Use Elasticsearch client version 8.x to match the Elasticsearch 8.11.0 server version.
   Client version 9.x uses protocol version 9, which is incompatible with Elasticsearch 8.x servers.

### Initialize Elasticsearch Index

Run the initialization script to create the `chunks` index:

```bash
python scripts/init_elasticsearch.py
```

**Options**:

- `--url`: Elasticsearch server URL (default: http://localhost:9200)
- `--index`: Index name (default: chunks)
- `--recreate`: Delete existing index and create new one

**Examples**:

```bash
# Default setup - creates chunks index with BM25 similarity
python scripts/init_elasticsearch.py

# Recreate index (if you need to reset)
python scripts/init_elasticsearch.py --recreate

# Custom Elasticsearch URL
python scripts/init_elasticsearch.py --url http://localhost:9200
```

### Update Elasticsearch Mapping for Phase 2 (Multimodal)

If you have an existing `chunks` index from Phase 1, update it to support multimodal content:

```bash
python scripts/update_elasticsearch_mapping.py
```

**Options**:

- `--url`: Elasticsearch server URL (default: http://localhost:9200)
- `--index`: Index name to update (default: chunks)
- `--dry-run`: Show what would be updated without making changes

**Examples**:

```bash
# Update existing index mapping for Phase 2 support
python scripts/update_elasticsearch_mapping.py

# Dry run (see what would be updated)
python scripts/update_elasticsearch_mapping.py --dry-run

# Custom Elasticsearch URL
python scripts/update_elasticsearch_mapping.py --url http://localhost:9200
```

**What gets added**:

- `chunk_type` (keyword): text, table, image, mixed
- `embedding_type` (keyword): text, table, image
- `table_markdown` (text): Searchable table content
- `image_caption` (text): Searchable image captions
- Extended `metadata` fields:
  - `image_type` (keyword): diagram, chart, photo, screenshot
  - `table_headers` (text): Searchable table headers
  - `row_count` (integer): Number of rows in table
  - `col_count` (integer): Number of columns in table

**Important Notes**:

- ✅ **Backward compatible**: Existing documents continue to work
- ✅ **Non-destructive**: Only adds new fields, doesn't modify existing ones
- ✅ **Safe to run**: Can be run multiple times (idempotent)
- ⚠️ **Index must exist**: Run `init_elasticsearch.py` first if the index doesn't exist

### Environment Variables

You can also use environment variables:

```bash
export ELASTICSEARCH_URL=http://localhost:9200
export ELASTICSEARCH_INDEX=chunks
python scripts/init_elasticsearch.py
```

### Index Configuration

The `chunks` index is configured with:

- **Similarity**: BM25 (default, optimized for full-text search)
  - k1: 1.2 (term frequency saturation)
  - b: 0.75 (length normalization)
- **Analyzer**: Standard analyzer (tokenization for BM25)
- **Shards**: 1 (single shard for single-node setup)
- **Replicas**: 0 (no replicas for single-node setup)
- **Indexed fields** (Phase 1):
  - `chunk_text`: Full-text search field (primary BM25 search target)
  - `chunk_id`: Keyword field (exact match)
  - `document_id`: Keyword field (exact match, for filtering)
  - `filename`: Text field with keyword subfield (searchable and filterable)
  - `document_type`: Keyword field (for filtering)
  - `source_path`: Keyword field (for filtering)
  - `metadata`: Object field with nested properties:
    - `title`: Text and keyword
    - `tags`: Keyword array
    - `author`: Keyword
    - `version`: Keyword
    - `page_number`: Integer
    - `section`: Text
    - `chunk_index`: Integer
  - `created_at`: Date field
  - `updated_at`: Date field

- **Additional indexed fields** (Phase 2 - added via `update_elasticsearch_mapping.py`):
  - `chunk_type`: Keyword field (text, table, image, mixed)
  - `embedding_type`: Keyword field (text, table, image)
  - `table_markdown`: Text field (searchable table content)
  - `image_caption`: Text field (searchable image captions)
  - Extended `metadata` properties:
    - `image_type`: Keyword (diagram, chart, photo, screenshot)
    - `table_headers`: Text (searchable table headers)
    - `row_count`: Integer (number of rows)
    - `col_count`: Integer (number of columns)

### Performance Notes

**BM25 Search Performance**:

- Query latency: 10-50ms for typical searches (1k-100k documents)
- Indexing speed: 1000-5000 documents/second (depends on document size)
- Memory: ~512MB allocated (configurable in docker-compose.yml)

**Hybrid Retrieval Performance**:

- BM25 search: 10-30ms (parallel with vector search)
- Vector search: 10-50ms (Qdrant)
- Result merging: 5-10ms
- Total retrieval: **~65-190ms** ✅ Well under 1 second

### Verify Index

After initialization, you can verify the index was created:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Check if index exists
if es.indices.exists(index="chunks"):
    # Get index info
    index_info = es.indices.get(index="chunks")
    print(index_info)

    # Get index stats
    stats = es.indices.stats(index="chunks")
    print(f"Document count: {stats['indices']['chunks']['total']['docs']['count']}")
```

Or use curl:

```bash
# Get index information
curl http://localhost:9200/chunks

# Get index stats
curl http://localhost:9200/chunks/_stats

# Search test (should return empty results initially)
curl -X POST "http://localhost:9200/chunks/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match_all": {}
  }
}'
```
