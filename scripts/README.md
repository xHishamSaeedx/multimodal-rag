# Setup Scripts

This directory contains initialization and setup scripts for the multimodal RAG system.

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

### Initialize Qdrant Collection

Run the initialization script to create the `text_chunks` collection:

```bash
python scripts/init_qdrant.py
```

**Options**:

- `--url`: Qdrant server URL (default: http://localhost:6333)
- `--collection`: Collection name (default: text_chunks)
- `--vector-size`: Vector dimension - 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2 (default: 384)
- `--recreate`: Delete existing collection and create new one

**Examples**:

```bash
# Default setup (768 dimensions for e5-base-v2/all-mpnet-base-v2 - best quality, recommended)
# Supports sub-second retrieval (< 200-300ms)
python scripts/init_qdrant.py

# For 384 dimensions (all-MiniLM-L6-v2 - faster embedding generation)
python scripts/init_qdrant.py --vector-size 384

# Recreate collection (if you need to reset)
python scripts/init_qdrant.py --recreate

# Custom Qdrant URL
python scripts/init_qdrant.py --url http://localhost:6333
```

### Environment Variables

You can also use environment variables:

```bash
export QDRANT_URL=http://localhost:6333
export QDRANT_VECTOR_SIZE=384
python scripts/init_qdrant.py
```

### Collection Configuration

The `text_chunks` collection is configured with:

- **Vector size**: 768 dimensions (default, for e5-base-v2/all-mpnet-base-v2) or 384 (for all-MiniLM-L6-v2)
- **Distance metric**: Cosine similarity (best for text embeddings)
- **Performance**: 768 dimensions supports sub-second retrieval (< 200-300ms total)
- **Payload fields** (to be stored with each vector):
  - `chunk_id`: UUID of the chunk
  - `document_id`: UUID of the parent document
  - `text`: The actual chunk text
  - `metadata`: JSON object with additional metadata (page_number, section, etc.)

### Performance Notes

**With 768 dimensions (e5-base-v2):**

- Query embedding: 40-100ms (CPU) / 8-20ms (GPU)
- Vector search: 10-50ms (Qdrant is very efficient even with 768 dim)
- BM25 search: 10-30ms (parallel with vector search)
- Total retrieval: **~65-190ms** ✅ Well under 1 second

The additional dimension (384 → 768) adds minimal overhead to vector search (~10-20ms) but provides significantly better semantic understanding and retrieval quality.

### Verify Collection

After initialization, you can verify the collection was created:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_info = client.get_collection("text_chunks")
print(collection_info)
```

Or use the Qdrant dashboard: http://localhost:6333/dashboard

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

Run the initialization script to create the `text_chunks` index:

```bash
python scripts/init_elasticsearch.py
```

**Options**:

- `--url`: Elasticsearch server URL (default: http://localhost:9200)
- `--index`: Index name (default: text_chunks)
- `--recreate`: Delete existing index and create new one

**Examples**:

```bash
# Default setup - creates text_chunks index with BM25 similarity
python scripts/init_elasticsearch.py

# Recreate index (if you need to reset)
python scripts/init_elasticsearch.py --recreate

# Custom Elasticsearch URL
python scripts/init_elasticsearch.py --url http://localhost:9200
```

### Environment Variables

You can also use environment variables:

```bash
export ELASTICSEARCH_URL=http://localhost:9200
export ELASTICSEARCH_INDEX=text_chunks
python scripts/init_elasticsearch.py
```

### Index Configuration

The `text_chunks` index is configured with:

- **Similarity**: BM25 (default, optimized for full-text search)
  - k1: 1.2 (term frequency saturation)
  - b: 0.75 (length normalization)
- **Analyzer**: Standard analyzer (tokenization for BM25)
- **Shards**: 1 (single shard for single-node setup)
- **Replicas**: 0 (no replicas for single-node setup)
- **Indexed fields**:
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
if es.indices.exists(index="text_chunks"):
    # Get index info
    index_info = es.indices.get(index="text_chunks")
    print(index_info)

    # Get index stats
    stats = es.indices.stats(index="text_chunks")
    print(f"Document count: {stats['indices']['text_chunks']['total']['docs']['count']}")
```

Or use curl:

```bash
# Get index information
curl http://localhost:9200/text_chunks

# Get index stats
curl http://localhost:9200/text_chunks/_stats

# Search test (should return empty results initially)
curl -X POST "http://localhost:9200/text_chunks/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match_all": {}
  }
}'
```
