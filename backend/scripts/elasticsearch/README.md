# Elasticsearch Management Scripts

This directory contains scripts for managing Elasticsearch indexes in the multimodal RAG system.

## Scripts Overview

### 1. `init_elasticsearch.py`
**Purpose**: Initialize or recreate Elasticsearch indexes for BM25 sparse search on document chunks.

**Key Features**:
- Creates indexes with proper BM25 similarity scoring
- Configurable index settings (shards, replicas, analyzers)
- Supports multiple connection methods for reliability
- Validates index creation and health

**Usage**:
```bash
# Create index (if it doesn't exist)
python init_elasticsearch.py

# Recreate index (delete existing and create new)
python init_elasticsearch.py --recreate

# Custom settings
python init_elasticsearch.py --url http://localhost:9200 --index my_chunks
```

**Options**:
- `--url`: Elasticsearch server URL (default: http://localhost:9200)
- `--index`: Index name (default: chunks)
- `--recreate`: Delete existing index and create new one

**Index Configuration**:
- **Similarity**: BM25 with k1=1.2, b=0.75
- **Analyzer**: Standard analyzer for tokenization
- **Fields**: chunk_text, document_id, filename, metadata, timestamps
- **Shards**: 1 (single-node setup)
- **Replicas**: 0 (single-node setup)

---

### 2. `clean_elasticsearch.py`
**Purpose**: Clean and manage documents within existing Elasticsearch indexes.

**Key Features**:
- Delete all documents from index
- Selective deletion by document_id, filename, or source_path
- Index statistics and document listing
- Safe operations with confirmation prompts

**Usage**:
```bash
# Show index statistics
python clean_elasticsearch.py --stats

# List documents (first 20)
python clean_elasticsearch.py --list

# Delete all documents (with confirmation)
python clean_elasticsearch.py --delete-all

# Delete all documents (no confirmation)
python clean_elasticsearch.py --delete-all-yes

# Delete specific document by UUID
python clean_elasticsearch.py --delete-by-doc-id "123e4567-e89b-12d3-a456-426614174000"

# Delete by filename
python clean_elasticsearch.py --delete-by-filename "document.pdf"

# Delete by source path
python clean_elasticsearch.py --delete-by-path "/data/documents/"
```

**Options**:
- `--stats`: Show index statistics
- `--list`: List documents in index (use --list-limit to change count)
- `--delete-all`: Delete all documents (requires confirmation)
- `--delete-all-yes`: Delete all documents without confirmation
- `--delete-by-doc-id`: Delete chunks for specific document UUID
- `--delete-by-filename`: Delete chunks for documents with specific filename
- `--delete-by-path`: Delete chunks for documents with specific source path
- `--url`: Elasticsearch server URL (default: http://localhost:9200)
- `--index`: Index name (default: chunks)

---

## Typical Workflow

### Initial Setup
```bash
# 1. Create the index
python init_elasticsearch.py

# 2. Verify it's ready
python clean_elasticsearch.py --stats
```

### During Development/Testing
```bash
# Clean all test data
python clean_elasticsearch.py --delete-all-yes

# Or clean specific documents
python clean_elasticsearch.py --delete-by-filename "test.pdf"
```

### Complete Reset
```bash
# Delete and recreate the entire index
python init_elasticsearch.py --recreate
```

---

## Environment Variables

Both scripts support environment variable overrides:
- `ELASTICSEARCH_URL`: Override default Elasticsearch URL
- `ELASTICSEARCH_INDEX`: Override default index name

Example:
```bash
export ELASTICSEARCH_URL="http://my-cluster:9200"
export ELASTICSEARCH_INDEX="production_chunks"
python init_elasticsearch.py
```

---

## Index Schema

The Elasticsearch index contains the following fields:

### Primary Fields
- `chunk_id`: Unique identifier for each text chunk (keyword)
- `document_id`: UUID of the parent document (keyword)
- `chunk_text`: Full text content for BM25 search (text, analyzed)
- `filename`: Original filename (text with keyword subfield)
- `document_type`: Type of document (keyword)
- `source_path`: Original file path (keyword)

### Metadata Fields
- `metadata`: Nested object containing:
  - `title`: Document title
  - `tags`: Array of tags
  - `author`: Document author
  - `version`: Document version
  - `page_number`: Page number (for PDFs)
  - `section`: Document section
  - `chunk_index`: Index of chunk within document

### Timestamp Fields
- `created_at`: When chunk was indexed
- `updated_at`: When chunk was last updated

---

## Troubleshooting

### Connection Issues
If you see connection errors:
1. Verify Elasticsearch is running: `curl http://localhost:9200/_cluster/health`
2. Check firewall settings
3. Try different connection methods (script handles this automatically)

### Index Already Exists
When running init without `--recreate`, the script will report the index exists. Use `--recreate` to delete and recreate.

### Permission Errors
Ensure your Elasticsearch user has permissions to:
- Create/delete indexes
- Perform search and delete operations
- Access cluster health information

---

## Integration with Pipeline

These scripts are used by the main pipeline operations:

- **Clean Slate** (`clean_slate.py`): Uses clean_elasticsearch.py to remove all documents
- **Reconstruction** (`reconstruct.py`): Uses init_elasticsearch.py to recreate the index structure

For automated operations in the full pipeline, use the main scripts in `../pipeline_ops/`.
