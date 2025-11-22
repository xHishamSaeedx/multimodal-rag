# Phase 1: Foundation & MVP - Text-Only Hybrid RAG

## Overview

Phase 1 establishes the foundational infrastructure and implements a minimal viable product (MVP) that supports **text-only hybrid retrieval** with basic answer generation. This phase focuses on building the core data pipeline, storage systems, and retrieval mechanisms that will serve as the foundation for all subsequent multimodal capabilities.

**Goal**: Build a working end-to-end RAG system that can ingest text documents, index them using hybrid search (BM25 + vector), retrieve relevant chunks, and generate answers with citations.

**Timeline Estimate**: 4-6 weeks for a small team

---

## Phase 1 Objectives

1. ✅ Set up core storage infrastructure (raw data lake, document store, vector DB, sparse index)
2. ✅ Build basic document ingestion pipeline for text extraction
3. ✅ Implement text embedding service
4. ✅ Create hybrid retrieval pipeline (BM25 + dense vectors)
5. ✅ Build simple query processing and answer generation
6. ✅ Create basic API endpoint for queries
7. ✅ Implement minimal monitoring and logging

---

## Architecture Components (Phase 1 Scope)

### 1.1 Storage Infrastructure

#### Raw Data Lake

- **Technology Choice**: Start with local filesystem or MinIO (S3-compatible) for development
- **Structure**: `raw_documents/{source}/{document_type}/{filename}`
- **Supported Formats**: PDF, DOCX, TXT, MD (text-only for Phase 1)
- **Requirements**:
  - Version tracking (simple file naming with timestamps)
  - Basic access control (read/write permissions)

#### Processed Document Store

- **Technology Choice**: **Supabase** (PostgreSQL-based with JSONB support)

  - Supabase provides PostgreSQL with additional features (REST API, Auth, Real-time)
  - Full JSONB support for flexible metadata storage
  - Auto-generated REST API for easy integration
  - Built-in authentication ready for Phase 2+
  - Free tier available at [supabase.com](https://supabase.com)

- **Setup Instructions**:

  1. Create a new project at [app.supabase.com](https://app.supabase.com)
  2. Go to **Project Settings > API** to get:
     - Project URL (e.g., `https://xxxxx.supabase.co`)
     - `anon` key (public key, safe for client-side)
     - `service_role` key (secret, server-side only - keep secure!)
  3. Add these credentials to your `.env` file (see `.env.example`):
     - `SUPABASE_URL`
     - `SUPABASE_ANON_KEY`
     - `SUPABASE_SERVICE_ROLE_KEY`

- **Schema**:

  ```sql
  documents (
    id UUID PRIMARY KEY,
    source_path TEXT,
    filename TEXT,
    document_type TEXT,
    extracted_text TEXT,
    metadata JSONB,  -- author, tags, version, upload_date
    created_at TIMESTAMP,
    updated_at TIMESTAMP
  )

  chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    chunk_text TEXT,
    chunk_index INTEGER,
    chunk_type TEXT DEFAULT 'text',
    metadata JSONB,  -- page_number, section, etc.
    created_at TIMESTAMP
  )
  ```

#### Vector Database

- **Technology Choice**: Qdrant (recommended) or ChromaDB for simplicity
- **Collections**:

  - `text_chunks`: Stores embeddings for text chunks
  - Each vector: 384-768 dimensions (depending on embedding model)
  - Payload includes: `chunk_id`, `document_id`, `text`, `metadata`

- **Setup Instructions**:

  1. Start Qdrant service (if not already running):

     ```bash
     docker-compose up -d qdrant
     ```

  2. Verify Qdrant is accessible:

     ```bash
     curl http://localhost:6333/health
     ```

     Or visit the dashboard: http://localhost:6333/dashboard

  3. Install Python client:

     ```bash
     pip install qdrant-client
     ```

  4. Initialize the collection:
     ```bash
     python scripts/init_qdrant.py
     ```
     - Default: Creates `text_chunks` collection with **768 dimensions** (for e5-base-v2/all-mpnet-base-v2 - best quality)
     - For 384 dimensions (faster): `python scripts/init_qdrant.py --vector-size 384`
     - See `scripts/README.md` for more options
     - **Performance**: 768 dimensions supports sub-second retrieval (< 200-300ms)

  The collection will be ready to store embeddings after initialization.

#### Sparse Index (BM25)

- **Technology Choice**: **Elasticsearch** (production-ready, enterprise-grade)

  - Elasticsearch provides industry-standard BM25 search with advanced features
  - Production-tested at scale with excellent performance
  - Rich query capabilities (full-text, filters, aggregations)
  - Persistent index storage with high availability options
  - Industry-recognized technology that demonstrates enterprise system design

- **Indexed Fields**:
  - Chunk text (full-text search with BM25 scoring)
  - Document metadata (title, tags, author, version)
  - Filename and document ID
  - Timestamps (created_at, updated_at)
- **Setup Instructions** (Step-by-Step):

  **Step 1**: Start Elasticsearch service

  Elasticsearch is already configured in `docker-compose.yml`. Start it with:

  ```bash
  docker-compose up -d elasticsearch
  ```

  ⚠️ **Note**: Elasticsearch may take 30-60 seconds to fully start on first run (it needs to initialize). The container will be "running" but Elasticsearch itself needs time to become ready.

  **Step 2**: Verify Elasticsearch is running

  Wait a moment, then check if Elasticsearch is ready:

  ```bash
  curl http://localhost:9200/_cluster/health
  ```

  Expected response:

  ```json
  {
    "cluster_name": "docker-cluster",
    "status": "yellow",
    "timed_out": false,
    ...
  }
  ```

  - `"status": "yellow"` is normal for single-node setup (means primary shards are allocated)
  - `"status": "green"` is ideal but not required for single-node
  - `"status": "red"` means there's a problem

  **Alternative verification** (if curl doesn't work):

  ```bash
  # Check if container is running
  docker ps | grep elasticsearch

  # Check container logs
  docker logs multimodal-rag-elasticsearch
  ```

  Or visit in browser: http://localhost:9200 (should return cluster info)

  **Step 3**: Install Python Elasticsearch client

  ```bash
  pip install "elasticsearch>=8.0.0,<9.0.0"
  ```

  **Important**: Use Elasticsearch client version 8.x to match the Elasticsearch 8.11.0 server.
  Client version 9.x is incompatible (uses protocol version 9, server only supports 7-8).

  Or add to your `requirements.txt`:

  ```
  elasticsearch>=8.0.0
  ```

  **Step 4**: Initialize the Elasticsearch index

  Run the initialization script to create the `text_chunks` index:

  ```bash
  python scripts/init_elasticsearch.py
  ```

  This will:

  - Create the `text_chunks` index with proper mappings
  - Configure BM25 similarity scoring (default, but explicit)
  - Set up standard analyzer for tokenization
  - Define all metadata fields for filtering
  - Verify the index is ready

  **Expected output**:

  ```
  Connecting to Elasticsearch at http://localhost:9200...
  ✓ Connected to Elasticsearch successfully!
  Creating index 'text_chunks' with BM25 similarity...
  ✓ Index 'text_chunks' created successfully!
  ✓ Index 'text_chunks' is ready.
    - Status: yellow
    - Shards: 1
    - Similarity: BM25
    - Analyzer: standard (for BM25)
  ```

  **Step 5**: (Optional) Verify the index was created

  ```bash
  # Check index exists
  curl http://localhost:9200/text_chunks

  # Or test with Python
  python -c "from elasticsearch import Elasticsearch; es = Elasticsearch([{'host': 'localhost', 'port': 9200}]); print('Index exists:', es.indices.exists(index='text_chunks'))"
  ```

  **Index Configuration Summary**:

  - Index name: `text_chunks`
  - Similarity: BM25 (k1=1.2, b=0.75)
  - Analyzer: `standard` (for tokenization)
  - Shards: 1 (optimized for single-node)
  - Replicas: 0 (single-node setup)
  - Indexed fields:
    - `chunk_text`: Full-text search (primary BM25 field)
    - `chunk_id`, `document_id`: Keywords (exact match, filtering)
    - `filename`, `document_type`: Searchable and filterable
    - `metadata`: Object with nested fields (title, tags, author, etc.)
    - `created_at`, `updated_at`: Date fields

- **Performance**:
  - Query latency: 10-50ms for typical searches (1k-100k documents)
  - Indexing speed: 1000-5000 documents/second (depends on document size)
  - Memory: ~512MB allocated (configurable in docker-compose.yml)

---

### 1.2 Document Ingestion Pipeline

#### Text Extraction Service

- **Libraries**:
  - PDF: `PyMuPDF` (fitz) or `pdfplumber`
  - DOCX: `python-docx`
  - TXT/MD: Direct read
- **Output**: Clean text with basic metadata

#### Chunking Strategy

- **Approach**: Semantic chunking with overlap
- **Method**:
  - Split by paragraphs/sentences
  - Target chunk size: 500-1000 tokens
  - Overlap: 100-200 tokens
  - Preserve document structure (headers, sections)
- **Library**: `langchain.text_splitter` or custom implementation

#### Pipeline Flow

```
Raw Document → Text Extraction → Chunking → Store in Document DB →
Generate Embeddings → Store in Vector DB → Index in BM25 → Done
```

---

### 1.3 Embedding Service

#### Text Embedding Model

- **Phase 1 Options**:
  - **Option A**: `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast, good quality)
    - Embedding time: ~10-30ms per query (CPU), ~2-5ms (GPU)
    - Best for: Rapid prototyping, large-scale deployments
  - **Option B**: `sentence-transformers/all-mpnet-base-v2` (768 dim, better quality)
    - Embedding time: ~30-80ms per query (CPU), ~5-15ms (GPU)
    - Best for: Balance of quality and speed
  - **Option C**: `intfloat/e5-base-v2` (768 dim, state-of-the-art) ⭐ **Recommended**
    - Embedding time: ~40-100ms per query (CPU), ~8-20ms (GPU)
    - Best for: Maximum accuracy, production-ready
    - **Performance**: Supports sub-second retrieval (< 200-300ms for vector search + BM25)
- **Service Type**:
  - Start as Python function/library calls
  - Later refactor to REST API (Phase 2)

#### Performance Notes for Sub-Second Retrievals

**With 768 dimensions (e5-base-v2)**, typical retrieval latency breakdown:

- Query embedding generation: 40-100ms (CPU) / 8-20ms (GPU)
- Vector search in Qdrant: 10-50ms (even with 768 dim - very fast!)
- BM25 search (Elasticsearch): 10-30ms
- Result merging & deduplication: 5-10ms
- **Total retrieval time: ~65-190ms** ✅ Well under 1 second

**Tips for sub-second retrieval:**

- Use GPU for embedding generation (10-20x faster)
- Qdrant handles 768-dim vectors efficiently (minimal overhead vs 384)
- Parallel retrieval (vector + BM25 simultaneously) reduces latency
- Consider embedding caching for repeated queries

#### Embedding Generation

- Batch processing for efficiency
- Store embeddings with chunk metadata
- Handle failures gracefully (retry logic)

---

### 1.4 Hybrid Retrieval Pipeline

#### Retrieval Flow

1. **Query Processing**:
   - Clean and normalize query
   - Generate query embedding
2. **Parallel Retrieval**:
   - **Sparse (BM25)**: Retrieve top-k chunks (e.g., k=50)
   - **Dense (Vector)**: Retrieve top-k chunks (e.g., k=50)
3. **Merge & Deduplicate**:
   - Combine results from both indexes
   - Remove duplicate chunks
   - Score normalization (optional for Phase 1)
4. **Reranking** (Optional for Phase 1):

   - Simple cross-encoder or keep as-is
   - Select top-N chunks (e.g., N=10)

5. **Metadata Filtering**:
   - Apply basic filters (document type, date range)
   - Filter by source if needed

---

### 1.5 Answer Generation

#### LLM Integration

- **Options**:
  - OpenAI GPT-4/GPT-3.5-turbo
  - Anthropic Claude
  - Local: Ollama with Llama 2/Mistral
- **Prompt Template**:

  ```
  Context:
  {retrieved_chunks}

  Question: {user_query}

  Instructions:
  - Answer based only on the provided context
  - Cite sources using [Document: filename, Chunk: N]
  - If information is not in context, say "I don't have that information"
  - Be concise and accurate
  ```

#### Hallucination Guardrails (Basic)

- Check that answer is grounded in retrieved chunks
- Flag if answer contains information not in context
- Simple keyword matching (Phase 1) → Advanced validation (later phases)

---

### 1.6 API Layer

#### REST API Endpoints

- **POST `/api/v1/ingest`**: Upload and process document
- **POST `/api/v1/query`**: Submit query, get answer
- **GET `/api/v1/documents`**: List documents
- **GET `/api/v1/health`**: Health check

#### Technology Stack

- **Framework**: FastAPI (Python) or Flask
- **Authentication**: Basic API keys (Phase 1) → OAuth2 (later)
- **Response Format**: JSON with answer, citations, sources

---

## Implementation Steps

### Week 1: Infrastructure Setup

1. **Day 1-2**: Set up development environment

   - Python virtual environment
   - Set up Supabase project (cloud-based PostgreSQL)
   - Docker Compose for services (Qdrant, Elasticsearch, MinIO)
   - Project structure
   - Configure environment variables (.env file)

2. **Day 3-4**: Database schemas

   - Create Supabase tables (via SQL Editor or migrations)
   - Set up Qdrant collections
   - Set up BM25 index structure

3. **Day 5**: Basic storage utilities
   - Document upload handler
   - Metadata extraction
   - File organization

### Week 2: Ingestion Pipeline

1. **Day 1-2**: Text extraction

   - PDF extraction
   - DOCX extraction
   - Error handling

2. **Day 3**: Chunking implementation

   - Semantic chunking logic
   - Overlap handling
   - Metadata preservation

3. **Day 4-5**: End-to-end ingestion
   - Pipeline orchestration
   - Database storage
   - Testing with sample documents

### Week 3: Embedding & Indexing

1. **Day 1-2**: Embedding service

   - Model selection and setup
   - Batch embedding generation
   - Vector storage in Qdrant

2. **Day 3**: BM25 indexing

   - Index creation
   - Document indexing
   - Query testing

3. **Day 4-5**: Hybrid retrieval
   - Sparse retrieval
   - Dense retrieval
   - Merge logic
   - Testing retrieval quality

### Week 4: Query & Answer Generation

1. **Day 1-2**: Query processing

   - Query normalization
   - Embedding generation
   - Retrieval orchestration

2. **Day 3**: Answer generation

   - LLM integration
   - Prompt engineering
   - Citation extraction

3. **Day 4**: API development

   - FastAPI endpoints
   - Request/response handling
   - Error handling

4. **Day 5**: Integration testing
   - End-to-end tests
   - Performance testing
   - Bug fixes

### Week 5-6: Polish & Documentation

1. **Week 5**:

   - Monitoring and logging
   - Error handling improvements
   - Performance optimization
   - Basic UI (optional)

2. **Week 6**:
   - Documentation
   - Deployment guide
   - Testing suite
   - Demo preparation

---

## Technology Stack (Phase 1)

### Core Languages & Frameworks

- **Python 3.10+**
- **FastAPI** (API framework)
- **Pydantic** (data validation)

### Storage

- **Supabase** (document store - PostgreSQL-based with JSONB)
- **Qdrant** (vector DB)
- **Elasticsearch** (BM25 sparse index)
- **MinIO** or local filesystem (raw storage)

### ML/AI Libraries

- **sentence-transformers** (embeddings)
- **langchain** (optional, for chunking utilities)
- **OpenAI/Anthropic SDK** (LLM)

### Infrastructure

- **Docker** & **Docker Compose** (local development)
- **SQLAlchemy** (database ORM)
- **Alembic** (database migrations)

### Testing & Monitoring

- **pytest** (testing)
- **logging** (Python logging)
- **Prometheus** (optional, metrics)

---

## Success Criteria

### Functional Requirements

- ✅ Can ingest PDF, DOCX, TXT documents
- ✅ Documents are chunked and stored correctly
- ✅ Hybrid retrieval returns relevant chunks
- ✅ Answers are generated with citations
- ✅ API endpoints work end-to-end

### Performance Requirements

- Ingestion: Process 100 documents in < 10 minutes
- Query latency: < 2 seconds for simple queries
- Retrieval: Top-10 relevant chunks for 80%+ of test queries

### Quality Requirements

- Answer accuracy: Answers are grounded in retrieved chunks
- Citation quality: Citations point to correct sources
- Error handling: Graceful failures, no data loss

---

## Limitations & Future Phases

### What Phase 1 Does NOT Include

- ❌ Multimodal support (images, tables, diagrams)
- ❌ Knowledge graph
- ❌ Advanced reranking
- ❌ Complex query routing
- ❌ MLOps pipelines (experiment tracking, model registry)
- ❌ Production-grade monitoring
- ❌ User authentication/authorization
- ❌ Web UI (optional, can be added)

### What Comes Next (Phase 2)

- Table extraction and embedding
- Image extraction and CLIP embeddings
- Multimodal fusion layer
- Advanced reranking
- Query router for modality detection
- Basic MLOps setup

---

## Getting Started Checklist

### Prerequisites

- [ ] Python 3.10+ installed
- [ ] Docker & Docker Compose installed
- [ ] Git repository initialized
- [ ] Development environment set up

### Initial Setup

- [ ] Clone/create project structure
- [ ] Set up virtual environment
- [ ] Install dependencies
- [ ] Create Supabase project at [app.supabase.com](https://app.supabase.com)
- [ ] Get Supabase credentials (URL, anon key, service role key, DB password)
- [ ] Copy `.env.example` to `.env` and fill in credentials
- [ ] Start Docker services: `docker-compose up -d` (Qdrant, Elasticsearch, MinIO are already configured)
- [ ] Verify services are running:
  - Qdrant: `curl http://localhost:6333/health`
  - Elasticsearch: `curl http://localhost:9200/_cluster/health`
  - MinIO: http://localhost:9090 (console UI)

### Service Access URLs

When services are running via Docker Compose, access them at:

**Qdrant (Vector Database)**

- REST API: `http://localhost:6333`
- Dashboard/Web UI: `http://localhost:6333/dashboard`
- gRPC: `localhost:6334` (gRPC endpoint, not HTTP)

**Elasticsearch (BM25 Sparse Index)**

- HTTP API: `http://localhost:9200`
- Cluster Health: `http://localhost:9200/_cluster/health`
- Transport: `localhost:9300` (not HTTP)

**MinIO (S3-compatible Storage)**

- S3 API: `http://localhost:9000`
- Console UI: `http://localhost:9090`
  - Default credentials: `admin` / `admin12345`

**Quick Access:**

- Qdrant Dashboard: `http://localhost:6333`
- Elasticsearch: `http://localhost:9200`
- MinIO Console: `http://localhost:9090` (login with admin/admin12345)

All services are on the `rag-network` Docker network and can communicate using their service names (`qdrant`, `elasticsearch`, `minio`).

- [ ] Create database tables in Supabase SQL Editor (see schema above)
- [ ] Install Python dependencies: `pip install qdrant-client elasticsearch`
- [ ] Initialize Qdrant collection: `python scripts/init_qdrant.py`
- [ ] Initialize Elasticsearch index: `python scripts/init_elasticsearch.py`

### First Milestone

- [ ] Can upload a PDF document
- [ ] Document is extracted and chunked
- [ ] Chunks are stored in database
- [ ] Embeddings are generated and stored
- [ ] Can query and get results

---

## Sample Project Structure

```
multimodal-rag/
├── src/
│   ├── ingestion/
│   │   ├── extractors/
│   │   │   ├── pdf_extractor.py
│   │   │   ├── docx_extractor.py
│   │   │   └── text_extractor.py
│   │   ├── chunkers/
│   │   │   └── semantic_chunker.py
│   │   └── pipeline.py
│   ├── storage/
│   │   ├── document_store.py
│   │   ├── vector_store.py
│   │   └── sparse_index.py
│   ├── embeddings/
│   │   └── text_embedder.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── generation/
│   │   └── answer_generator.py
│   └── api/
│       ├── main.py
│       ├── routes/
│       │   ├── ingest.py
│       │   └── query.py
│       └── models.py
├── tests/
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Next Steps

Once Phase 1 is complete and validated, proceed to:

- **Phase 2**: Multimodal Support (tables, images)
- **Phase 3**: Knowledge Graph Integration
- **Phase 4**: Advanced Routing & Reasoning
- **Phase 5**: Full MLOps Pipeline

---

## Resources & References

- [Supabase Documentation](https://supabase.com/docs)
- [Supabase PostgreSQL Guide](https://supabase.com/docs/guides/database)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
