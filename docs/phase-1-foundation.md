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

#### Sparse Index (BM25)

- **Technology Choice**:
  - **Option A**: Elasticsearch (production-ready, more setup)
  - **Option B**: `rank-bm25` Python library (simpler, good for MVP)
  - **Option C**: `whoosh` (pure Python, lightweight)
- **Indexed Fields**:
  - Chunk text
  - Document metadata (title, tags)
  - Filename

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
  - **Option B**: `sentence-transformers/all-mpnet-base-v2` (768 dim, better quality)
  - **Option C**: `intfloat/e5-base-v2` (768 dim, state-of-the-art)
- **Service Type**:
  - Start as Python function/library calls
  - Later refactor to REST API (Phase 2)

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
- **Qdrant** or **ChromaDB** (vector DB)
- **Elasticsearch** or **rank-bm25** (BM25)
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
- [ ] Configure Docker Compose services (Qdrant, Elasticsearch, MinIO)
- [ ] Start Docker services: `docker-compose up -d`
- [ ] Create database tables in Supabase SQL Editor (see schema above)
- [ ] Initialize Qdrant collections
- [ ] Initialize Elasticsearch index

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
