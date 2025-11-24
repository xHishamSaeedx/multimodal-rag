# Phase 2: Multimodal Support & Advanced Retrieval

## Overview

Phase 2 extends the text-only RAG system from Phase 1 to support **multimodal content** (tables, images, diagrams) and introduces advanced retrieval capabilities including query routing, multimodal fusion, and improved reranking. This phase transforms the system from a text-focused tool into a comprehensive multimodal knowledge retrieval platform.

**Goal**: Enable the system to ingest, index, retrieve, and reason over tables and images in addition to text, with intelligent query routing that automatically selects the appropriate retrieval strategy based on query intent.

**Timeline Estimate**: 6-8 weeks for a small team

**Prerequisites**: Phase 1 must be complete and validated before starting Phase 2.

---

## Phase 2 Objectives

1. ✅ Extend ingestion pipeline to extract tables and images from documents
2. ✅ Implement table embedding service (structured data embeddings)
3. ✅ Implement image embedding service (CLIP/SigLIP)
4. ✅ Extend storage schemas to support multimodal chunks
5. ✅ Create multimodal vector collections in Qdrant
6. ✅ Build query router for modality detection
7. ✅ Implement multimodal fusion layer
8. ✅ Add advanced reranking (cross-encoder or LLM-based)
9. ✅ Set up basic MLOps infrastructure (experiment tracking, model registry)
10. ✅ Extend API endpoints for multimodal queries

---

## Architecture Components (Phase 2 Scope)

### 2.1 Extended Storage Infrastructure

#### Updated Document Store Schema

Extend the Supabase schema to support multimodal content:

```sql
-- Extend existing chunks table
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT DEFAULT 'text';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_data JSONB;  -- For table chunks
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_path TEXT;   -- Path to image file
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_caption TEXT; -- Optional caption
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_type TEXT DEFAULT 'text'; -- text, table, image

-- New table for storing raw images
CREATE TABLE IF NOT EXISTS images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id),
  chunk_id UUID REFERENCES chunks(id),
  image_path TEXT NOT NULL,
  image_type TEXT,  -- diagram, chart, photo, screenshot
  extracted_text TEXT,  -- OCR text if applicable
  caption TEXT,
  metadata JSONB,  -- dimensions, format, page_number
  created_at TIMESTAMP DEFAULT NOW()
);

-- New table for storing table structures
CREATE TABLE IF NOT EXISTS tables (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id),
  chunk_id UUID REFERENCES chunks(id),
  table_data JSONB NOT NULL,  -- Structured table data
  table_markdown TEXT,  -- Markdown representation
  table_text TEXT,  -- Flattened text representation
  metadata JSONB,  -- row_count, col_count, headers
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### Extended Vector Database (Qdrant)

Create additional collections for multimodal embeddings:

**Collections to Add:**

1. **`table_chunks`** (768 dimensions)

   - Stores embeddings for table content
   - Payload: `chunk_id`, `document_id`, `table_data`, `table_markdown`, `metadata`

2. **`image_chunks`** (512 dimensions for CLIP, 768 for SigLIP)
   - Stores image embeddings
   - Payload: `chunk_id`, `document_id`, `image_path`, `caption`, `image_type`, `metadata`

**Setup Instructions:**

1. Initialize table collection:

   ```bash
   python scripts/init_qdrant.py --collection table_chunks --vector-size 768
   ```

2. Initialize image collection (CLIP - 512 dim):

   ```bash
   python scripts/init_qdrant.py --collection image_chunks --vector-size 512
   ```

   Or for SigLIP (768 dim, better quality):

   ```bash
   python scripts/init_qdrant.py --collection image_chunks --vector-size 768
   ```

#### Extended Sparse Index (Elasticsearch)

Update the `text_chunks` index to support multimodal content:

```json
{
  "mappings": {
    "properties": {
      "chunk_text": { "type": "text", "analyzer": "standard" },
      "chunk_type": { "type": "keyword" }, // text, table, image, mixed
      "table_markdown": { "type": "text" }, // Searchable table content
      "image_caption": { "type": "text" }, // Searchable image captions
      "chunk_id": { "type": "keyword" },
      "document_id": { "type": "keyword" },
      "embedding_type": { "type": "keyword" }, // text, table, image
      "metadata": {
        "type": "object",
        "properties": {
          "image_type": { "type": "keyword" },
          "table_headers": { "type": "text" },
          "row_count": { "type": "integer" },
          "col_count": { "type": "integer" }
        }
      }
    }
  }
}
```

**Setup Instructions:**

1. Update existing index mapping:

   ```bash
   python scripts/update_elasticsearch_mapping.py
   ```

   Or manually via Elasticsearch API:

   ```bash
   curl -X PUT "localhost:9200/text_chunks/_mapping" -H 'Content-Type: application/json' -d @mapping_update.json
   ```

---

### 2.2 Multimodal Ingestion Pipeline

#### Table Extraction Service

**Technology Choices:**

- **PDF Tables**: `camelot-py` (best for structured tables) or `tabula-py` (simpler, good for basic tables)
- **DOCX Tables**: `python-docx` (native support)
- **HTML Tables**: `BeautifulSoup` + `pandas`
- **Excel/CSV**: `pandas` (direct read)

**Libraries to Install:**

```bash
pip install camelot-py[cv] tabula-py python-docx pandas beautifulsoup4
```

**Table Extraction Flow:**

1. **Detection**: Identify table regions in documents
2. **Extraction**: Extract table structure (rows, columns, headers)
3. **Normalization**: Convert to structured format (JSON)
4. **Text Representation**: Generate markdown and flattened text versions
5. **Chunking**: Split large tables into smaller semantic chunks (if needed)

**Table Representation Formats:**

- **JSON**: Structured data for programmatic access

  ```json
  {
    "headers": ["Column 1", "Column 2", "Column 3"],
    "rows": [
      ["Value 1", "Value 2", "Value 3"],
      ["Value 4", "Value 5", "Value 6"]
    ]
  }
  ```

- **Markdown**: Human-readable format for LLM context

  ```markdown
  | Column 1 | Column 2 | Column 3 |
  | -------- | -------- | -------- |
  | Value 1  | Value 2  | Value 3  |
  | Value 4  | Value 5  | Value 6  |
  ```

- **Flattened Text**: For embedding generation
  ```
  Column 1: Value 1, Column 2: Value 2, Column 3: Value 3
  Column 1: Value 4, Column 2: Value 5, Column 3: Value 6
  ```

#### Image Extraction Service

**Technology Choices:**

- **PDF Images**: `PyMuPDF` (fitz) - extracts images with coordinates
- **DOCX Images**: `python-docx` - extracts embedded images
- **OCR for Images**: `pytesseract` or `easyocr` (for text in images)
- **Image Processing**: `PIL/Pillow` for format conversion and preprocessing

**Libraries to Install:**

```bash
pip install PyMuPDF python-docx pytesseract easyocr pillow
```

**Image Extraction Flow:**

1. **Detection**: Identify images in documents (diagrams, charts, photos, screenshots)
2. **Extraction**: Extract image files with metadata (page number, position, dimensions)
3. **OCR (Optional)**: Extract text from images if applicable
4. **Captioning (Optional)**: Generate captions using vision-language models
5. **Storage**: Store images in raw data lake (MinIO/S3) with references in database

**Image Types to Support:**

- **Diagrams**: Architecture diagrams, flowcharts, system designs
- **Charts**: Bar charts, line graphs, pie charts
- **Screenshots**: UI screenshots, code snippets
- **Photos**: Product images, documentation photos

#### Updated Pipeline Flow

```
Raw Document → Multimodal Extraction:
  ├─ Text Extraction (Phase 1)
  ├─ Table Extraction → Table JSON/Markdown → Table Embeddings
  └─ Image Extraction → Image Files → Image Embeddings (CLIP/SigLIP)

All Content → Chunking → Store in Document DB →
  ├─ Text Chunks → Text Embeddings → Vector DB (text_chunks)
  ├─ Table Chunks → Table Embeddings → Vector DB (table_chunks)
  └─ Image Chunks → Image Embeddings → Vector DB (image_chunks)

All Chunks → Index in BM25 (Elasticsearch) → Done
```

---

### 2.3 Multimodal Embedding Services

#### Table Embedding Service

**Approach Options:**

1. **Flattened Text Embedding** (Simpler, Phase 2)

   - Convert table to flattened text: "Column1: Value1, Column2: Value2..."
   - Use same text embedding model (e5-base-v2)
   - Pros: Simple, reuses existing model
   - Cons: May lose structural relationships

2. **Structured Embedding** (Advanced, Phase 2+)
   - Use specialized table embedding models (e.g., TAPAS, TaBERT)
   - Preserves table structure and relationships
   - Pros: Better semantic understanding
   - Cons: Requires additional model, more complex

**Phase 2 Recommendation**: Start with **Flattened Text Embedding** using e5-base-v2, then upgrade to structured embeddings in later phases.

**Implementation:**

- Generate embeddings from table markdown or flattened text
- Store in `table_chunks` collection (768 dimensions)
- Include table metadata in payload for filtering

#### Image Embedding Service

**Model Options:**

1. **CLIP (OpenAI)** ⭐ **Recommended for Phase 2**

   - Model: `openai/clip-vit-base-patch32` or `openai/clip-vit-large-patch14`
   - Dimensions: 512 (base) or 768 (large)
   - Embedding time: ~50-150ms per image (CPU), ~10-30ms (GPU)
   - Best for: General image understanding, diagrams, charts
   - Installation: `pip install transformers torch`

2. **SigLIP (Google)** (Alternative)

   - Model: `google/siglip-base-patch16-224`
   - Dimensions: 768
   - Embedding time: ~60-180ms per image (CPU), ~12-35ms (GPU)
   - Best for: Better zero-shot performance, more recent
   - Installation: `pip install transformers torch`

3. **BLIP-2** (For captioning + embedding)
   - Can generate captions AND embeddings
   - More resource-intensive
   - Best for: When captions are needed

**Phase 2 Recommendation**: Use **CLIP base model** (512 dim) for balance of speed and quality.

**Setup Instructions:**

1. Install dependencies:

   ```bash
   pip install transformers torch pillow
   ```

2. Initialize image embedding service:

   ```python
   from transformers import CLIPProcessor, CLIPModel

   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
   ```

3. Generate embeddings:

   ```python
   from PIL import Image

   image = Image.open(image_path)
   inputs = processor(images=image, return_tensors="pt")
   image_embedding = model.get_image_features(**inputs)
   ```

**Performance Notes:**

- **Batch Processing**: Process multiple images in batches for efficiency
- **GPU Acceleration**: 10-20x faster with GPU (highly recommended)
- **Caching**: Cache embeddings for unchanged images
- **Storage**: Store images in MinIO/S3, embeddings in Qdrant

#### Embedding Service Architecture

**Phase 2**: Start with Python functions/library calls (same as Phase 1 text embeddings)

**Future (Phase 3+)**: Refactor to REST API microservices for:

- Independent scaling
- Model versioning
- A/B testing
- Better resource management

---

### 2.4 Query Router & Modality Detection

#### Query Router Service

**Purpose**: Analyze user queries to determine which modalities (text, table, image) are needed and route to appropriate retrieval paths.

**Detection Methods:**

1. **Keyword-Based Detection** (Phase 2 - Simple)

   - Keywords for tables: "compare", "table", "data", "statistics", "rate", "limit", "pricing"
   - Keywords for images: "diagram", "chart", "image", "show me", "visual", "architecture", "screenshot"
   - Keywords for text: default fallback

2. **LLM-Based Classification** (Phase 2 - Advanced)

   - Use lightweight LLM (e.g., GPT-3.5-turbo) to classify query intent
   - More accurate but adds latency (~100-200ms)
   - Can be cached for common queries

3. **Hybrid Approach** (Phase 2 - Recommended)
   - Fast keyword check first
   - LLM classification for ambiguous queries
   - Confidence scoring for each modality

**Router Output:**

```python
{
  "modalities": {
    "text": 0.9,      # Confidence score
    "table": 0.7,
    "image": 0.2
  },
  "primary_modality": "table",
  "retrieval_strategy": "multimodal",  # text_only, table_focused, image_focused, multimodal
  "query_rewrite": "API rate limits comparison free vs premium users"
}
```

**Implementation:**

- Create `QueryRouter` service in `app/services/routing/`
- Integrate with query endpoint
- Cache routing decisions for performance

---

### 2.5 Multimodal Hybrid Retrieval Pipeline

#### Extended Retrieval Flow

1. **Query Processing**:

   - Clean and normalize query
   - **Query Router**: Determine needed modalities
   - Generate query embeddings (text, and optionally image if query includes image)

2. **Parallel Multimodal Retrieval**:

   - **Sparse (BM25)**: Retrieve top-k chunks across all modalities (k=50)
   - **Dense (Vector) - Text**: Retrieve top-k text chunks (k=50)
   - **Dense (Vector) - Tables**: Retrieve top-k table chunks (k=30, if tables needed)
   - **Dense (Vector) - Images**: Retrieve top-k image chunks (k=20, if images needed)

3. **Modality-Specific Filtering**:

   - Filter by `chunk_type` based on router decision
   - Apply metadata filters (document type, date range)

4. **Merge & Deduplicate**:

   - Combine results from all retrieval sources
   - Remove duplicate chunks (same `chunk_id`)
   - Normalize scores across modalities (optional for Phase 2)

5. **Advanced Reranking**:

   - Cross-encoder reranker or LLM-based reranking
   - Score chunks considering query intent and modality relevance
   - Select top-N chunks (e.g., N=10-15)

6. **Multimodal Fusion**:
   - Combine text, table, and image chunks into unified context
   - Format tables as markdown
   - Include image references and captions

---

### 2.6 Advanced Reranking

#### Reranking Options

1. **Cross-Encoder Reranker** ⭐ **Recommended for Phase 2**

   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast) or `cross-encoder/ms-marco-MiniLM-L-12-v2` (better quality)
   - Input: Query + Chunk text (concatenated)
   - Output: Relevance score (0-1)
   - Latency: ~10-30ms per chunk (CPU), ~2-5ms (GPU)
   - Installation: `pip install sentence-transformers`

2. **LLM-Based Reranking** (Alternative)

   - Use GPT-3.5-turbo or Claude to score chunks
   - More flexible but slower (~200-500ms per chunk)
   - Better for complex reasoning about relevance

3. **Hybrid Scoring** (Advanced)
   - Combine cross-encoder scores with BM25 and vector similarity
   - Weighted fusion: `final_score = 0.4 * cross_encoder + 0.3 * vector + 0.3 * bm25`

**Phase 2 Recommendation**: Start with **Cross-Encoder** for speed and quality balance.

**Implementation:**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Rerank chunks
pairs = [(query, chunk.text) for chunk in retrieved_chunks]
scores = reranker.predict(pairs)

# Sort by scores and select top-N
reranked_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:N]
```

---

### 2.7 Multimodal Fusion Layer

#### Fusion Strategy

**Purpose**: Combine retrieved text, table, and image chunks into a unified context for the LLM.

**Fusion Process:**

1. **Format Text Chunks**: Include as-is with citations
2. **Format Table Chunks**: Convert to markdown tables with context
3. **Format Image Chunks**: Include image references, captions, and descriptions
4. **Order by Relevance**: Use reranker scores to order chunks
5. **Add Modality Tags**: Tag each chunk with its type for LLM awareness

**Fused Context Template:**

```
[Text Chunk 1]
Source: document.pdf, Page 3
{text content}

[Table Chunk 1]
Source: document.pdf, Page 5
{table markdown}

[Image Chunk 1]
Source: document.pdf, Page 7
Image: architecture_diagram.png
Caption: Payment pipeline architecture v3
{image description if available}

[Text Chunk 2]
...
```

**Implementation:**

- Create `MultimodalFusion` service in `app/services/fusion/`
- Integrate with answer generation pipeline
- Handle edge cases (no tables/images found, etc.)

---

### 2.8 Extended Answer Generation

#### Enhanced Prompt Template

```
Context (Multimodal):
{text_chunks}
{table_chunks_as_markdown}
{image_chunks_with_captions}

Question: {user_query}

Instructions:
- Answer based only on the provided context
- Use tables to provide specific data and comparisons
- Reference images/diagrams when relevant
- Cite sources using [Document: filename, Page: N, Type: text/table/image]
- If information is not in context, say "I don't have that information"
- Be concise and accurate
- For table-based questions, present data clearly (e.g., bullet points, comparisons)
```

#### Enhanced Hallucination Guardrails

- Validate that table data cited in answer matches retrieved tables
- Check that image references are valid
- Flag if answer mentions images/tables not in context
- Verify numerical claims against table data

---

### 2.9 Basic MLOps Infrastructure

#### Experiment Tracking (MLflow)

**Purpose**: Track embedding models, rerankers, and retrieval performance.

**Setup Instructions:**

1. Install MLflow:

   ```bash
   pip install mlflow
   ```

2. Start MLflow server (local):

   ```bash
   mlflow ui --port 5000
   ```

   Or add to docker-compose.yml:

   ```yaml
   mlflow:
     image: python:3.10
     command: mlflow ui --host 0.0.0.0 --port 5000
     volumes:
       - ./mlruns:/mlruns
     ports:
       - "5000:5000"
   ```

3. Log experiments:

   ```python
   import mlflow

   mlflow.set_experiment("multimodal-rag-phase2")

   with mlflow.start_run():
       mlflow.log_param("embedding_model", "e5-base-v2")
       mlflow.log_param("image_model", "clip-vit-base")
       mlflow.log_metric("retrieval_precision", 0.85)
       mlflow.log_metric("retrieval_recall", 0.78)
   ```

**What to Track:**

- Embedding model versions (text, table, image)
- Reranker model versions
- Retrieval metrics (precision, recall, NDCG)
- Query routing accuracy
- Answer quality scores
- Latency metrics

#### Model Registry

**Phase 2**: Use MLflow model registry (basic)

- Register embedding models
- Version control for models
- Promote models to production

**Future (Phase 3+)**: Full model registry with A/B testing, canary deployments

#### Data Versioning (Optional for Phase 2)

- Use DVC or simple versioning for:
  - Raw documents
  - Processed chunks
  - Embeddings snapshots

**Phase 2**: Can be deferred to Phase 3 if needed.

---

### 2.10 Extended API Layer

#### New/Updated Endpoints

**POST `/api/v1/ingest`** (Updated)

- Now supports multimodal extraction
- Returns extraction summary: `{text_chunks: 10, table_chunks: 3, image_chunks: 2}`

**POST `/api/v1/query`** (Updated)

- Enhanced response with multimodal results:
  ```json
  {
    "answer": "...",
    "sources": [
      {
        "type": "text",
        "document": "doc.pdf",
        "page": 3,
        "snippet": "..."
      },
      {
        "type": "table",
        "document": "doc.pdf",
        "page": 5,
        "table_markdown": "| ... |"
      },
      {
        "type": "image",
        "document": "doc.pdf",
        "page": 7,
        "image_url": "/api/v1/images/{image_id}",
        "caption": "..."
      }
    ],
    "modalities_used": ["text", "table"],
    "retrieval_metadata": {
      "text_chunks_retrieved": 8,
      "table_chunks_retrieved": 2,
      "image_chunks_retrieved": 0
    }
  }
  ```

**GET `/api/v1/documents/{doc_id}/tables`** (New)

- List all tables extracted from a document

**GET `/api/v1/documents/{doc_id}/images`** (New)

- List all images extracted from a document

**GET `/api/v1/images/{image_id}`** (New)

- Serve image file (with authentication)

**GET `/api/v1/health`** (Updated)

- Include multimodal service health checks

---

## Implementation Steps

### Week 1: Extended Storage & Schema

1. **Day 1-2**: Database schema updates

   - Add tables for images and table structures
   - Update chunks table with multimodal fields
   - Create migration scripts

2. **Day 3**: Qdrant collections setup

   - Create `table_chunks` collection
   - Create `image_chunks` collection
   - Update initialization scripts

3. **Day 4**: Elasticsearch mapping updates

   - Extend index mapping for multimodal content
   - Update mapping script for multimodal fields
   - Verify mapping structure (testing with real data deferred until after extraction services)

**Note**: Day 5 (Storage utilities updates) is deferred until after Week 3, when we have real table and image chunks from extraction services to test with.

### Week 2: Table Extraction & Embedding

1. **Day 1-2**: Table extraction service

   - Implement PDF table extraction (camelot/tabula)
   - Implement DOCX table extraction
   - Error handling and edge cases

2. **Day 3**: Table processing and normalization

   - Convert tables to JSON, markdown, flattened text
   - Table chunking strategy (if needed)

3. **Day 4**: Table embedding service

   - Implement flattened text embedding
   - Store in `table_chunks` collection
   - Test retrieval

4. **Day 5**: Integration testing
   - End-to-end table extraction and indexing
   - Test with sample documents

### Week 3: Image Extraction & Embedding

1. **Day 1-2**: Image extraction service

   - Implement PDF image extraction (PyMuPDF)
   - Implement DOCX image extraction
   - Image storage in MinIO/S3

2. **Day 3**: Image processing

   - OCR for text in images (optional)
   - Image captioning (optional, can use CLIP)
   - Image type classification

3. **Day 4**: Image embedding service

   - Set up CLIP model
   - Generate image embeddings
   - Store in `image_chunks` collection

4. **Day 5**: Integration testing
   - End-to-end image extraction and indexing
   - Test retrieval with image queries

### Week 3.5: Storage Utilities Updates (Deferred from Week 1)

**Prerequisites**: Week 2 and Week 3 must be complete (table and image extraction services ready)

1. **Update repositories to handle multimodal chunks**:

   - **DocumentRepository**:

     - Add methods to store/retrieve from `images` and `tables` tables
     - Update chunk storage to handle `table_data`, `image_path`, `image_caption` fields
     - Support `chunk_type` and `embedding_type` fields

   - **VectorRepository**:

     - Extend to store embeddings in `table_chunks` and `image_chunks` collections
     - Handle different payload structures per collection type
     - Support table and image embedding storage

   - **SparseRepository**:
     - Update indexing to include multimodal fields (`chunk_type`, `embedding_type`, `table_markdown`, `image_caption`)
     - Index extended metadata fields (image_type, table_headers, row_count, col_count)
     - Support filtering by chunk type and embedding type

2. **Test storage and retrieval** (now possible with real chunks):

   - Test storing text chunks (existing functionality - verify still works)
   - Test storing table chunks (new - using real extracted tables)
   - Test storing image chunks (new - using real extracted images)
   - Verify data integrity across all systems:
     - Supabase (documents, chunks, images, tables tables)
     - Qdrant (text_chunks, table_chunks, image_chunks collections)
     - Elasticsearch (chunks index with multimodal fields)
     - MinIO (image file storage)

3. **Update ingestion pipeline**:

   - Integrate table extraction into pipeline
   - Integrate image extraction into pipeline
   - Ensure end-to-end flow works: extraction → storage → indexing

### Week 4: Query Router & Multimodal Retrieval

1. **Day 1-2**: Query router implementation

   - Keyword-based modality detection
   - LLM-based classification (optional)
   - Router service integration

2. **Day 3**: Multimodal retrieval pipeline

   - Extend hybrid retriever for tables and images
   - Parallel retrieval across modalities
   - Merge and deduplication logic

3. **Day 4**: Advanced reranking

   - Implement cross-encoder reranker
   - Integrate with retrieval pipeline
   - Test reranking quality

4. **Day 5**: Multimodal fusion layer
   - Implement fusion service
   - Format chunks for LLM context
   - Test fusion output

### Week 5: Answer Generation & API Updates

1. **Day 1-2**: Enhanced answer generation

   - Update prompt templates for multimodal context
   - Improve citation handling for tables/images
   - Enhanced hallucination guardrails

2. **Day 3**: API endpoint updates

   - Update query endpoint for multimodal responses
   - Add new endpoints (tables, images)
   - Update request/response schemas

3. **Day 4**: Integration testing

   - End-to-end multimodal queries
   - Test all query types (text, table, image, mixed)
   - Performance testing

4. **Day 5**: Bug fixes and refinements

### Week 6: MLOps & Monitoring

1. **Day 1-2**: MLflow setup

   - Install and configure MLflow
   - Set up experiment tracking
   - Log initial experiments

2. **Day 3**: Model registry

   - Register embedding models
   - Version control setup
   - Model promotion workflow

3. **Day 4**: Monitoring and logging

   - Enhanced logging for multimodal operations
   - Performance metrics collection
   - Error tracking

4. **Day 5**: Documentation and testing
   - Update API documentation
   - Create multimodal query examples
   - Integration test suite

### Week 7-8: Polish & Optimization

1. **Week 7**:

   - Performance optimization
   - Caching strategies
   - Error handling improvements
   - User experience refinements

2. **Week 8**:
   - Comprehensive testing
   - Documentation updates
   - Demo preparation
   - Deployment guide

---

## Technology Stack (Phase 2 Additions)

### New Libraries & Tools

**Table Extraction:**

- `camelot-py[cv]` - PDF table extraction
- `tabula-py` - Alternative PDF table extraction
- `python-docx` - DOCX table extraction
- `pandas` - Table processing

**Image Processing:**

- `PyMuPDF` (fitz) - PDF image extraction
- `PIL/Pillow` - Image processing
- `pytesseract` or `easyocr` - OCR (optional)

**Image Embeddings:**

- `transformers` - CLIP/SigLIP models
- `torch` - PyTorch for model inference

**Reranking:**

- `sentence-transformers` - Cross-encoder rerankers

**MLOps:**

- `mlflow` - Experiment tracking and model registry

**Storage:**

- `boto3` or `minio` - S3-compatible storage for images

---

## Success Criteria

### Functional Requirements

- ✅ Can extract tables from PDF, DOCX documents
- ✅ Can extract images from PDF, DOCX documents
- ✅ Tables are embedded and searchable
- ✅ Images are embedded and searchable
- ✅ Query router correctly identifies needed modalities
- ✅ Multimodal retrieval returns relevant chunks across modalities
- ✅ Reranking improves result quality
- ✅ Answers include citations for text, tables, and images
- ✅ API endpoints support multimodal queries

### Performance Requirements

- Table extraction: Process 100 documents with tables in < 15 minutes
- Image extraction: Process 100 documents with images in < 20 minutes
- Query latency: < 3 seconds for multimodal queries (including image embedding if query includes image)
- Retrieval quality: Top-10 relevant chunks for 85%+ of test queries
- Reranking latency: < 500ms for reranking 50 chunks

### Quality Requirements

- Table extraction accuracy: 90%+ tables correctly extracted
- Image extraction: 95%+ images correctly extracted with metadata
- Query routing accuracy: 85%+ correct modality detection
- Answer accuracy: Answers are grounded in retrieved chunks (text, table, image)
- Citation quality: Citations correctly reference sources with modality type

---

## Limitations & Future Phases

### What Phase 2 Does NOT Include

- ❌ Knowledge graph integration
- ❌ Advanced query rewriting and decomposition
- ❌ Multi-step agentic reasoning
- ❌ Complex table understanding (TAPAS, TaBERT models)
- ❌ Advanced image understanding (object detection, diagram parsing)
- ❌ Full MLOps pipeline (automated retraining, A/B testing)
- ❌ Production-grade monitoring and alerting
- ❌ User authentication/authorization (can be added)
- ❌ Web UI (optional, can be added)

### What Comes Next (Phase 3)

- Knowledge graph integration
- Advanced query routing and rewriting
- Entity extraction and linking
- Relationship extraction
- Graph-based retrieval for multi-hop queries
- Advanced table understanding models
- Diagram parsing and understanding

---

## Getting Started Checklist

### Prerequisites

- [ ] Phase 1 complete and validated
- [ ] All Phase 1 services running (Qdrant, Elasticsearch, Supabase)
- [ ] Python 3.10+ with virtual environment
- [ ] Docker & Docker Compose (for MinIO if using)

### Initial Setup

- [ ] Update database schema in Supabase (run migration scripts)
- [ ] Create new Qdrant collections (`table_chunks`, `image_chunks`)
- [ ] Update Elasticsearch index mapping
- [ ] Install new dependencies: `pip install camelot-py tabula-py transformers torch sentence-transformers mlflow`
- [ ] Set up MinIO/S3 for image storage (if not already done)
- [ ] Configure image embedding model (CLIP)
- [ ] Set up MLflow (local or server)
- [ ] Update environment variables (.env file)

### First Milestone (After Week 3.5)

- [ ] Can extract tables from a PDF document
- [ ] Tables are stored in database and embedded
- [ ] Can extract images from a PDF document
- [ ] Images are stored and embedded
- [ ] Storage utilities updated and tested with real chunks
- [ ] Query router identifies table/image queries
- [ ] Can retrieve relevant tables/images
- [ ] Answers include table/image citations

---

## Sample Project Structure Updates

```
multimodal-rag/
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── embedding/
│   │   │   │   ├── text_embedder.py (Phase 1)
│   │   │   │   ├── table_embedder.py (NEW)
│   │   │   │   └── image_embedder.py (NEW)
│   │   │   ├── ingestion/
│   │   │   │   ├── extractor.py (Phase 1 - text)
│   │   │   │   ├── table_extractor.py (NEW)
│   │   │   │   └── image_extractor.py (NEW)
│   │   │   ├── retrieval/
│   │   │   │   ├── hybrid_retriever.py (UPDATED)
│   │   │   │   ├── reranker.py (UPDATED - advanced)
│   │   │   │   └── multimodal_retriever.py (NEW)
│   │   │   ├── routing/
│   │   │   │   └── query_router.py (NEW)
│   │   │   ├── fusion/
│   │   │   │   └── multimodal_fusion.py (NEW)
│   │   │   └── generation/
│   │   │       └── answer_generator.py (UPDATED)
│   │   ├── repositories/
│   │   │   ├── table_repository.py (NEW)
│   │   │   └── image_repository.py (NEW)
│   │   └── api/
│   │       └── routes/
│   │           ├── query.py (UPDATED)
│   │           ├── tables.py (NEW)
│   │           └── images.py (NEW)
│   └── tools/
│       └── mlflow_tracking.py (NEW)
├── scripts/
│   ├── init_qdrant.py (UPDATED - multimodal collections)
│   ├── update_elasticsearch_mapping.py (NEW)
│   └── test_multimodal_extraction.py (NEW)
├── mlruns/ (MLflow experiments)
└── requirements.txt (UPDATED)
```

---

## Next Steps

Once Phase 2 is complete and validated, proceed to:

- **Phase 3**: Knowledge Graph Integration
- **Phase 4**: Advanced Routing & Reasoning
- **Phase 5**: Full MLOps Pipeline

---

## Resources & References

- [Camelot PDF Table Extraction](https://camelot-py.readthedocs.io/)
- [Tabula PDF Table Extraction](https://tabula.technology/)
- [CLIP Model (OpenAI)](https://openai.com/research/clip)
- [SigLIP Model (Google)](https://arxiv.org/abs/2303.15343)
- [Cross-Encoder Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyMuPDF Image Extraction](https://pymupdf.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)

---

## Quick Reference: Phase 2 Implementation Timeline

### Week 1: Extended Storage & Schema

- **Day 1-2**: Database schema updates (images, tables tables, multimodal fields)
- **Day 3**: Qdrant collections setup (`table_chunks`, `image_chunks`)
- **Day 4**: Elasticsearch mapping updates (multimodal fields)
- **Day 5**: Deferred to Week 3.5 (needs real chunks to test)

### Week 2: Table Extraction & Embedding

- **Day 1-2**: Table extraction service (PDF: camelot/tabula, DOCX: python-docx)
- **Day 3**: Table processing (JSON, markdown, flattened text conversion)
- **Day 4**: Table embedding service (flattened text → embeddings → `table_chunks`)
- **Day 5**: Integration testing (end-to-end table extraction and indexing)

### Week 3: Image Extraction & Embedding

- **Day 1-2**: Image extraction service (PDF: PyMuPDF, DOCX: python-docx, MinIO storage)
- **Day 3**: Image processing (OCR optional, captioning optional, type classification)
- **Day 4**: Image embedding service (CLIP/SigLIP → embeddings → `image_chunks`)
- **Day 5**: Integration testing (end-to-end image extraction and indexing)

### Week 3.5: Storage Utilities Updates (Deferred from Week 1)

- **Update repositories**: DocumentRepository, VectorRepository, SparseRepository for multimodal chunks
- **Test storage**: Text, table, and image chunks with real data
- **Update ingestion pipeline**: Integrate table and image extraction
- **Verify data integrity**: Supabase, Qdrant, Elasticsearch, MinIO

### Week 4: Query Router & Multimodal Retrieval

- **Day 1-2**: Query router (keyword-based + optional LLM-based modality detection)
- **Day 3**: Multimodal retrieval pipeline (parallel retrieval, merge, deduplication)
- **Day 4**: Advanced reranking (cross-encoder reranker integration)
- **Day 5**: Multimodal fusion layer (format chunks for LLM context)

### Week 5: Answer Generation & API Updates

- **Day 1-2**: Enhanced answer generation (multimodal prompts, citations, guardrails)
- **Day 3**: API endpoint updates (multimodal query endpoint, tables/images endpoints, schemas)
- **Day 4**: Integration testing (end-to-end queries, all query types, performance)
- **Day 5**: Bug fixes and refinements

### Week 6: MLOps & Monitoring

- **Day 1-2**: MLflow setup (experiment tracking, logging)
- **Day 3**: Model registry (embedding models, version control, promotion workflow)
- **Day 4**: Monitoring and logging (multimodal operations, metrics, error tracking)
- **Day 5**: Documentation and testing (API docs, examples, test suite)

### Week 7-8: Polish & Optimization

- **Week 7**: Performance optimization, caching, error handling, UX refinements
- **Week 8**: Comprehensive testing, documentation, demo prep, deployment guide

---

## Phase 2 Completion Checklist

### Infrastructure (Week 1)

- [ ] Database schema extended (images, tables tables)
- [ ] Qdrant collections created (`table_chunks`, `image_chunks`)
- [ ] Elasticsearch mapping updated (multimodal fields)

### Extraction Services (Weeks 2-3)

- [ ] Table extraction from PDF/DOCX working
- [ ] Image extraction from PDF/DOCX working
- [ ] Tables embedded and stored in `table_chunks`
- [ ] Images embedded and stored in `image_chunks`

### Storage Layer (Week 3.5)

- [ ] Repositories updated for multimodal chunks
- [ ] Ingestion pipeline integrated with extraction services
- [ ] All systems tested with real chunks

### Retrieval System (Week 4)

- [ ] Query router identifies modalities
- [ ] Multimodal retrieval pipeline working
- [ ] Reranking improves results
- [ ] Fusion layer formats chunks for LLM

### User Interface (Week 5)

- [ ] API endpoints support multimodal queries
- [ ] Answer generation handles tables/images
- [ ] Citations include modality types
- [ ] End-to-end testing complete

### Production Readiness (Week 6)

- [ ] MLflow tracking set up
- [ ] Model registry configured
- [ ] Monitoring and logging enhanced
- [ ] Documentation complete

### Final Polish (Weeks 7-8)

- [ ] Performance optimized
- [ ] Comprehensive testing done
- [ ] Deployment guide ready
- [ ] Demo prepared

---

# Next Phases

Phase 3: Knowledge Graph Integration (optional but valuable)
Phase 4: Advanced Routing & Reasoning (important for complex queries)
Phase 5: Full MLOps Pipeline (critical for production)
