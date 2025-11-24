## Table Data Storage & Processing Flow

### 1. **Table Extraction Output Formats**

From `phase-2-multimodal.md` (lines 177-204), each extracted table is converted into three formats:

#### **a) JSON Format** (Structured Data)

```json
{
  "headers": ["Model", "F1", "Accuracy", "AUC", "Precision"],
  "rows": [
    ["Random Forest", "0.86", "0.88", "0.93", "0.84"],
    ["LSTM", "0.83", "0.86", "0.90", "0.80"],
    ...
  ]
}
```

- Purpose: Structured storage for programmatic access
- Stored in: `tables.table_data` (JSONB field in Supabase)

#### **b) Markdown Format** (Human-Readable)

```markdown
| Model         | F1   | Accuracy | AUC  | Precision |
| ------------- | ---- | -------- | ---- | --------- |
| Random Forest | 0.86 | 0.88     | 0.93 | 0.84      |
| LSTM          | 0.83 | 0.86     | 0.90 | 0.80      |
```

- Purpose: LLM context, display, citations
- Stored in: `tables.table_markdown` (TEXT field in Supabase)

#### **c) Flattened Text Format** (For Embeddings)

```
Model: Random Forest, F1: 0.86, Accuracy: 0.88, AUC: 0.93, Precision: 0.84
Model: LSTM, F1: 0.83, Accuracy: 0.86, AUC: 0.90, Precision: 0.80
```

- Purpose: Embedding generation (converts structured data to text)
- Stored in: `tables.table_text` (TEXT field in Supabase)

---

### 2. **Storage Locations**

#### **A) Supabase (PostgreSQL) - Primary Storage**

From `db.sql` (lines 58-67), there are two related tables:

**`chunks` table** (lines 14-26):

- `chunk_type`: Set to `'table'` for table chunks
- `table_data`: JSONB field storing the JSON format
- `embedding_type`: Set to `'table'`
- `chunk_text`: Can store flattened text or markdown
- `metadata`: JSONB with table metadata (row_count, col_count, headers, etc.)

**`tables` table** (lines 58-67):

- `table_data`: JSONB - Full structured table (JSON format)
- `table_markdown`: TEXT - Markdown representation
- `table_text`: TEXT - Flattened text for embeddings
- `metadata`: JSONB - Additional metadata (row_count, col_count, headers, page_number)
- Links to `document_id` and `chunk_id`

**Storage Flow:**

1. Extract table → Generate all 3 formats
2. Create entry in `tables` table with all formats
3. Create corresponding entry in `chunks` table with `chunk_type='table'`
4. Link them via `chunk_id`

---

#### **B) Qdrant (Vector Database) - Embeddings Storage**

From `init_qdrant.py` (lines 122-179), the `table_chunks` collection stores:

**Collection**: `table_chunks`

- **Vector dimensions**: 768 (matching text embeddings)
- **Distance metric**: Cosine similarity

**Payload structure** (from phase-2-multimodal.md lines 78-81):

```python
{
  "chunk_id": "uuid",
  "document_id": "uuid",
  "table_data": {...},      # JSON format
  "table_markdown": "...",   # Markdown string
  "metadata": {
    "row_count": 4,
    "col_count": 5,
    "headers": ["Model", "F1", ...],
    "page_number": 8,
    ...
  }
}
```

**Processing:**

1. Take `table_text` (flattened format)
2. Generate embedding using text embedding model (e5-base-v2, same as text chunks)
3. Store 768-dim vector in Qdrant with payload containing table data

---

#### **C) Elasticsearch (BM25 Sparse Index) - Full-Text Search**

From `update_elasticsearch_mapping.py` (referenced in phase-2-multimodal.md lines 107-149), the `chunks` index is extended with:

**New fields for tables:**

- `chunk_type`: Keyword field (`'text'`, `'table'`, `'image'`, `'mixed'`)
- `embedding_type`: Keyword field (`'text'`, `'table'`, `'image'`)
- `table_markdown`: Text field - **Searchable table content** (this is key!)
- Extended `metadata` fields:
  - `table_headers`: Text field (searchable headers)
  - `row_count`: Integer
  - `col_count`: Integer

**Indexing:**

1. Index `table_markdown` as searchable text
2. Index `metadata.table_headers` for header searches
3. Store metadata for filtering (row_count, col_count, page_number)
4. Same `chunk_id` links to Supabase and Qdrant entries

---

### 3. **Processing Pipeline Flow**

Based on phase-2-multimodal.md (lines 236-250):

```
PDF/DOCX Document
    ↓
Table Extraction Service
    ↓
[Table 1] → Generate 3 formats:
    ├─ JSON (structured)
    ├─ Markdown (readable)
    └─ Flattened Text (for embeddings)
    ↓
Store in Supabase:
    ├─ tables table (all 3 formats + metadata)
    └─ chunks table (chunk_type='table', embedding_type='table')
    ↓
Generate Embeddings:
    └─ Use flattened text → e5-base-v2 → 768-dim vector
    ↓
Store in Qdrant:
    └─ table_chunks collection (vector + payload with table_data, table_markdown)
    ↓
Index in Elasticsearch:
    └─ chunks index (table_markdown searchable, metadata indexed)
    ↓
Ready for Retrieval!
```

---

### 4. **Retrieval & Usage**

When a query comes in:

1. **Query Router** determines if tables are needed (keywords like "compare", "table", "data", "statistics")

2. **Hybrid Retrieval**:

   - **BM25 (Elasticsearch)**: Searches `table_markdown` field for keyword matches
   - **Vector Search (Qdrant)**: Semantic search using table embeddings
   - **Merge results** from both sources

3. **Reranking**: Cross-encoder reranker scores table chunks for relevance

4. **Fusion Layer**: Formats retrieved tables as markdown for LLM context:

   ```
   [Table Chunk 1]
   Source: document.pdf, Page 8
   | Model | F1 | Accuracy | AUC | Precision |
   | ----- | -- | -------- | --- | --------- |
   | Random Forest | 0.86 | 0.88 | 0.93 | 0.84 |
   ...
   ```

5. **Answer Generation**: LLM uses markdown tables in context to answer questions

---

### 5. **Key Design Decisions**

1. **Three formats**: JSON (structure), Markdown (readability), Flattened Text (embeddings)
2. **Same embedding model**: Tables use the same text embedding model (e5-base-v2) for consistency
3. **Dual storage**: `tables` table stores all formats; `chunks` table links to embedding/indexing
4. **Searchable markdown**: Elasticsearch indexes markdown for keyword search
5. **Metadata preservation**: Row/column counts, headers, page numbers stored for filtering

This design supports:

- ✅ Structured queries (filter by headers, row counts)
- ✅ Semantic search (vector similarity)
- ✅ Keyword search (BM25 on markdown)
- ✅ LLM-friendly format (markdown in context)
- ✅ Programmatic access (JSON format)

The table data flows through all three storage systems (Supabase → Qdrant → Elasticsearch) in formats optimized for each use case.

Examples using your `tables.json`:

## Table 1: Student Data

### **Markdown Format** (Human-readable, for LLM context):

```markdown
| N   | A   | D   |
| --- | --- | --- |
| H   | 22  | CS  |
| A   | 21  | EE  |
| R   | 23  | ME  |
```

### **Flattened Text Format** (For embedding generation):

```
N: H, A: 22, D: CS
N: A, A: 21, D: EE
N: R, A: 23, D: ME
```

---

## Table 2: Project Data

### **Markdown Format** (Human-readable, for LLM context):

```markdown
| P   | S   | Hrs | O   |
| --- | --- | --- | --- |
| RAG | IP  | 120 | H   |
| UI  | C   | 45  | A   |
| ETL | P   | 0   | R   |
```

### **Flattened Text Format** (For embedding generation):

```
P: RAG, S: IP, Hrs: 120, O: H
P: UI, S: C, Hrs: 45, O: A
P: ETL, S: P, Hrs: 0, O: R
```

---

## Differences

| Aspect          | Markdown Format                   | Flattened Text Format      |
| --------------- | --------------------------------- | -------------------------- |
| **Structure**   | Table with rows/columns           | One line per row           |
| **Readability** | ✅ Easy to read visually          | ❌ Less readable           |
| **Use Case**    | LLM context, display, citations   | Embedding generation       |
| **Storage**     | `tables.table_markdown` (TEXT)    | `tables.table_text` (TEXT) |
| **Search**      | Indexed in Elasticsearch for BM25 | Not directly searchable    |
| **LLM Usage**   | ✅ Directly usable in prompts     | ❌ Not used directly       |

---

## Why both formats?

1. **Markdown** → Used when:

   - Sending context to LLM (readable table format)
   - Displaying in UI
   - Citations in answers
   - BM25 keyword search in Elasticsearch

2. **Flattened Text** → Used when:
   - Generating embeddings (text embedding models need plain text)
   - Semantic similarity search
   - Converting structured data to text for ML models

**Example Query Flow:**

- User asks: "What projects have more than 50 hours?"
- **BM25 search** (Elasticsearch): Searches `table_markdown` for keywords "projects", "hours", "50"
- **Vector search** (Qdrant): Searches using embeddings generated from `table_text` format
- **Results merged** → Markdown format sent to LLM for answer generation

The flattened text preserves the column-value relationships in a format that embedding models can understand semantically.
