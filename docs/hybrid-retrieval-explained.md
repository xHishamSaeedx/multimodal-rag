# Hybrid RAG Retrieval Architecture Explained

## Overview: Hybrid RAG Retrieval System

This document explains the components of the hybrid retrieval system that combines **sparse (BM25)** and **dense (vector)** retrieval methods for optimal search results.

---

## Core Concepts

### Sparse vs Dense Retrieval

**Sparse (BM25) - Keyword-Based:**
- Searches for **exact keyword matches** in text
- Fast and efficient for specific terms
- Great for: technical terms, IDs, names, exact phrases
- Uses: **Elasticsearch** with BM25 scoring algorithm
- Example: Query "Python API" finds documents containing both "Python" and "API"

**Dense (Vector) - Semantic-Based:**
- Uses **embeddings** to find semantically similar content
- Understands meaning and context
- Great for: paraphrases, conceptual queries, synonyms
- Uses: **Qdrant** with cosine similarity on embedding vectors
- Example: Query "machine learning" finds documents about "ML", "AI algorithms", "neural networks"

**Why Both?**
- **BM25** catches exact matches and technical terms
- **Vector** catches semantic similarity and paraphrases
- **Together** = Best of both worlds! 🎯

---

## Component Breakdown

### 1. **SparseRepository** (`sparse_repository.py`)

**Purpose:** Low-level Elasticsearch operations for BM25 indexing and search.

**Responsibilities:**
- Index chunks into Elasticsearch (text, metadata, IDs)
- Search using BM25 algorithm
- Delete chunks from index

**What it does:**
- `index_chunks()`: Bulk indexes chunks with text, metadata, timestamps
- `search()`: Performs BM25 search with optional filters
- `delete_chunks()`: Removes chunks from index

**Analogy:** Think of it as the "database layer" for keyword search - it handles all the Elasticsearch operations.

---

### 2. **SparseRetriever** (`sparse_retriever.py`)

**Purpose:** Service layer for BM25 retrieval - wraps SparseRepository.

**Responsibilities:**
- Wraps SparseRepository with business logic
- Handles query processing and error handling
- Provides convenient methods for common use cases

**What it does:**
- `retrieve()`: Takes text query, calls SparseRepository, returns formatted results
- `retrieve_by_document()`: Filter by document ID
- `retrieve_by_type()`: Filter by document type

**Analogy:** Think of it as a "service" that uses the database layer to perform searches - it's the interface you'd use in your application code.

---

### 3. **DenseRetriever** (`dense_retriever.py`)

**Purpose:** Service layer for vector similarity search.

**Responsibilities:**
- Generates query embeddings using TextEmbedder
- Searches Qdrant using vector similarity
- Formats results consistently

**What it does:**
- `retrieve()`: Embeds query text, searches Qdrant, returns results with similarity scores
- Handles special cases (e.g., e5-base-v2 model needs "query:" prefix)
- Validates embedding dimensions match vector store

**Analogy:** Similar to SparseRetriever, but for semantic/vector search instead of keyword search.

---

### 4. **HybridRetriever** (`hybrid_retriever.py`)

**Purpose:** Combines sparse and dense retrieval for best results.

**Responsibilities:**
- Orchestrates both retrieval methods
- Merges and deduplicates results
- Normalizes and combines scores
- Ranks final results

**What it does:**

1. **Parallel Retrieval**: Calls SparseRetriever and DenseRetriever simultaneously
2. **Merging**: Combines results from both, deduplicates by `chunk_id`
3. **Score Normalization**: Normalizes BM25 and vector scores to [0, 1] range
4. **Score Combination**: Weighted average (40% BM25 + 60% vector)
5. **Ranking**: Sorts by combined score, returns top-N results

**Analogy:** Think of it as a "conductor" that orchestrates both search methods and combines their results intelligently.

---

## How They Work Together

```
User Query: "What is machine learning?"

┌─────────────────────────────────────────────────────────────┐
│                    HybridRetriever                           │
│  (Orchestrates both retrieval methods)                      │
└──────────────┬──────────────────────┬────────────────────────┘
               │                      │
               │                      │
    ┌──────────▼──────────┐  ┌───────▼──────────┐
    │  SparseRetriever    │  │ DenseRetriever   │
    │  (BM25 Search)      │  │ (Vector Search) │
    └──────────┬──────────┘  └───────┬──────────┘
               │                      │
               │                      │
    ┌──────────▼──────────┐  ┌───────▼──────────┐
    │  SparseRepository   │  │ VectorRepository  │
    │  (Elasticsearch)    │  │ (Qdrant)         │
    └─────────────────────┘  └──────────────────┘
```

---

## Data Flow Example

Let's trace through a complete example:

1. **User Query**: "What is machine learning?"

2. **HybridRetriever splits into two parallel paths:**
   - **Path A (Sparse)**: SparseRetriever → SparseRepository → Elasticsearch (BM25 search)
   - **Path B (Dense)**: DenseRetriever → embeds query → VectorRepository → Qdrant (vector search)

3. **Results come back:**
   - **Sparse**: 10 chunks with BM25 scores (e.g., 2.5, 1.8, 1.2...)
   - **Dense**: 10 chunks with similarity scores (e.g., 0.85, 0.72, 0.65...)

4. **Deduplication**: Remove duplicate `chunk_id`s (some chunks appear in both results)

5. **Score Normalization**: 
   - BM25 scores: [0.0, 1.0] range
   - Vector scores: [0.0, 1.0] range

6. **Score Combination**: Weighted average (40% BM25 + 60% vector)

7. **Final Ranking**: Sort by combined score, return top 10

---

## Why Hybrid Retrieval?

**BM25 (Sparse) is great for:**
- ✅ Exact keyword matches
- ✅ Technical terms
- ✅ IDs, names, specific phrases
- ✅ Fast retrieval

**Vector (Dense) is great for:**
- ✅ Semantic similarity
- ✅ Paraphrases ("car" vs "automobile")
- ✅ Conceptual queries
- ✅ Understanding context

**Hybrid (Both) gives you:**
- ✅ Best of both worlds
- ✅ Catches exact matches AND semantic matches
- ✅ More comprehensive results
- ✅ Better overall retrieval quality

---

## Repository vs Retriever Pattern

**Repository Pattern:**
- **Low-level data access** (database operations)
- Direct interaction with storage (Elasticsearch, Qdrant)
- CRUD operations (Create, Read, Update, Delete)
- Examples: `SparseRepository`, `VectorRepository`

**Retriever Pattern:**
- **Service layer** (business logic)
- Uses repositories to perform searches
- Handles query processing, error handling
- Provides convenient methods
- Examples: `SparseRetriever`, `DenseRetriever`, `HybridRetriever`

**Why this separation?**
- **Separation of concerns**: Data access vs business logic
- **Testability**: Can mock repositories when testing retrievers
- **Maintainability**: Changes to storage don't affect business logic
- **Reusability**: Repositories can be used by multiple services

---

## Summary

| Component | Purpose | Technology | What It Does |
|-----------|---------|------------|--------------|
| **SparseRepository** | Elasticsearch operations | Elasticsearch | Index/search chunks with BM25 |
| **SparseRetriever** | BM25 search service | Uses SparseRepository | Provides BM25 search interface |
| **DenseRetriever** | Vector search service | Uses VectorRepository | Provides semantic search interface |
| **HybridRetriever** | Combines both methods | Uses both retrievers | Best of both worlds! |

Together, these components create a powerful hybrid retrieval system that leverages both keyword matching and semantic understanding for optimal search results! 🚀

