# 🚀 Hybrid Multimodal RAG Architecture (Enterprise-Grade)

A next-generation Retrieval-Augmented Generation system optimized for large-scale, multimodal enterprise knowledge bases.

This architecture combines hybrid retrieval (BM25 + dense vectors + metadata filtering) with multimodal understanding (text, tables, images, diagrams) and optional graph-based reasoning for complex multi-hop queries. It is designed for speed, accuracy, scalability, and long-term maintainability, supported by full MLOps pipelines for continuous improvement.

## 📹 Demo Video

Watch the system in action:

<div align="center">
  <a href="https://youtu.be/WwYofMSvdPE" target="_blank">
    <img src="https://img.youtube.com/vi/WwYofMSvdPE/maxresdefault.jpg" alt="Multimodal RAG Demo Video" width="800" style="max-width: 100%; border-radius: 8px;">
  </a>
</div>

<p align="center">
  <a href="https://youtu.be/WwYofMSvdPE" target="_blank">🎥 Watch on YouTube</a>
</p>

## 🏗️ System Architecture

### High-Level Architecture

<div align="center">
  <img src="./docs/diagrams/system-architecture.png" alt="System Architecture" width="100%" style="max-width: 1200px;">
</div>

### Document Processing Flow

<div align="center">
  <img src="./docs/diagrams/document-processing-flow.png" alt="Document Processing Flow" width="100%" style="max-width: 1200px;">
</div>

### Query Processing Flow

<div align="center">
  <img src="./docs/diagrams/query-processing-flow.png" alt="Query Processing Flow" width="100%" style="max-width: 1200px;">
</div>

### End-to-End Pipeline

<div align="center">
  <img src="./docs/diagrams/end-to-end-pipeline.png" alt="End-to-End Pipeline" width="100%" style="max-width: 1200px;">
</div>

### Executive Summary

<div align="center">
  <img src="./docs/diagrams/executive.png" alt="Executive Summary" width="100%" style="max-width: 1200px;">
</div>

## 📊 Performance Metrics

### Document Ingestion Performance

**Total Ingestion Time**: 25.6 seconds (12-page PDF with 5 images, 5 tables)

| Component                  | Time (s) | % of Total | Status                |
| -------------------------- | -------- | ---------- | --------------------- |
| **Storage Operations**     | 7.372    | 28.8%      | 🔴 Primary bottleneck |
| **Table/Image Extraction** | 6.640    | 25.9%      | 🟡 Parallel working   |
| **Neo4j Graph Building**   | 5.583    | 21.8%      | 🟡 Entity-heavy       |
| **Vision Processing**      | 2.857    | 11.2%      | ✅ Good               |
| **Embedding Generation**   | 1.362    | 5.3%       | ✅ Good               |
| **Elasticsearch Indexing** | 0.292    | 1.1%       | ✅ Fast               |
| **Qdrant Vector Storage**  | 0.188    | 0.7%       | ✅ Very fast          |
| **Text Processing**        | 0.004    | 0.02%      | ✅ Negligible         |

**Key Highlights:**

- ✅ Parallel extraction working (saved 5.95 seconds)
- ✅ Qdrant vector storage: 9.4ms per vector
- ✅ Elasticsearch indexing: 19.5ms per document
- 🔴 Image uploads: 3.441s (largest single bottleneck)
- 🟡 Table extraction: 1.328s per table

**Optimization Potential:**

- Conservative: 22.6s (11.7% faster)
- Aggressive: 13.0s (49.2% faster)

---

### Retrieval Performance Comparison

#### Dense Retriever (Vector Similarity Search)

| Model                              | Retrieval Time   | Relevance Score | Dimensions | Notes                             |
| ---------------------------------- | ---------------- | --------------- | ---------- | --------------------------------- |
| **intfloat/e5-base-v2**            | 20.5 ms          | 81.0%           | 768        | Baseline model                    |
| **Thenlper/GTE-Base**              | 12.6 ms ⭐       | 84.0% ⭐        | 768        | +38% faster, +3.7% relevance      |
| **Thenlper/GTE-Large**             | 10.8 ms ⭐⭐     | 84.9% ⭐⭐      | 1024       | +47% faster, +4.8% relevance      |
| **intfloat/e5-large-v2**           | 10.5 ms ⭐⭐⭐   | 80.3% ⭐⭐⭐    | 1024       | +49% faster, -0.7% relevance      |
| **intfloat/multilingual-e5-large** | 12.7 ms ⭐⭐⭐⭐ | 79.7% ⭐⭐⭐⭐  | 1024       | +38% faster, multilingual support |

**Performance Summary:**

- ⚡ **Fastest**: e5-large-v2 (10.5ms)
- 🎯 **Most Accurate**: GTE-Large (84.9% relevance)
- 🌍 **Multilingual**: multilingual-e5-large (79.7% relevance, cross-language support)

#### Sparse Retriever (BM25 Keyword Search)

| Metric                     | Value    | Notes                                        |
| -------------------------- | -------- | -------------------------------------------- |
| **Average Retrieval Time** | 113.2 ms | ⭐⭐⭐ Good (under 200ms)                    |
| **Relevance Score**        | 99.2%    | 🟢 Excellent - near-perfect keyword matching |
| **Total Queries**          | 5        | Consistent performance                       |

**Characteristics:**

- ✅ Near-perfect keyword matching (99.2% relevance)
- ✅ Stable 113ms response time
- ✅ Independent of embedding model changes

#### Image Retriever (Visual Similarity Search)

| Model                                   | Retrieval Time | Relevance Score | Dimensions | Notes                          |
| --------------------------------------- | -------------- | --------------- | ---------- | ------------------------------ |
| **sentence-transformers/clip-ViT-L-14** | 51.2 ms        | 27.4%           | 768        | ⭐⭐⭐ Moderate performance    |
| **CLIP ViT-B-32**                       | 505.1 ms       | 15.8%           | 512        | ⭐⭐ Moderate-slow             |
| **SigLIP vit_base_patch16_siglip_224**  | 509.2 ms       | 1.9%            | 768        | 🔴 Poor (compatibility issues) |

**Performance Analysis:**

- ✅ **Best Model**: CLIP ViT-L-14 (51.2ms, 27.4% relevance)
- ⚠️ **SigLIP Issues**: Significant degradation (1.9% relevance, 10x slower)
- 🎯 **Use Case**: Visual content search in technical documents

#### Knowledge Graph Retrieval

**Unified Performance:**

- **Average Retrieval Time**: 243 ms
- **Max Retrieval Time**: 507 ms
- **Total Queries**: 20 (4 query types × 5 queries)

**Performance by Query Type:**

| Query Type           | Average Duration | Max Duration | Queries | Ranking    |
| -------------------- | ---------------- | ------------ | ------- | ---------- |
| **graph_traversal**  | 107 ms           | 186 ms       | 5       | 🥇 Fastest |
| **by_topics**        | 119 ms           | 214 ms       | 5       | 🥈         |
| **by_section_title** | 223 ms           | 422 ms       | 5       | 🥉         |
| **by_keywords**      | 670 ms           | 1.21 s       | 5       | Slowest    |

**Chunk Retrieval:**

- **Total Chunks Retrieved**: 94 chunks
- **Average Chunks per Query**: 18.7 chunks
- **Graph vs Hybrid**: Graph provides ~87% more chunks per query (18.7 vs 10.0)

#### Embedding Generation Performance

| Embedding Type      | Average Time | Performance Rating | Notes                                   |
| ------------------- | ------------ | ------------------ | --------------------------------------- |
| **Text Embedding**  | 225.3 ms     | ⭐⭐ Moderate      | Semantic text vector generation         |
| **Image Embedding** | 264.6 ms     | ⭐⭐ Moderate      | Visual feature extraction (CLIP/SigLIP) |

**Generation Volume (6-hour window):**

- **Text Embeddings**: ~64 embeddings (0.0030/sec)
- **Image Embeddings**: ~11 embeddings (0.0005/sec)

---

### Performance Summary Table

| Component              | Metric       | Value        | Status                |
| ---------------------- | ------------ | ------------ | --------------------- |
| **Dense Retriever**    | Speed        | 10.5-12.7 ms | ✅ Excellent          |
| **Dense Retriever**    | Relevance    | 79.7-84.9%   | ✅ Excellent          |
| **Sparse Retriever**   | Speed        | 113.2 ms     | ✅ Good               |
| **Sparse Retriever**   | Relevance    | 99.2%        | ✅ Excellent          |
| **Image Retriever**    | Speed        | 51.2 ms      | ✅ Moderate           |
| **Image Retriever**    | Relevance    | 27.4%        | 🟡 Moderate           |
| **Knowledge Graph**    | Speed        | 243 ms       | ✅ Good               |
| **Knowledge Graph**    | Chunks/Query | 18.7         | ✅ High recall        |
| **Text Embedding**     | Generation   | 225.3 ms     | ✅ Moderate           |
| **Image Embedding**    | Generation   | 264.6 ms     | ✅ Moderate           |
| **Document Ingestion** | Total Time   | 25.6 s       | 🟡 Good (optimizable) |
| **Qdrant Storage**     | Per Vector   | 9.4 ms       | ✅ Very fast          |
| **Elasticsearch**      | Per Document | 19.5 ms      | ✅ Fast               |

---

### Model Performance Evolution

**Dense Retriever Model Comparison:**

| Aspect           | e5-base-v2 | GTE-Base | GTE-Large | e5-large-v2 | multilingual-e5-large | Improvement    |
| ---------------- | ---------- | -------- | --------- | ----------- | --------------------- | -------------- |
| **Speed**        | 20.5 ms    | 12.6 ms  | 10.8 ms   | 10.5 ms     | 12.7 ms               | +38-49% faster |
| **Relevance**    | 81.0%      | 84.0%    | 84.9%     | 80.3%       | 79.7%                 | +4.8% better   |
| **Dimensions**   | 768        | 768      | 1024      | 1024        | 1024                  | Higher quality |
| **Capabilities** | English    | English  | English   | English     | Multilingual          | Cross-language |

---

## 🚀 Quick Start

### Service Access URLs

When running services via Docker Compose, access them at the following localhost URLs:

#### Qdrant (Vector Database)

- **REST API**: `http://localhost:6333`
- **Dashboard/Web UI**: `http://localhost:6333/dashboard`
- **gRPC**: `localhost:6334` (gRPC endpoint, not HTTP)

#### Elasticsearch (BM25 Sparse Index)

- **HTTP API**: `http://localhost:9200`
- **Cluster Health**: `http://localhost:9200/_cluster/health`
- **Transport**: `localhost:9300` (not HTTP)

#### MinIO (S3-compatible Storage)

- **S3 API**: `http://localhost:9000`
- **Console UI**: `http://localhost:9090`
  - Default credentials: `admin` / `admin12345`

**Quick Access:**

- Qdrant Dashboard: `http://localhost:6333`
- Elasticsearch: `http://localhost:9200`
- MinIO Console: `http://localhost:9090` (login with admin/admin12345)

All services are on the `rag-network` Docker network and can communicate using their service names (`qdrant`, `elasticsearch`, `minio`).

---

## 🎯 Key Features

### Hybrid Retrieval

- **Sparse (BM25)**: 99.2% keyword matching relevance, 113ms response
- **Dense (Vector)**: 79.7-84.9% semantic relevance, 10.5-12.7ms response
- **Graph-based**: 243ms average, 18.7 chunks per query
- **Multimodal**: Text, tables, images with unified scoring

### Multimodal Understanding

- **Text**: Fast extraction and chunking (0.004s processing)
- **Tables**: Camelot extraction with JSON/markdown conversion
- **Images**: CLIP embeddings with captioning (51.2ms retrieval)
- **Diagrams**: OCR + visual similarity search

### Knowledge Graph

- **Entity Extraction**: spaCy NER with cross-document resolution
- **Relationship Mapping**: Co-occurrence analysis
- **Multi-hop Reasoning**: Graph traversal for complex queries
- **Topic Navigation**: Cross-document topic linking

### Performance Optimizations

- **Parallel Extraction**: Saves 5.95s per document
- **Fast Vector Storage**: 9.4ms per vector (Qdrant)
- **Efficient Indexing**: 19.5ms per document (Elasticsearch)
- **Model Evolution**: 38-49% speed improvements across models

---

## 📈 Performance Insights

### Strengths

- ✅ **Fast Retrieval**: Sub-15ms dense retrieval, sub-250ms graph retrieval
- ✅ **High Relevance**: 80-99% relevance scores across retrieval types
- ✅ **Parallel Processing**: Efficient extraction and indexing
- ✅ **Scalable Architecture**: Multi-index strategy with independent scaling

### Optimization Opportunities

- 🔴 **Image Uploads**: 3.441s bottleneck (connection pooling, parallel uploads)
- 🟡 **Table Extraction**: 1.328s per table (library optimization)
- 🟡 **Graph Building**: 5.583s (optional feature, faster NER)
- 🟡 **Image Relevance**: 27.4% (model fine-tuning, preprocessing)

---

## 📚 Architecture Components

### Data & Storage Layer

- **Raw Data Lake**: MinIO/S3 for unprocessed documents
- **Processed Document Store**: Supabase PostgreSQL for metadata
- **Vector Database**: Qdrant for dense embeddings
- **Sparse Index**: Elasticsearch for BM25 search
- **Knowledge Graph**: Neo4j for entity relationships

### ML Services Layer

- **Multimodal Ingestion**: Text, table, image extraction
- **Embedding Services**: Text (E5/GTE), Image (CLIP/SigLIP)
- **Knowledge Graph Builder**: Entity extraction, relationship mapping

### Retrieval & Reasoning Layer

- **Query Router**: Modality detection and routing
- **Hybrid Retrieval**: Parallel sparse + dense + graph search
- **Multimodal Fusion**: Score normalization and weighted merging
- **Answer Generation**: LLM with hallucination guardrails

### MLOps Layer

- **Experiment Tracking**: Model versioning and metrics
- **Data Versioning**: Document and embedding versioning
- **Automated Pipelines**: Ingestion, embedding, indexing
- **Monitoring**: Prometheus + Grafana for real-time metrics

---

## 🔧 Technology Stack

- **Vector Database**: Qdrant
- **Sparse Search**: Elasticsearch (BM25)
- **Graph Database**: Neo4j
- **Object Storage**: MinIO (S3-compatible)
- **Database**: Supabase (PostgreSQL)
- **Embedding Models**:
  - Text: `intfloat/multilingual-e5-large` (1024d)
  - Image: `sentence-transformers/clip-ViT-L-14` (768d)
- **NLP**: spaCy (NER), BLIP (image captioning)
- **Monitoring**: Prometheus + Grafana

---

## 📊 Monitoring

Access real-time metrics at:

- **Prometheus**: `http://localhost:9091`
- **Grafana**: `http://localhost:3001`

Key metrics tracked:

- Retrieval latency by type (sparse, dense, graph, image)
- Relevance scores per retrieval method
- Embedding generation times
- Document ingestion performance
- Knowledge graph query performance

---

## 📝 License

[Add your license information here]

---

## 🤝 Contributing

[Add contribution guidelines here]

---

**Last Updated**: December 21, 2025  
**Performance Data**: Based on real-time execution logs and Prometheus metrics
