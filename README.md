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

| Component                  | Time (s) | % of Total |
| -------------------------- | -------- | ---------- |
| **Storage Operations**     | 7.372    | 28.8%      |
| **Table/Image Extraction** | 6.640    | 25.9%      |
| **Neo4j Graph Building**   | 5.583    | 21.8%      |
| **Vision Processing**      | 2.857    | 11.2%      |
| **Embedding Generation**   | 1.362    | 5.3%       |
| **Elasticsearch Indexing** | 0.292    | 1.1%       |
| **Qdrant Vector Storage**  | 0.188    | 0.7%       |
| **Text Processing**        | 0.004    | 0.02%      |

---

### Retrieval Performance Comparison

#### Dense Retriever (Vector Similarity Search)

| Model                              | Retrieval Time | Relevance Score | Dimensions |
| ---------------------------------- | -------------- | --------------- | ---------- |
| **intfloat/e5-base-v2**            | 20.5 ms        | 81.0%           | 768        |
| **Thenlper/GTE-Base**              | 12.6 ms        | 84.0%           | 768        |
| **Thenlper/GTE-Large**             | 10.8 ms        | 84.9%           | 1024       |
| **intfloat/e5-large-v2**           | 10.5 ms        | 80.3%           | 1024       |
| **intfloat/multilingual-e5-large** | 12.7 ms        | 79.7%           | 1024       |

#### Sparse Retriever (BM25 Keyword Search)

| Metric                     | Value    |
| -------------------------- | -------- |
| **Average Retrieval Time** | 113.2 ms |
| **Relevance Score**        | 99.2%    |

#### Image Retriever (Visual Similarity Search)

| Model                                   | Retrieval Time | Relevance Score | Dimensions |
| --------------------------------------- | -------------- | --------------- | ---------- |
| **sentence-transformers/clip-ViT-L-14** | 51.2 ms        | 27.4%           | 768        |
| **CLIP ViT-B-32**                       | 505.1 ms       | 15.8%           | 512        |
| **SigLIP vit_base_patch16_siglip_224**  | 509.2 ms       | 1.9%            | 768        |

#### Knowledge Graph Retrieval

| Query Type           | Average Duration | Max Duration |
| -------------------- | ---------------- | ------------ |
| **graph_traversal**  | 107 ms           | 186 ms       |
| **by_topics**        | 119 ms           | 214 ms       |
| **by_section_title** | 223 ms           | 422 ms       |
| **by_keywords**      | 670 ms           | 1.21 s       |

#### Embedding Generation Performance

| Embedding Type      | Average Time |
| ------------------- | ------------ |
| **Text Embedding**  | 225.3 ms     |
| **Image Embedding** | 264.6 ms     |

#### Answer Generation Performance

| Model                  | Time to First Token | Total Generation Time |
| ---------------------- | ------------------- | --------------------- |
| **openai/gpt-oss-20b** | 1.16 s              | 2.5 s                 |

#### Query Processing Performance (End-to-End)

| Component              | Average Time | Notes                    |
| ---------------------- | ------------ | ------------------------ |
| **Query Embedding**    | 149 ms       | Query text to vector     |
| **Retrieval (All)**    | ~243 ms      | Parallel sparse + dense + graph + image |
| **Merge Time**         | 48.5 μs      | Result fusion            |
| **Answer Generation**  | 2.5 s        | LLM response generation  |
| **Time to First Token**| 1.16 s       | Initial LLM response     |
| **Total Query Time**   | 3.0 s        | Complete end-to-end (all retrievers) |

---

### Performance Summary Table

| Component              | Metric       | Value        |
| ---------------------- | ------------ | ------------ |
| **Dense Retriever**    | Speed        | 10.5-12.7 ms |
| **Dense Retriever**    | Relevance    | 79.7-84.9%   |
| **Sparse Retriever**   | Speed        | 113.2 ms     |
| **Sparse Retriever**   | Relevance    | 99.2%        |
| **Image Retriever**    | Speed        | 51.2 ms      |
| **Image Retriever**    | Relevance    | 27.4%        |
| **Knowledge Graph**    | Speed        | 243 ms       |
| **Knowledge Graph**    | Chunks/Query | 18.7         |
| **Text Embedding**     | Generation   | 225.3 ms     |
| **Image Embedding**    | Generation   | 264.6 ms     |
| **Query Embedding**    | Generation   | 149 ms       |
| **Answer Generation**  | Total Time   | 2.5 s        |
| **Answer Generation**  | Time to First Token | 1.16 s   |
| **Merge Time**         | Result Fusion| 48.5 μs      |
| **Query Processing**   | Total Time   | 3.0 s        |
| **Document Ingestion** | Total Time   | 25.6 s       |
| **Qdrant Storage**     | Per Vector   | 9.4 ms       |
| **Elasticsearch**      | Per Document | 19.5 ms      |

---

### Model Performance Evolution

| Aspect           | e5-base-v2 | GTE-Base | GTE-Large | e5-large-v2 | multilingual-e5-large |
| ---------------- | ---------- | -------- | --------- | ----------- | --------------------- |
| **Speed**        | 20.5 ms    | 12.6 ms  | 10.8 ms   | 10.5 ms     | 12.7 ms               |
| **Relevance**    | 81.0%      | 84.0%    | 84.9%     | 80.3%       | 79.7%                 |
| **Dimensions**   | 768        | 768      | 1024      | 1024        | 1024                  |
| **Capabilities** | English    | English  | English   | English     | Multilingual          |

---

## 🚀 Quick Start

This guide will help you set up and run the Multimodal RAG system from scratch.

### Prerequisites

- **Docker & Docker Compose** - For running infrastructure services
- **Python 3.9+** - For the backend API
- **Node.js 18+** - For the frontend
- **Supabase Account** - For PostgreSQL database and storage
- **API Keys** (optional but recommended):
  - Groq API key (for LLM answer generation)
  - OpenAI API key (for vision LLM, if using)

---

### Step 1: Start Infrastructure Services

Start all required services using Docker Compose:

```bash
# From the project root directory
docker-compose up -d
```

This will start:

- **Qdrant** (Vector Database) - `http://localhost:6333`
- **Elasticsearch** (BM25 Index) - `http://localhost:9200`
- **Neo4j** (Knowledge Graph) - `bolt://localhost:7687`
- **MinIO** (S3 Storage) - `http://localhost:9000` (Console: `http://localhost:9090`)
- **Prometheus** (Metrics) - `http://localhost:9091`
- **Grafana** (Dashboards) - `http://localhost:3001`
- **Loki** (Logs) - `http://localhost:3100`

**Verify services are running:**

- Qdrant Dashboard: `http://localhost:6333/dashboard`
- Elasticsearch Health: `http://localhost:9200/_cluster/health`
- MinIO Console: `http://localhost:9090` (login: `admin` / `admin12345`)

---

### Step 2: Set Up Supabase

1. **Create a Supabase Project:**

   - Go to [supabase.com](https://supabase.com) and create a new project
   - Wait for the project to be fully provisioned

2. **Get Your Supabase Credentials:**

   - Go to **Settings** → **API**
   - Copy the following:
     - **Project URL** (e.g., `https://xxxxx.supabase.co`)
     - **Service Role Key** (keep this secret!)
     - **Anon Key** (optional, for frontend)

3. **Create Database Tables:**
   - Go to **SQL Editor** in your Supabase dashboard
   - Open the file `db/db.sql` from this repository
   - Copy and paste the entire SQL script into the SQL Editor
   - Click **Run** to execute
   - Verify tables are created: `documents`, `chunks`, `images`, `tables`

---

### Step 3: Configure Backend

1. **Navigate to Backend Directory:**

   ```bash
   cd backend
   ```

2. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create Environment File:**
   Create a `.env` file in the `backend` directory with the following variables:

   ```env
   # Supabase Configuration
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   SUPABASE_ANON_KEY=your-anon-key

   # Neo4j Configuration
   NEO4J_PASSWORD=your-neo4j-password

   # Optional: LLM API Keys
   GROQ_API_KEY=your-groq-api-key
   OPENAI_API_KEY=your-openai-api-key
   ```

4. **Configure `config.yaml`:**
   - Open `backend/config.yaml`
   - Review and adjust settings as needed:
     - **Qdrant**: Already configured for localhost
     - **Elasticsearch**: Already configured for localhost
     - **MinIO**: Already configured for localhost
     - **Neo4j**: Update `uri` if running in Docker: `bolt://neo4j:7687`
     - **Embedding Models**: Adjust `device` (cpu/cuda) based on your hardware
     - **LLM Provider**: Configure your preferred provider (groq, anthropic, ollama)

---

### Step 4: Initialize System Schemas

Run the reconstruction script to create all necessary indexes, collections, and schemas:

```bash
# From the backend directory
python scripts/pipeline_ops/reconstruct.py
```

This will:

- ✅ Verify Supabase tables exist
- ✅ Create Elasticsearch index with proper mappings
- ✅ Create Qdrant collections (text_chunks, table_chunks, image_chunks)
- ✅ Verify Neo4j constraints and indexes
- ✅ Verify MinIO bucket exists

**Note:** If you need to start fresh, you can wipe everything first:

```bash
python scripts/pipeline_ops/clean_slate.py --confirm
# Then run reconstruct.py again
```

---

### Step 5: Start Backend API

1. **Navigate to Backend Directory:**

   ```bash
   cd backend
   ```

2. **Start the FastAPI Server:**

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at: `http://localhost:8000`

   - API Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

---

### Step 6: Start Frontend

1. **Navigate to Frontend Directory:**

   ```bash
   cd frontend
   ```

2. **Install Dependencies:**

   ```bash
   npm install
   ```

3. **Start Development Server:**

   ```bash
   npm run dev
   ```

   The frontend will be available at: `http://localhost:5173` (or the port shown in terminal)

---

### Step 7: Verify Everything is Running

1. **Backend API:** `http://localhost:8000/health` should return `{"status": "healthy"}`
2. **Frontend:** Open `http://localhost:5173` in your browser
3. **Qdrant:** `http://localhost:6333/dashboard` - Collections should be visible
4. **Elasticsearch:** `http://localhost:9200/_cat/indices` - `rag_chunks` index should exist
5. **MinIO:** `http://localhost:9090` - `raw-documents` bucket should exist

---

### Service Access URLs

Once everything is running, access services at:

| Service              | URL                               | Credentials            |
| -------------------- | --------------------------------- | ---------------------- |
| **Backend API**      | `http://localhost:8000`           | -                      |
| **API Docs**         | `http://localhost:8000/docs`      | -                      |
| **Frontend**         | `http://localhost:5173`           | -                      |
| **Qdrant Dashboard** | `http://localhost:6333/dashboard` | -                      |
| **Elasticsearch**    | `http://localhost:9200`           | -                      |
| **MinIO Console**    | `http://localhost:9090`           | `admin` / `admin12345` |
| **Grafana**          | `http://localhost:3001`           | `admin` / `admin`      |
| **Prometheus**       | `http://localhost:9091`           | -                      |

---

### Troubleshooting

**Backend won't start:**

- Check that all environment variables are set in `backend/.env`
- Verify Supabase credentials are correct
- Ensure Docker services are running: `docker-compose ps`

**Frontend won't connect:**

- Check that backend API is running on port 8000
- Verify CORS settings in `backend/config.yaml` include your frontend URL

**Database connection errors:**

- Verify Supabase project is active
- Check that `db/db.sql` was executed successfully
- Ensure `SUPABASE_SERVICE_ROLE_KEY` is set correctly

**Collection/index errors:**

- Run `python scripts/pipeline_ops/reconstruct.py` to recreate missing schemas
- Check service logs: `docker-compose logs [service-name]`

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
- Embedding generation times (query and document)
- Answer generation performance (time to first token, total generation time)
- Query processing end-to-end latency
- Document ingestion performance
- Knowledge graph query performance

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Hisham

---

**Last Updated**: December 21, 2025  
**Performance Data**: Based on real-time execution logs and Prometheus metrics
