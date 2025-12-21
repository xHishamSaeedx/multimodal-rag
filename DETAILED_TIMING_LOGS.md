# Detailed Timing Logs Implementation

**Date**: December 21, 2025  
**Status**: Implemented

---

## Overview

Added comprehensive timing logs throughout the document ingestion pipeline to track the duration of every operation. These logs are written directly to the terminal/log files (not Prometheus metrics) to help identify overhead and bottlenecks.

---

## Implementation

### 1. **Created `log_timing` Context Manager**

**File**: `backend/app/services/ingestion/pipeline.py`

A reusable context manager that logs the start and end time of any operation:

```python
@contextmanager
def log_timing(operation_name: str, logger=logger):
    """
    Context manager to log operation timing.

    Usage:
        with log_timing("Database Insert"):
            # operation code
            pass
    """
    start_time = time.time()
    logger.info(f"⏱️  [{operation_name}] Starting...")
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"⏱️  [{operation_name}] Completed in {duration:.3f}s")
```

---

## Operations Tracked

### **Storage Operations**

1. **MinIO Upload**

   - `⏱️  [MinIO Upload] Completed in X.XXXs`
   - Raw document upload to object storage

2. **Supabase - Create Document**

   - `⏱️  [Supabase - Create Document] Completed in X.XXXs`
   - Document record creation in database

3. **Supabase - Create Text Chunks**

   - `⏱️  [Supabase - Create N Text Chunks] Completed in X.XXXs`
   - Batch insertion of text chunks

4. **Supabase - Create Table Chunks**

   - `⏱️  [Supabase - Create N Table Chunks] Completed in X.XXXs`
   - Batch insertion of table chunks

5. **Supabase - Store Table Metadata**

   - `⏱️  [Supabase - Store N Table Metadata] Completed in X.XXXs`
   - Table metadata storage in tables table

6. **Supabase Storage - Upload Image**

   - `⏱️  [Supabase Storage - Upload Image N] Completed in X.XXXs`
   - Individual image upload to Supabase storage (per image)

7. **Supabase - Create Image Record**
   - `⏱️  [Supabase - Create Image N Record] Completed in X.XXXs`
   - Image metadata record creation (per image)

---

### **Processing Operations**

8. **Text Chunking**

   - `⏱️  [Text Chunking] Completed in X.XXXs`
   - Text splitting into chunks

9. **Process Tables**
   - `⏱️  [Process N Tables] Completed in X.XXXs`
   - Table conversion to JSON/markdown/text

---

### **Embedding Generation**

10. **Text Embeddings Generation**

    - `⏱️  [Text Embeddings Generation (N chunks)] Completed in X.XXXs`
    - Batch text embedding generation

11. **Table Embeddings Generation**

    - `⏱️  [Table Embeddings Generation (N tables)] Completed in X.XXXs`
    - Batch table embedding generation

12. **Image Embeddings Generation**
    - `⏱️  [Image Embeddings Generation (N images)] Completed in X.XXXs`
    - Batch image embedding generation

---

### **Vector Storage (Qdrant)**

13. **Qdrant - Store Text Vectors**

    - `⏱️  [Qdrant - Store N Text Vectors] Completed in X.XXXs`
    - Upsert text embeddings to Qdrant

14. **Qdrant - Store Table Vectors**

    - `⏱️  [Qdrant - Store N Table Vectors] Completed in X.XXXs`
    - Upsert table embeddings to Qdrant (table_chunks collection)

15. **Qdrant - Store Image Vectors**
    - `⏱️  [Qdrant - Store N Image Vectors] Completed in X.XXXs`
    - Upsert image embeddings to Qdrant (image_chunks collection)

---

### **Sparse Indexing (Elasticsearch)**

16. **Elasticsearch - Index Text Chunks**

    - `⏱️  [Elasticsearch - Index N Text Chunks] Completed in X.XXXs`
    - BM25 indexing for text chunks

17. **Elasticsearch - Index Table Chunks**
    - `⏱️  [Elasticsearch - Index N Table Chunks] Completed in X.XXXs`
    - BM25 indexing for table chunks

---

### **Vision Processing**

18. **Vision - Caption Image**
    - `⏱️  [Vision - Caption Image N] Completed in X.XXXs`
    - Image captioning per image (BLIP model)

---

### **Knowledge Graph**

19. **Neo4j - Build Knowledge Graph**
    - `⏱️  [Neo4j - Build Knowledge Graph] Completed in X.XXXs`
    - Complete graph construction (nodes + relationships)

---

### **Total Pipeline**

20. **TOTAL PIPELINE**
    - `⏱️  [TOTAL PIPELINE] Completed in X.XXXs`
    - End-to-end ingestion time

---

## Example Log Output

When you ingest a document, you'll now see detailed timing logs like this:

```
🚀 Starting document ingestion pipeline for: document.pdf
⏱️  [MinIO Upload] Starting...
⏱️  [MinIO Upload] Completed in 0.234s
Step 2: Extracting text, tables, and images from: document.pdf
⏱️  Text extraction completed in 0.06s
⏱️  Table extraction completed in 8.09s (5 tables)
⏱️  Image extraction completed in 6.87s (5 images)
⏱️  [Text Chunking] Starting...
⏱️  [Text Chunking] Completed in 0.002s
⏱️  [Supabase - Create Document] Starting...
⏱️  [Supabase - Create Document] Completed in 0.152s
⏱️  [Supabase - Create 10 Text Chunks] Starting...
⏱️  [Supabase - Create 10 Text Chunks] Completed in 0.234s
⏱️  [Process 5 Tables] Starting...
⏱️  [Process 5 Tables] Completed in 0.045s
⏱️  [Supabase - Create 5 Table Chunks] Starting...
⏱️  [Supabase - Create 5 Table Chunks] Completed in 0.187s
⏱️  [Supabase - Store 5 Table Metadata] Starting...
⏱️  [Supabase - Store 5 Table Metadata] Completed in 0.098s
⏱️  [Text Embeddings Generation (10 chunks)] Starting...
⏱️  [Text Embeddings Generation (10 chunks)] Completed in 0.567s
⏱️  [Table Embeddings Generation (5 tables)] Starting...
⏱️  [Table Embeddings Generation (5 tables)] Completed in 0.289s
⏱️  [Qdrant - Store 10 Text Vectors] Starting...
⏱️  [Qdrant - Store 10 Text Vectors] Completed in 0.312s
⏱️  [Qdrant - Store 5 Table Vectors] Starting...
⏱️  [Qdrant - Store 5 Table Vectors] Completed in 0.156s
⏱️  [Vision - Caption Image 1] Starting...
⏱️  [Vision - Caption Image 1] Completed in 0.592s
⏱️  [Supabase Storage - Upload Image 1] Starting...
⏱️  [Supabase Storage - Upload Image 1] Completed in 0.234s
⏱️  [Supabase - Create Image 1 Record] Starting...
⏱️  [Supabase - Create Image 1 Record] Completed in 0.089s
... (repeated for each image)
⏱️  [Image Embeddings Generation (5 images)] Starting...
⏱️  [Image Embeddings Generation (5 images)] Completed in 0.286s
⏱️  [Qdrant - Store 5 Image Vectors] Starting...
⏱️  [Qdrant - Store 5 Image Vectors] Completed in 0.145s
⏱️  [Elasticsearch - Index 10 Text Chunks] Starting...
⏱️  [Elasticsearch - Index 10 Text Chunks] Completed in 0.423s
⏱️  [Elasticsearch - Index 5 Table Chunks] Starting...
⏱️  [Elasticsearch - Index 5 Table Chunks] Completed in 0.198s
⏱️  [Neo4j - Build Knowledge Graph] Starting...
⏱️  [Neo4j - Build Knowledge Graph] Completed in 0.163s
✅ Pipeline complete in 27.36s: Document d123... ingested with 10 text chunks, 5 table chunks, 5 image chunks
⏱️  [TOTAL PIPELINE] Completed in 27.364s
```

---

## Benefits

### 1. **Granular Overhead Visibility**

- See exactly where time is spent
- Identify slow storage operations
- Detect network latency issues

### 2. **Per-Operation Breakdown**

- Every database call is timed
- Every storage operation is tracked
- Every processing step is measured

### 3. **Easy Analysis**

- Logs are human-readable
- Can be parsed programmatically
- Can be aggregated for analysis

### 4. **No Performance Impact**

- Minimal overhead from timing
- No external dependencies
- Simple Python time.time() calls

---

## Usage

### **View Logs in Real-Time**

When running the backend locally:

```bash
# Watch logs in real-time
tail -f backend/logs/app.log | grep "⏱️"
```

Or just watch the terminal output when ingesting documents.

### **Analyze Timing Logs**

You can parse the logs to create a summary:

```python
import re

with open('backend/logs/app.log', 'r') as f:
    logs = f.read()

# Extract all timing logs
pattern = r'⏱️  \[(.*?)\] Completed in ([\d.]+)s'
matches = re.findall(pattern, logs)

# Group by operation
from collections import defaultdict
timings = defaultdict(list)
for operation, duration in matches:
    timings[operation].append(float(duration))

# Calculate averages
for operation, durations in timings.items():
    avg = sum(durations) / len(durations)
    print(f"{operation}: {avg:.3f}s avg ({len(durations)} calls)")
```

---

## Analyzing Results

### **Expected Output Breakdown** (for 1 PDF with 5 images, 5 tables):

| Operation Category           | Expected Time  | Operations                        |
| ---------------------------- | -------------- | --------------------------------- |
| **Storage (Supabase/MinIO)** | ~2-4s          | Document, chunks, images, tables  |
| **Vector Storage (Qdrant)**  | ~0.6-1s        | Text, table, image vectors        |
| **Indexing (Elasticsearch)** | ~0.6-0.8s      | Text and table chunks             |
| **Embeddings**               | ~1.1-1.4s      | Text, table, image embeddings     |
| **Vision (Captioning)**      | ~2.5-3s        | 5 images × ~0.5s each             |
| **Graph (Neo4j)**            | ~0.2s          | Knowledge graph building          |
| **Processing**               | ~0.05s         | Text chunking, table processing   |
| **Extraction**               | ~8.1s          | Table/image extraction (parallel) |
| **Overhead/Other**           | Remaining time | Coordination, network, etc.       |

---

## Troubleshooting

### **If a specific operation is slow:**

1. **Storage Operations (>0.5s per operation)**

   - Check network latency to Supabase
   - Check database connection pool size
   - Consider batch operations

2. **Vector Storage (>0.2s per batch)**

   - Check Qdrant connection
   - Verify batch size is optimal
   - Check network latency

3. **Vision Processing (>1s per image)**

   - Verify GPU is being used
   - Check model loading time
   - Consider batching multiple images

4. **Elasticsearch Indexing (>0.5s)**
   - Check Elasticsearch health
   - Verify bulk indexing is working
   - Check network latency

---

## Next Steps

With these detailed logs, you can now:

1. **Identify the exact source of overhead**
2. **Optimize specific slow operations**
3. **Track performance improvements**
4. **Create performance benchmarks**
5. **Monitor production performance**

The logs will help you determine if the overhead is from:

- Database operations (Supabase inserts/updates)
- Vector storage (Qdrant upserts)
- Network latency
- Storage operations (MinIO, Supabase storage)
- Elasticsearch indexing
- Or something else entirely

---

## Summary

✅ **Implemented comprehensive timing logs for all operations**  
✅ **No performance impact (minimal overhead)**  
✅ **Easy to read and analyze**  
✅ **Covers all major pipeline stages**  
✅ **Helps identify bottlenecks and overhead**

**Status**: Ready for testing. Ingest a document and review the logs to see exactly where time is spent!
