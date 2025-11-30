# Prometheus Queries Reference

Complete reference guide for querying metrics from the Multimodal RAG application in Prometheus.

## Table of Contents

1. [Query Processing Metrics](#query-processing-metrics)
2. [Individual Retriever Metrics](#individual-retriever-metrics)
3. [Embedding Metrics](#embedding-metrics)
4. [Answer Generation Metrics](#answer-generation-metrics)
5. [Document Processing Metrics](#document-processing-metrics)
6. [HTTP/API Metrics](#httpapi-metrics)
7. [Comparison Queries](#comparison-queries)

---

## Query Processing Metrics

### Total Query Processing Time

**Average query processing time (works with single data point):**

```promql
query_processing_duration_seconds_sum / query_processing_duration_seconds_count
```

**Average query processing time (time series, requires multiple queries):**

```promql
rate(query_processing_duration_seconds_sum[5m]) / rate(query_processing_duration_seconds_count[5m])
```

**P95 query processing time:**

```promql
histogram_quantile(0.95, rate(query_processing_duration_seconds_bucket[5m]))
```

**P99 query processing time:**

```promql
histogram_quantile(0.99, rate(query_processing_duration_seconds_bucket[5m]))
```

**Total queries processed:**

```promql
queries_processed_total
```

**Queries processed by status:**

```promql
queries_processed_total
```

**Query processing count:**

```promql
query_processing_duration_seconds_count
```

---

## Individual Retriever Metrics

### Sparse Retriever (BM25)

**Average sparse retrieval duration (works with single data point):**

```promql
retrieval_duration_seconds_sum{retrieval_type="sparse"} / retrieval_duration_seconds_count{retrieval_type="sparse"}
```

**Average sparse retrieval duration (time series):**

```promql
rate(retrieval_duration_seconds_sum{retrieval_type="sparse"}[5m]) / rate(retrieval_duration_seconds_count{retrieval_type="sparse"}[5m])
```

**P95 sparse retrieval duration:**

```promql
histogram_quantile(0.95, rate(retrieval_duration_seconds_bucket{retrieval_type="sparse"}[5m]))
```

**Sparse chunks retrieved (total):**

```promql
chunks_retrieved_total{retrieval_type="sparse"}
```

**Average chunks per sparse query:**

```promql
chunks_retrieved_per_query_sum{retrieval_type="sparse"} / chunks_retrieved_per_query_count{retrieval_type="sparse"}
```

**Sparse retrieval count:**

```promql
retrieval_duration_seconds_count{retrieval_type="sparse"}
```

---

### Dense Retriever (Vector Text)

**Average dense retrieval duration (works with single data point):**

```promql
retrieval_duration_seconds_sum{retrieval_type="dense"} / retrieval_duration_seconds_count{retrieval_type="dense"}
```

**Average dense retrieval duration (time series):**

```promql
rate(retrieval_duration_seconds_sum{retrieval_type="dense"}[5m]) / rate(retrieval_duration_seconds_count{retrieval_type="dense"}[5m])
```

**P95 dense retrieval duration:**

```promql
histogram_quantile(0.95, rate(retrieval_duration_seconds_bucket{retrieval_type="dense"}[5m]))
```

**Dense chunks retrieved (total):**

```promql
chunks_retrieved_total{retrieval_type="dense"}
```

**Average chunks per dense query:**

```promql
chunks_retrieved_per_query_sum{retrieval_type="dense"} / chunks_retrieved_per_query_count{retrieval_type="dense"}
```

**Dense retrieval count:**

```promql
retrieval_duration_seconds_count{retrieval_type="dense"}
```

---

### Table Retriever

**Average table retrieval duration (works with single data point):**

```promql
retrieval_duration_seconds_sum{retrieval_type="table"} / retrieval_duration_seconds_count{retrieval_type="table"}
```

**Average table retrieval duration (time series):**

```promql
rate(retrieval_duration_seconds_sum{retrieval_type="table"}[5m]) / rate(retrieval_duration_seconds_count{retrieval_type="table"}[5m])
```

**P95 table retrieval duration:**

```promql
histogram_quantile(0.95, rate(retrieval_duration_seconds_bucket{retrieval_type="table"}[5m]))
```

**Table chunks retrieved (total):**

```promql
chunks_retrieved_total{retrieval_type="table"}
```

**Average chunks per table query:**

```promql
chunks_retrieved_per_query_sum{retrieval_type="table"} / chunks_retrieved_per_query_count{retrieval_type="table"}
```

**Table retrieval count:**

```promql
retrieval_duration_seconds_count{retrieval_type="table"}
```

---

### Image Retriever

**Average image retrieval duration (works with single data point):**

```promql
retrieval_duration_seconds_sum{retrieval_type="image"} / retrieval_duration_seconds_count{retrieval_type="image"}
```

**Average image retrieval duration (time series):**

```promql
rate(retrieval_duration_seconds_sum{retrieval_type="image"}[5m]) / rate(retrieval_duration_seconds_count{retrieval_type="image"}[5m])
```

**P95 image retrieval duration:**

```promql
histogram_quantile(0.95, rate(retrieval_duration_seconds_bucket{retrieval_type="image"}[5m]))
```

**Image chunks retrieved (total):**

```promql
chunks_retrieved_total{retrieval_type="image"}
```

**Average chunks per image query:**

```promql
chunks_retrieved_per_query_sum{retrieval_type="image"} / chunks_retrieved_per_query_count{retrieval_type="image"}
```

**Image retrieval count:**

```promql
retrieval_duration_seconds_count{retrieval_type="image"}
```

---

### Hybrid Retriever (Overall)

**Average hybrid retrieval duration (works with single data point):**

```promql
retrieval_duration_seconds_sum{retrieval_type="hybrid"} / retrieval_duration_seconds_count{retrieval_type="hybrid"}
```

**Average hybrid retrieval duration (time series):**

```promql
rate(retrieval_duration_seconds_sum{retrieval_type="hybrid"}[5m]) / rate(retrieval_duration_seconds_count{retrieval_type="hybrid"}[5m])
```

**P95 hybrid retrieval duration:**

```promql
histogram_quantile(0.95, rate(retrieval_duration_seconds_bucket{retrieval_type="hybrid"}[5m]))
```

**Hybrid chunks retrieved (total):**

```promql
chunks_retrieved_total{retrieval_type="hybrid"}
```

**Average chunks per hybrid query:**

```promql
chunks_retrieved_per_query_sum{retrieval_type="hybrid"} / chunks_retrieved_per_query_count{retrieval_type="hybrid"}
```

**Hybrid merge duration (average, works with single data point):**

```promql
hybrid_merge_duration_seconds_sum / hybrid_merge_duration_seconds_count
```

**Hybrid merge duration (time series):**

```promql
rate(hybrid_merge_duration_seconds_sum[5m]) / rate(hybrid_merge_duration_seconds_count[5m])
```

---

## Embedding Metrics

### Query Embedding Time

**Text embedding duration (average, works with single data point):**

```promql
text_embedding_duration_seconds_sum / text_embedding_duration_seconds_count
```

**Text embedding duration (time series):**

```promql
rate(text_embedding_duration_seconds_sum[5m]) / rate(text_embedding_duration_seconds_count[5m])
```

**P95 text embedding duration:**

```promql
histogram_quantile(0.95, rate(text_embedding_duration_seconds_bucket[5m]))
```

**Text embeddings generated (total):**

```promql
text_embeddings_generated_total
```

**Text embedding batch size (average):**

```promql
rate(text_embedding_batch_size_sum[5m]) / rate(text_embedding_batch_size_count[5m])
```

### Image Embedding Time

**Image embedding duration (average):**

```promql
rate(image_embedding_duration_seconds_sum[5m]) / rate(image_embedding_duration_seconds_count[5m])
```

**Image embeddings generated (total):**

```promql
image_embeddings_generated_total
```

**Image embedding batch size (average):**

```promql
rate(image_embedding_batch_size_sum[5m]) / rate(image_embedding_batch_size_count[5m])
```

---

## Answer Generation Metrics

### Answer Generation Duration

**Answer generation duration (average, works with single data point):**

```promql
answer_generation_duration_seconds_sum / answer_generation_duration_seconds_count
```

**Answer generation duration by model (average, time series):**

```promql
rate(answer_generation_duration_seconds_sum[5m]) / rate(answer_generation_duration_seconds_count[5m])
```

**P95 answer generation time:**

```promql
histogram_quantile(0.95, rate(answer_generation_duration_seconds_bucket[5m]))
```

**P95 answer generation time by model:**

```promql
histogram_quantile(0.95, rate(answer_generation_duration_seconds_bucket[5m])) by (model)
```

### Time to First Token (TTFT)

**TTFT (average, works with single data point):**

```promql
answer_generation_ttft_seconds_sum / answer_generation_ttft_seconds_count
```

**TTFT by model (average, time series):**

```promql
rate(answer_generation_ttft_seconds_sum[5m]) / rate(answer_generation_ttft_seconds_count[5m])
```

**P95 TTFT:**

```promql
histogram_quantile(0.95, rate(answer_generation_ttft_seconds_bucket[5m]))
```

**P95 TTFT by model:**

```promql
histogram_quantile(0.95, rate(answer_generation_ttft_seconds_bucket[5m])) by (model)
```

### Answers Generated

**Total answers generated:**

```promql
answers_generated_total
```

**Answers generated by model and status:**

```promql
answers_generated_total
```

**Answers generated by status:**

```promql
sum(answers_generated_total) by (status)
```

### Token Usage

**Total tokens used:**

```promql
tokens_used_total
```

**Token usage by type (input/output/total):**

```promql
tokens_used_total
```

**Token usage by model:**

```promql
sum(tokens_used_total) by (model)
```

**Token usage by type and model:**

```promql
sum(tokens_used_total) by (model, type)
```

**Total tokens used (time series):**

```promql
sum(rate(tokens_used_total[5m])) by (type)
```

---

## Document Processing Metrics

### Documents Ingested

**Total documents ingested:**

```promql
documents_ingested_total
```

**Documents ingested by file type:**

```promql
documents_ingested_total
```

**Documents ingested by status:**

```promql
sum(documents_ingested_total) by (status)
```

**Documents ingested by file type and status:**

```promql
documents_ingested_total
```

### Document Ingestion Duration

**Document ingestion duration (average):**

```promql
rate(document_ingestion_duration_seconds_sum[5m]) / rate(document_ingestion_duration_seconds_count[5m])
```

**Document ingestion duration by file type:**

```promql
rate(document_ingestion_duration_seconds_sum[5m]) / rate(document_ingestion_duration_seconds_count[5m])
```

**P95 document ingestion duration:**

```promql
histogram_quantile(0.95, rate(document_ingestion_duration_seconds_bucket[5m]))
```

### Chunks Created

**Total chunks created:**

```promql
chunks_created_total
```

**Chunks created by document type:**

```promql
chunks_created_total
```

**Average chunks per document:**

```promql
rate(chunks_created_per_document_sum[5m]) / rate(chunks_created_per_document_count[5m])
```

**Average chunks per document by file type:**

```promql
rate(chunks_created_per_document_sum[5m]) / rate(chunks_created_per_document_count[5m])
```

### Document Processing Errors

**Total document processing errors:**

```promql
document_processing_errors_total
```

**Document processing errors by error type:**

```promql
document_processing_errors_total
```

**Document processing errors by file type:**

```promql
sum(document_processing_errors_total) by (file_type)
```

**Document processing errors by error type and file type:**

```promql
document_processing_errors_total
```

---

## HTTP/API Metrics

### HTTP Requests

**Total HTTP requests:**

```promql
http_requests_total
```

**HTTP requests by endpoint:**

```promql
sum(http_requests_total) by (endpoint)
```

**HTTP requests by method:**

```promql
sum(http_requests_total) by (method)
```

**HTTP requests by status code:**

```promql
sum(http_requests_total) by (status_code)
```

**HTTP requests by endpoint and status:**

```promql
sum(http_requests_total) by (endpoint, status_code)
```

### HTTP Request Duration

**HTTP request duration (average):**

```promql
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])
```

**HTTP request duration by endpoint:**

```promql
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])
```

**P95 HTTP request duration:**

```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

**P95 HTTP request duration by endpoint:**

```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) by (endpoint)
```

**P99 HTTP request duration:**

```promql
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

### HTTP Request/Response Sizes

**Average HTTP request size:**

```promql
rate(http_request_size_bytes_sum[5m]) / rate(http_request_size_bytes_count[5m])
```

**Average HTTP response size:**

```promql
rate(http_response_size_bytes_sum[5m]) / rate(http_response_size_bytes_count[5m])
```

### Error Rate

**HTTP error rate (5xx errors):**

```promql
sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

**HTTP error percentage:**

```promql
(sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))) * 100
```

---

## Comparison Queries

### All Retrieval Types Comparison

**All retrieval types (average duration):**

```promql
rate(retrieval_duration_seconds_sum[5m]) / rate(retrieval_duration_seconds_count[5m])
```

**All retrieval types by type (average duration):**

```promql
sum(rate(retrieval_duration_seconds_sum[5m])) by (retrieval_type) / sum(rate(retrieval_duration_seconds_count[5m])) by (retrieval_type)
```

**All chunks retrieved by type:**

```promql
chunks_retrieved_total
```

**Retrieval count by type:**

```promql
retrieval_duration_seconds_count
```

**All retrieval types P95 comparison:**

```promql
histogram_quantile(0.95, rate(retrieval_duration_seconds_bucket[5m])) by (retrieval_type)
```

### End-to-End Query Breakdown

**Complete query breakdown (all components):**

```promql
# Query processing time
rate(query_processing_duration_seconds_sum[5m]) / rate(query_processing_duration_seconds_count[5m])

# Retrieval time
rate(retrieval_duration_seconds_sum{retrieval_type="hybrid"}[5m]) / rate(retrieval_duration_seconds_count{retrieval_type="hybrid"}[5m])

# Answer generation time
rate(answer_generation_duration_seconds_sum[5m]) / rate(answer_generation_duration_seconds_count[5m])
```

---

## Usage Tips

### Working with Single Data Points

When you have only one query executed, use queries without `rate()`:

- ✅ `metric_sum / metric_count` - Works with single data point
- ❌ `rate(metric_sum[5m]) / rate(metric_count[5m])` - Returns NaN with single data point

### Working with Time Series

After multiple queries, use `rate()` for time series:

- ✅ `rate(metric_sum[5m]) / rate(metric_count[5m])` - Shows trends over time
- ✅ `histogram_quantile(0.95, rate(metric_bucket[5m]))` - Shows percentiles

### Common Patterns

**Average calculation:**

```promql
sum / count
# or for time series:
rate(sum[5m]) / rate(count[5m])
```

**Percentile calculation:**

```promql
histogram_quantile(0.95, rate(bucket[5m]))
# Change 0.95 to 0.50, 0.99, etc. for different percentiles
```

**Grouping by labels:**

```promql
sum(metric) by (label_name)
# or
rate(metric[5m]) by (label_name)
```

---

## Quick Reference

### Most Common Queries

1. **Query processing time:**

   ```promql
   query_processing_duration_seconds_sum / query_processing_duration_seconds_count
   ```

2. **Retrieval time (hybrid):**

   ```promql
   retrieval_duration_seconds_sum{retrieval_type="hybrid"} / retrieval_duration_seconds_count{retrieval_type="hybrid"}
   ```

3. **Individual retriever times:**

   ```promql
   # Sparse
   retrieval_duration_seconds_sum{retrieval_type="sparse"} / retrieval_duration_seconds_count{retrieval_type="sparse"}

   # Dense
   retrieval_duration_seconds_sum{retrieval_type="dense"} / retrieval_duration_seconds_count{retrieval_type="dense"}

   # Table
   retrieval_duration_seconds_sum{retrieval_type="table"} / retrieval_duration_seconds_count{retrieval_type="table"}

   # Image
   retrieval_duration_seconds_sum{retrieval_type="image"} / retrieval_duration_seconds_count{retrieval_type="image"}
   ```

4. **Answer generation time:**

   ```promql
   answer_generation_duration_seconds_sum / answer_generation_duration_seconds_count
   ```

5. **TTFT:**

   ```promql
   answer_generation_ttft_seconds_sum / answer_generation_ttft_seconds_count
   ```

6. **Chunks retrieved:**

   ```promql
   chunks_retrieved_total
   ```

7. **Token usage:**
   ```promql
   tokens_used_total
   ```

---

## Accessing Prometheus

- **Prometheus UI:** `http://localhost:9091`
- **Metrics Endpoint:** `http://localhost:8000/metrics`
- **Grafana:** `http://localhost:3001` (admin/admin)

---

## Notes

- All duration metrics are in **seconds**
- Histogram metrics require using `sum` and `count` for averages, or `histogram_quantile()` for percentiles
- Use `rate()` for time series analysis (requires multiple data points)
- Use direct `sum / count` for single data point analysis
- Label filters can be added to any query: `{label_name="value"}`
