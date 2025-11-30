"""
Prometheus metrics for the Multimodal RAG application.

This module defines and exports all Prometheus metrics used throughout the application.
Metrics are organized by category:
- API metrics (request rate, latency, errors)
- Ingestion metrics (documents processed, chunks created)
- Retrieval metrics (query latency, retrieval counts)
- Embedding metrics (embedding generation time, batch size)
- Storage metrics (Qdrant/Elasticsearch query latency)
- Business metrics (queries per day, documents ingested)
"""
from prometheus_client import Counter, Histogram, Gauge, Summary
from prometheus_client import REGISTRY

# ============================================================================
# API Metrics
# ============================================================================

# HTTP request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

# ============================================================================
# Ingestion Metrics
# ============================================================================

# Document ingestion metrics
documents_ingested_total = Counter(
    'documents_ingested_total',
    'Total number of documents ingested',
    ['file_type', 'status']
)

document_ingestion_duration_seconds = Histogram(
    'document_ingestion_duration_seconds',
    'Document ingestion duration in seconds',
    ['file_type'],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)

chunks_created_total = Counter(
    'chunks_created_total',
    'Total number of chunks created',
    ['document_type']
)

chunks_created_per_document = Histogram(
    'chunks_created_per_document',
    'Number of chunks created per document',
    ['file_type'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

document_processing_errors_total = Counter(
    'document_processing_errors_total',
    'Total number of document processing errors',
    ['error_type', 'file_type']
)

# ============================================================================
# Retrieval Metrics
# ============================================================================

# Query processing metrics
queries_processed_total = Counter(
    'queries_processed_total',
    'Total number of queries processed',
    ['status']
)

query_processing_duration_seconds = Histogram(
    'query_processing_duration_seconds',
    'Total query processing duration (retrieval + generation)',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Retrieval stage metrics
retrieval_duration_seconds = Histogram(
    'retrieval_duration_seconds',
    'Retrieval duration in seconds',
    ['retrieval_type'],  # 'hybrid', 'sparse', 'dense', 'table', 'image'
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

chunks_retrieved_total = Counter(
    'chunks_retrieved_total',
    'Total number of chunks retrieved',
    ['retrieval_type']
)

chunks_retrieved_per_query = Histogram(
    'chunks_retrieved_per_query',
    'Number of chunks retrieved per query',
    ['retrieval_type'],
    buckets=[1, 5, 10, 20, 50, 100]
)

# Hybrid retrieval metrics
hybrid_merge_duration_seconds = Histogram(
    'hybrid_merge_duration_seconds',
    'Hybrid retrieval merge duration in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

# Reranking metrics
reranking_duration_seconds = Histogram(
    'reranking_duration_seconds',
    'Reranking duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

chunks_reranked_total = Counter(
    'chunks_reranked_total',
    'Total number of chunks reranked'
)

# ============================================================================
# Embedding Metrics
# ============================================================================

# Text embedding metrics
text_embeddings_generated_total = Counter(
    'text_embeddings_generated_total',
    'Total number of text embeddings generated'
)

text_embedding_duration_seconds = Histogram(
    'text_embedding_duration_seconds',
    'Text embedding generation duration in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

text_embedding_batch_size = Histogram(
    'text_embedding_batch_size',
    'Text embedding batch size',
    buckets=[1, 5, 10, 20, 32, 50, 64, 100, 128]
)

# Image embedding metrics
image_embeddings_generated_total = Counter(
    'image_embeddings_generated_total',
    'Total number of image embeddings generated'
)

image_embedding_duration_seconds = Histogram(
    'image_embedding_duration_seconds',
    'Image embedding generation duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

image_embedding_batch_size = Histogram(
    'image_embedding_batch_size',
    'Image embedding batch size',
    buckets=[1, 5, 10, 20, 32, 50, 64, 100]
)

# ============================================================================
# Storage Metrics
# ============================================================================

# Qdrant (Vector DB) metrics
qdrant_query_duration_seconds = Histogram(
    'qdrant_query_duration_seconds',
    'Qdrant query duration in seconds',
    ['operation'],  # 'search', 'upsert', 'delete', etc.
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

qdrant_queries_total = Counter(
    'qdrant_queries_total',
    'Total number of Qdrant queries',
    ['operation', 'status']
)

qdrant_connection_errors_total = Counter(
    'qdrant_connection_errors_total',
    'Total number of Qdrant connection errors'
)

# Elasticsearch (BM25) metrics
elasticsearch_query_duration_seconds = Histogram(
    'elasticsearch_query_duration_seconds',
    'Elasticsearch query duration in seconds',
    ['operation'],  # 'search', 'index', 'delete', etc.
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

elasticsearch_queries_total = Counter(
    'elasticsearch_queries_total',
    'Total number of Elasticsearch queries',
    ['operation', 'status']
)

elasticsearch_connection_errors_total = Counter(
    'elasticsearch_connection_errors_total',
    'Total number of Elasticsearch connection errors'
)

# MinIO (S3) metrics
minio_operation_duration_seconds = Histogram(
    'minio_operation_duration_seconds',
    'MinIO operation duration in seconds',
    ['operation'],  # 'put', 'get', 'delete', etc.
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

minio_operations_total = Counter(
    'minio_operations_total',
    'Total number of MinIO operations',
    ['operation', 'status']
)

# ============================================================================
# Generation (LLM) Metrics
# ============================================================================

# Answer generation metrics
answers_generated_total = Counter(
    'answers_generated_total',
    'Total number of answers generated',
    ['model', 'status']
)

answer_generation_duration_seconds = Histogram(
    'answer_generation_duration_seconds',
    'Answer generation duration in seconds',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

answer_generation_ttft_seconds = Histogram(
    'answer_generation_ttft_seconds',
    'Time to first token (TTFT) in seconds',
    ['model'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

tokens_used_total = Counter(
    'tokens_used_total',
    'Total number of tokens used',
    ['model', 'type']  # type: 'input', 'output', 'total'
)

# ============================================================================
# Business Metrics
# ============================================================================

# Active documents and chunks
active_documents = Gauge(
    'active_documents',
    'Number of active documents in the system'
)

active_chunks = Gauge(
    'active_chunks',
    'Number of active chunks in the system'
)

# Query quality metrics (if feedback is collected)
query_satisfaction_score = Histogram(
    'query_satisfaction_score',
    'Query satisfaction score (0-1)',
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# ============================================================================
# System Metrics
# ============================================================================

# Application uptime
application_uptime_seconds = Gauge(
    'application_uptime_seconds',
    'Application uptime in seconds'
)

# Active connections
active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

# ============================================================================
# Helper Functions
# ============================================================================

def get_all_metrics():
    """Get all registered metrics."""
    return REGISTRY

