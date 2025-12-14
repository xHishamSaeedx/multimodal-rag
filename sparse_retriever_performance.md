# Sparse Retriever Performance Metrics Report

_Report generated on: December 14, 2025_
_Data from: BM25/Elasticsearch keyword search performance_

## Executive Summary

This report focuses specifically on **Sparse Retriever** (BM25 keyword search) performance metrics. Sparse retrieval is independent of embedding model changes and provides consistent keyword-based search capabilities.

## 📊 Sparse Retriever Performance Overview

### Retrieval Performance

#### BM25/Elasticsearch Search

| Metric                      | Average Time | Total Queries | Notes                                        |
| --------------------------- | ------------ | ------------- | -------------------------------------------- |
| **Sparse Retriever** (BM25) | 113.2 ms     | 5             | Keyword-based search with stable performance |

### Quality Metrics

#### Sparse Retriever Relevance

| Metric                         | Average Score | Scale                  | Total Queries | Notes                              |
| ------------------------------ | ------------- | ---------------------- | ------------- | ---------------------------------- |
| **Sparse Retriever Relevance** | 0.991         | 0-1 (higher is better) | 6             | Near-perfect BM25 keyword matching |

## 📈 Detailed Sparse Retriever Metrics

### Sparse Retriever (BM25 Search)

- **Raw Sum**: 0.5661935806274414 seconds
- **Query Count**: 5
- **Average**: 0.11323871612548828 seconds (113.2 ms)
- **Performance**: ⭐⭐⭐ Good (under 200ms acceptable)

### Sparse Retriever Relevance Scores

#### Previous Data (Historical)

- **Raw Sum**: 5.948896462
- **Query Count**: 6 (includes historical data)
- **Average Score**: 0.9915 (99.2%)
- **Quality Assessment**: 🟢 Excellent - near-perfect relevance

#### Latest Data (Past 30 minutes)

- **Raw Rate**: 0.9858045727777778 (98.6%)
- **Query Count**: 5 (past 30 minutes)
- **Quality Assessment**: 🟢 Excellent - consistent near-perfect relevance
- **Model**: BM25 implementation (unchanged)

### Comparison Table

| Retrieval Type       | Average Relevance | Quality Rating | Notes                      |
| -------------------- | ----------------- | -------------- | -------------------------- |
| **Sparse Retriever** | 99.2%             | 🟢 Excellent   | Excellent keyword matching |

## 🔍 Sparse Retriever Analysis

### Strengths

- **Keyword matching**: Near-perfect relevance (99.2%) - excellent for exact term matching
- **Consistent performance**: Stable 113ms response time
- **Independent of embedding models**: Performance unaffected by dense retriever model changes

### Characteristics

- **Retrieval method**: BM25 keyword search using Elasticsearch
- **Use case**: Best for queries with specific keywords or technical terms
- **Complement**: Works alongside dense retrieval for hybrid search

## 📋 Sparse Retriever Technical Details

### Implementation

- **Engine**: Elasticsearch with BM25 similarity
- **Configuration**: Standard BM25 parameters (k1=1.2, b=0.75)
- **Indexing**: Keyword-based inverted index
- **Query processing**: Term frequency analysis with document length normalization

### Metrics Collection

- **Source**: Prometheus metrics endpoint (`http://localhost:9091`)
- **Metrics**: `retrieval_duration_seconds{retrieval_type="sparse"}`, `average_retrieval_relevance_per_query{retrieval_type="sparse"}`
- **Instance**: `backend-1`
- **Service**: `multimodal-rag-backend`

### Prometheus Queries for Monitoring

**Sparse Retriever Relevance**:

```promql
rate(average_retrieval_relevance_per_query_sum{retrieval_type="sparse"}[30m]) /
rate(average_retrieval_relevance_per_query_count{retrieval_type="sparse"}[30m])
```

**Sparse Retriever Retrieval Time**:

```promql
rate(retrieval_duration_seconds_sum{retrieval_type="sparse"}[30m]) /
rate(retrieval_duration_seconds_count{retrieval_type="sparse"}[30m])
```

---

_This report contains sparse retriever metrics extracted from the main performance report. Sparse retrieval performance is independent of embedding model changes._
