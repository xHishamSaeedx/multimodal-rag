# Embedding Generation Performance Report

_Report generated on: December 14, 2025_
_Time range: Last 6 hours_
_Data source: Prometheus metrics (http://localhost:9091)_

## Executive Summary

This report analyzes the embedding generation performance for the multimodal RAG system, covering both text and image modalities over the last 6 hours of operation.

## 📊 Embedding Generation Performance Overview

### Average Generation Times (Last 6 Hours)

| Embedding Type      | Average Time | Performance Rating | Notes                                   |
| ------------------- | ------------ | ------------------ | --------------------------------------- |
| **Text Embedding**  | 225.3 ms     | ⭐⭐ Moderate      | Semantic text vector generation         |
| **Image Embedding** | 264.6 ms     | ⭐⭐ Moderate      | Visual feature extraction (CLIP/SigLIP) |

### Generation Volume (Measured over 6-hour window)

| Embedding Type      | Rate (per second) | Est. Total (6h window) | Notes                         |
| ------------------- | ----------------- | ---------------------- | ----------------------------- |
| **Text Embedding**  | 0.0030            | ~64 embeddings         | Steady text processing volume |
| **Image Embedding** | 0.0005            | ~11 embeddings         | Lower image processing volume |

## 📈 Detailed Embedding Metrics

### Text Embedding Performance

- **Average Generation Time**: 0.2252778820693493 seconds (225.3 ms)
- **Generation Rate**: 0.0029641439594901087 embeddings/second
- **Estimated Volume**: ~64 text embeddings in the measured 6-hour window
- **Performance Assessment**: Moderate - suitable for real-time text processing

### Image Embedding Performance

- **Average Generation Time**: 0.26456013592806726 seconds (264.6 ms)
- **Generation Rate**: 0.0005094412523051213 embeddings/second
- **Estimated Volume**: ~11 image embeddings in the measured 6-hour window
- **Performance Assessment**: Moderate - compute-intensive visual processing

### Performance Analysis

#### Text Embedding Characteristics

- **Latency**: 225ms average generation time
- **Throughput**: Low but steady generation rate
- **Use Case**: Semantic text understanding for retrieval
- **Model**: Likely E5/GTE/multilingual-e5-large based on system configuration

#### Image Embedding Characteristics

- **Latency**: 265ms average generation time (18% slower than text)
- **Throughput**: Very low generation rate (6x less frequent than text)
- **Use Case**: Visual similarity search and multimodal fusion
- **Model**: CLIP/SigLIP variants for visual feature extraction

## 🔍 Performance Insights

### Strengths

- **Stable Performance**: Consistent generation times across both modalities
- **Moderate Latency**: Both under 300ms, suitable for interactive applications
- **Balanced Processing**: Appropriate ratio of text vs image processing

### Areas for Optimization

- **Image Processing**: 265ms is relatively slow for visual embeddings
- **Throughput**: Low embedding generation rate suggests either:
  - Batch processing opportunities
  - Caching strategies for frequent embeddings
  - On-demand generation only

### Comparative Analysis

| Aspect               | Text Embedding | Image Embedding | Difference      |
| -------------------- | -------------- | --------------- | --------------- |
| **Average Latency**  | 225.3 ms       | 264.6 ms        | +39.3 ms (+17%) |
| **Processing Rate**  | 0.0030/sec     | 0.0005/sec      | 6x lower volume |
| **Est. Volume (6h)** | ~64 units      | ~11 units       | 6x lower volume |

## 📋 Technical Details

### Metrics Collection

- **Source**: Prometheus histogram metrics
- **Text Embedding Metric**: `text_embedding_duration_seconds`
- **Image Embedding Metric**: `image_embedding_duration_seconds`
- **Time Range**: 6 hours (21600 seconds)
- **Query Method**: Rate calculation using sum/count pattern

### Prometheus Queries Used

**Text Embedding Average Time:**

```promql
rate(text_embedding_duration_seconds_sum[6h]) /
rate(text_embedding_duration_seconds_count[6h])
```

**Image Embedding Average Time:**

```promql
rate(image_embedding_duration_seconds_sum[6h]) /
rate(image_embedding_duration_seconds_count[6h])
```

---

_This report provides real-time embedding generation performance metrics from the multimodal RAG system's Prometheus monitoring. Performance averages and rates are calculated over a 6-hour measurement window._
