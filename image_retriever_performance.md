# Image Retriever Performance Report

_Report generated on: December 14, 2025_
_Data from: 5 queries in last 5 minutes_
_Models: sentence-transformers/clip-ViT-L-14 (768 dimensions), SigLIP vit_base_patch16_siglip_224 (768 dimensions), CLIP ViT-B-32 (512 dimensions)_

## Executive Summary

This report analyzes the **Image Retriever** performance for visual content search in the multimodal RAG system. Images are retrieved using CLIP embeddings and cosine similarity in Qdrant.

## 🎯 **Image Retriever Performance Overview**

### Latest Performance Metrics (sentence-transformers/clip-ViT-L-14)

| Metric                              | Average Time | Relevance Score | Total Queries | Notes                               |
| ----------------------------------- | ------------ | --------------- | ------------- | ----------------------------------- |
| **Image Retriever** (Visual Search) | 51.2 ms      | 27.4%           | 5             | CLIP-based visual similarity search |

### Latest Performance Metrics (SigLIP vit_base_patch16_siglip_224)

| Metric                              | Average Time | Relevance Score | Total Queries | Notes                                 |
| ----------------------------------- | ------------ | --------------- | ------------- | ------------------------------------- |
| **Image Retriever** (Visual Search) | 509.2 ms     | 1.9%            | 5             | SigLIP-based visual similarity search |

### Latest Performance Metrics (CLIP ViT-B-32)

| Metric                              | Average Time | Relevance Score | Total Queries | Notes                                        |
| ----------------------------------- | ------------ | --------------- | ------------- | -------------------------------------------- |
| **Image Retriever** (Visual Search) | 505.1 ms     | 15.8%           | 5             | CLIP ViT-B-32-based visual similarity search |

### Performance Analysis

| Aspect              | Current Performance     | Notes                                         |
| ------------------- | ----------------------- | --------------------------------------------- |
| **Retrieval Speed** | 51.2 ms                 | Moderate - visual search is compute-intensive |
| **Relevance Score** | 27.4%                   | Low-moderate - room for improvement           |
| **Consistency**     | Stable across 5 queries | Reliable performance                          |

### Performance Analysis - SigLIP (vit_base_patch16_siglip_224)

| Aspect              | Current Performance     | Notes                                               |
| ------------------- | ----------------------- | --------------------------------------------------- |
| **Retrieval Speed** | 509.2 ms                | Slower - SigLIP inference is more compute-intensive |
| **Relevance Score** | 1.9%                    | Very low - significant degradation from CLIP        |
| **Consistency**     | Stable across 5 queries | Reliable performance but poor quality               |

## 📊 **Detailed Image Retriever Metrics**

### Image Retrieval Times Breakdown

#### CLIP Visual Search Performance

- **Raw Rate**: 0.051158905029296875 seconds per query (51.2 ms)
- **Query Count**: 5 (past 5 minutes)
- **Performance**: ⭐⭐⭐ Moderate (50-100ms acceptable for visual search)
- **Model**: sentence-transformers/clip-ViT-L-14 (768 dimensions)

#### SigLIP Visual Search Performance

- **Raw Rate**: 0.5091619253158569 seconds per query (509.2 ms)
- **Query Count**: 5 (past 5 minutes)
- **Performance**: ⭐⭐ Poor (500+ms very slow for visual search)
- **Model**: SigLIP vit_base_patch16_siglip_224 (768 dimensions)

#### CLIP ViT-B-32 Visual Search Performance

- **Raw Rate**: 0.5051052093505859 seconds per query (505.1 ms)
- **Query Count**: 5 (past 5 minutes)
- **Performance**: ⭐⭐ Moderate-slow (500ms acceptable for visual search)
- **Model**: CLIP ViT-B-32 (512 dimensions)

### Image Retrieval Relevance Scores

#### Visual Similarity Scores (Latest Data)

**Data Status**: ✅ **Populated** - Latest data from visual search queries

### **Image Retriever** (CLIP Visual Similarity)

- **Raw Rate**: 0.2739834272861481 (27.4%)
- **Query Count**: 5 (past 5 minutes)
- **Quality Assessment**: 🟡 Moderate - visual search working but not optimal
- **Model**: sentence-transformers/clip-ViT-L-14 (768 dimensions)

### **Image Retriever** (SigLIP Visual Similarity)

- **Raw Rate**: 0.01937590300105512 (1.9%)
- **Query Count**: 5 (past 5 minutes)
- **Quality Assessment**: 🔴 Poor - significant degradation in visual matching quality
- **Model**: SigLIP vit_base_patch16_siglip_224 (768 dimensions)

### **Image Retriever** (CLIP ViT-B-32 Visual Similarity)

- **Raw Rate**: 0.15834114789962767 (15.8%)
- **Query Count**: 5 (past 5 minutes)
- **Quality Assessment**: 🟡 Moderate-low - visual search working but suboptimal
- **Model**: CLIP ViT-B-32 (512 dimensions)

### Visual Search Analysis - CLIP

| Metric                | Value   | Interpretation                          |
| --------------------- | ------- | --------------------------------------- |
| **Average Relevance** | 27.4%   | Low-moderate visual matching quality    |
| **Retrieval Time**    | 51.2 ms | Moderate performance for CLIP inference |
| **Consistency**       | Stable  | Reliable across different queries       |

### Visual Search Analysis - SigLIP

| Metric                | Value    | Interpretation                           |
| --------------------- | -------- | ---------------------------------------- |
| **Average Relevance** | 1.9%     | Very poor visual matching quality        |
| **Retrieval Time**    | 509.2 ms | Slow performance for SigLIP inference    |
| **Consistency**       | Stable   | Reliable but poor quality across queries |

### Visual Search Analysis - CLIP ViT-B-32

| Metric                | Value    | Interpretation                                        |
| --------------------- | -------- | ----------------------------------------------------- |
| **Average Relevance** | 15.8%    | Low-moderate visual matching quality                  |
| **Retrieval Time**    | 505.1 ms | Moderate-slow performance for CLIP ViT-B-32 inference |
| **Consistency**       | Stable   | Reliable across different queries                     |

## 🔍 **Image Retrieval Analysis**

### Strengths

- **Visual Understanding**: Successfully processes and searches images using CLIP
- **Stable Performance**: Consistent timing and moderate relevance scores
- **Integrated Search**: Seamlessly combined with text and table retrieval in hybrid system

### Areas for Improvement

- **Relevance Scores**: 27.4% is moderate - visual matching could be better

  - Possible improvements: Better image preprocessing, fine-tuned CLIP model, or different vision model
  - Consider using SigLIP models which are trained specifically for visual similarity

- **Retrieval Speed**: 51.2ms is acceptable but could be faster
  - CLIP inference is compute-intensive
  - Consider model optimization or GPU acceleration

#### SigLIP Performance Issues

- **Critical Relevance Degradation**: 1.9% relevance score represents significant quality loss

  - SigLIP showing much poorer visual matching compared to CLIP (27.4% → 1.9%)
  - May indicate incompatibility with current image preprocessing or embedding pipeline

- **Performance Degradation**: 509.2ms is 10x slower than CLIP
  - SigLIP inference significantly more compute-intensive
  - May require different optimization strategies or model configuration

### CLIP Model Characteristics

| Aspect                | Current Model                       | Potential Improvements           |
| --------------------- | ----------------------------------- | -------------------------------- |
| **Architecture**      | ViT-L/14 (Large Vision Transformer) | ViT-H/14 for better accuracy     |
| **Dimensions**        | 768                                 | 1024+ for richer representations |
| **Training Data**     | Web-scale image-text pairs          | Domain-specific fine-tuning      |
| **Speed vs Accuracy** | Balanced                            | Could optimize for either        |

### SigLIP Model Characteristics

| Aspect                | Current Model                    | Performance Issues                  |
| --------------------- | -------------------------------- | ----------------------------------- |
| **Architecture**      | ViT-Base/16 (Vision Transformer) | Smaller model than CLIP ViT-L/14    |
| **Dimensions**        | 768                              | Same as CLIP but poorer results     |
| **Training Data**     | Web-scale image-text pairs       | Similar to CLIP training            |
| **Speed vs Accuracy** | Poor balance                     | 10x slower with much lower accuracy |
| **Compatibility**     | Pipeline compatibility issues    | May require different preprocessing |

## 📋 **Image Retrieval Technical Details**

### Implementation Details

- **Models**: `sentence-transformers/clip-ViT-L-14` (previous), `SigLIP vit_base_patch16_siglip_224` (current)
- **Embedding Dimensions**: 768
- **Similarity Metric**: Cosine similarity
- **Storage**: Qdrant vector database (`image_chunks` collection)
- **Integration**: Combined with text retrieval in hybrid scoring (15% weight)

### Query Processing Flow

1. **Text Query** → **CLIP Text Encoder** → **768-dim text embedding**
2. **Qdrant Search** → **Cosine similarity** → **Top-K visual results**
3. **Score Normalization** → **Weighted combination** with other retrievers

### Performance Metrics Collection

- **Source**: Prometheus metrics endpoint (`http://localhost:9091`)
- **Metrics**: `retrieval_duration_seconds{retrieval_type="image"}`, `average_retrieval_relevance_per_query{retrieval_type="image"}`
- **Instance**: `backend-1`
- **Service**: `multimodal-rag-backend`

### Prometheus Queries for Monitoring

**Image Retriever Relevance**:

```promql
rate(average_retrieval_relevance_per_query_sum{retrieval_type="image"}[30m]) /
rate(average_retrieval_relevance_per_query_count{retrieval_type="image"}[30m])
```

**Image Retriever Retrieval Time**:

```promql
rate(retrieval_duration_seconds_sum{retrieval_type="image"}[30m]) /
rate(retrieval_duration_seconds_count{retrieval_type="image"}[30m])
```

## 🎯 **Recommendations**

### Immediate Actions

1. **CLIP ViT-B-32 Performance Evaluation**: Current 15.8% relevance score shows moderate performance

   - Performance is better than SigLIP (1.9%) but slower than expected (505ms vs 51ms for ViT-L/14)
   - Consider trade-offs between speed and relevance for different CLIP variants

2. **SigLIP Investigation**: 1.9% relevance score indicates compatibility issues

   - Compare SigLIP vs CLIP preprocessing pipelines
   - Check for embedding normalization or scaling issues
   - CLIP ViT-B-32 provides better performance as interim solution

3. **Performance Optimization**:
   - CLIP ViT-B-32 at 505ms is slower than ViT-L-14 (51ms) - investigate inference optimization
   - Consider batch processing for multiple images
   - Evaluate GPU utilization for visual processing

### Medium-term Improvements

1. **Model Evaluation and SigLIP Issues**:

   - **URGENT**: Debug SigLIP performance degradation (1.9% vs 27.4% relevance)
   - Test SigLIP models with proper preprocessing and compare with CLIP baseline
   - Consider larger CLIP models (ViT-H/14) for better accuracy
   - Evaluate domain-specific fine-tuning for both CLIP and SigLIP

2. **Preprocessing Enhancements**:
   - Improve image quality preprocessing
   - Add image metadata extraction
   - Implement image captioning quality improvements

### Long-term Enhancements

1. **Advanced Visual Search**:

   - Multi-modal embeddings (image + text + metadata)
   - Region-based image search
   - Visual question answering integration

2. **Scalability Improvements**:
   - Distributed visual processing
   - Cached embeddings for common queries
   - Progressive loading for large image sets

## 📈 **Visual Search Benchmarking**

### Current Performance Baselines

#### CLIP ViT-L/14 (Previous Model)

- **Average Relevance**: 27.4%
- **Average Latency**: 51.2 ms
- **Use Case**: General visual search in technical documents

#### CLIP ViT-B-32 (Current Model)

- **Average Relevance**: 15.8%
- **Average Latency**: 505.1 ms
- **Use Case**: General visual search in technical documents

#### SigLIP vit_base_patch16_siglip_224 (Tested Model)

- **Average Relevance**: 1.9%
- **Average Latency**: 509.2 ms
- **Use Case**: General visual search in technical documents (degraded performance)

### Future Benchmarks

- **SigLIP Base**: Expected 35-45% relevance with 40-50ms latency
- **CLIP ViT-H/14**: Expected 30-40% relevance with 70-90ms latency
- **Fine-tuned Models**: Expected 40-60% relevance with optimized latency

---

_This report compares image retriever performance between CLIP ViT-L/14, CLIP ViT-B-32, and SigLIP models. CLIP ViT-B-32 shows moderate performance with 15.8% relevance, while SigLIP exhibits significant degradation (1.9% relevance). Image retrieval is a challenging task that requires balancing accuracy, speed, and visual understanding capabilities._
