# Qdrant Initialization Script

This script initializes Qdrant vector database collections for a multimodal RAG (Retrieval-Augmented Generation) system.

## Overview

The script creates collections for different types of multimodal content:

### Text Collections
- **Collection**: `text_chunks`
- **Vector Dimensions**: 384-768 (configurable based on embedding model)
- **Payload Fields**: `chunk_id`, `document_id`, `text`, `metadata`

### Table Collections
- **Collection**: `table_chunks`
- **Vector Dimensions**: 768 (matching text embeddings)
- **Payload Fields**: `chunk_id`, `document_id`, `table_data`, `table_markdown`, `metadata`

### Image Collections
- **Collection**: `image_chunks`
- **Vector Dimensions**: 512 (CLIP base), 768 (SigLIP base), or 1024 (SigLIP large)
- **Payload Fields**: `chunk_id`, `document_id`, `image_path`, `caption`, `image_type`, `metadata`

## Prerequisites

1. **Qdrant Server**: Ensure Qdrant is running and accessible
   ```bash
   # Using Docker Compose (recommended)
   docker-compose up -d qdrant

   # Verify Qdrant is healthy
   curl http://localhost:6333/health
   ```

2. **Python Dependencies**:
   ```bash
   pip install qdrant-client
   ```

## Quick Start

### Initialize All Collections
```bash
cd backend/scripts/qdrant
python init_qdrant.py --all
```

### Initialize Collections by Type
```bash
# Text collections only
python init_qdrant.py --collection text_chunks

# Table collections only
python init_qdrant.py --collection table_chunks

# Image collections only
python init_qdrant.py --collection image_chunks --vector-size 1024
```

### Initialize Multimodal Collections (Tables + Images)
```bash
python init_qdrant.py --multimodal
```

## Command Line Options

### Basic Options
- `--url`: Qdrant server URL (default: `http://localhost:6333`)
- `--collection`: Collection name to create (`text_chunks`, `table_chunks`, `image_chunks`)
- `--vector-size`: Vector dimension (384, 512, 768, 1024)
- `--recreate`: Delete existing collection and create new one

### Mode Options
- `--all`: Create all collections (text_chunks, table_chunks, image_chunks)
- `--multimodal`: Create multimodal collections only (table_chunks, image_chunks)
- `--verify`: Verify existing collections configuration (no changes made)

### Advanced Options
- `--image-vector-size`: Vector dimension for image_chunks (512, 768, 1024, default: 1024)

## Detailed Usage Examples

### Text Collections
```bash
# Default configuration (768 dimensions for e5-base-v2/all-mpnet-base-v2)
python init_qdrant.py --collection text_chunks

# Smaller model (384 dimensions for all-MiniLM-L6-v2)
python init_qdrant.py --collection text_chunks --vector-size 384

# Recreate existing collection
python init_qdrant.py --collection text_chunks --recreate
```

### Table Collections
```bash
# Default configuration (768 dimensions)
python init_qdrant.py --collection table_chunks

# Custom vector size
python init_qdrant.py --collection table_chunks --vector-size 768
```

### Image Collections
```bash
# CLIP base model (512 dimensions)
python init_qdrant.py --collection image_chunks --vector-size 512

# SigLIP base model (768 dimensions)
python init_qdrant.py --collection image_chunks --vector-size 768

# SigLIP large model (1024 dimensions) - Default
python init_qdrant.py --collection image_chunks --vector-size 1024
```

### Initialize All Collections
```bash
# All collections with default settings
python init_qdrant.py --all

# Custom text vector size, default image size
python init_qdrant.py --all --vector-size 384

# Custom image vector size
python init_qdrant.py --all --image-vector-size 512

# Recreate all collections
python init_qdrant.py --all --recreate
```

### Initialize Multimodal Collections Only
```bash
# Default image vector size (1024)
python init_qdrant.py --multimodal

# CLIP base for images
python init_qdrant.py --multimodal --image-vector-size 512

# Recreate multimodal collections
python init_qdrant.py --multimodal --recreate
```

### Verification
```bash
# Verify all collections are properly configured
python init_qdrant.py --verify

# Verify with custom Qdrant URL
python init_qdrant.py --verify --url http://custom-qdrant:6333
```

## Configuration

The script automatically tries to load configuration from the backend settings. You can also override settings using environment variables:

```bash
# Set Qdrant URL via environment variable
export QDRANT_URL="http://your-qdrant-server:6333"
python init_qdrant.py --collection text_chunks

# Set vector size via environment variable
export QDRANT_VECTOR_SIZE=384
python init_qdrant.py --collection text_chunks
```

## Performance Optimizations

The script includes performance optimizations:

- **Image Chunks**: Uses aggressively optimized HNSW settings (`m=4`, `ef_construct=64`) and scalar quantization (INT8) for ultra-fast image search
- **Text/Table Chunks**: Uses balanced HNSW settings (`m=12`, `ef_construct=128`) for good speed-accuracy tradeoff

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   ```bash
   # Make sure Qdrant is running
   docker-compose up -d qdrant

   # Check Qdrant health
   curl http://localhost:6333/health

   # Check Qdrant logs
   docker-compose logs qdrant
   ```

2. **Collection Already Exists**
   ```bash
   # Use --recreate to delete and recreate
   python init_qdrant.py --collection text_chunks --recreate
   ```

3. **Import Error**
   ```bash
   # Install required dependencies
   pip install qdrant-client
   ```

4. **Vector Size Mismatch**
   ```bash
   # Use --verify to check current configuration
   python init_qdrant.py --verify

   # Recreate with correct vector size
   python init_qdrant.py --collection image_chunks --vector-size 1024 --recreate
   ```

## File Structure

```
backend/scripts/qdrant/
├── init_qdrant.py    # Main initialization script
└── README.md         # This documentation
```

## Integration

This script is designed to work with the multimodal RAG backend. After initialization, the collections will be ready to receive:

- Text embeddings from models like `e5-base-v2` or `all-mpnet-base-v2`
- Table data embeddings (768 dimensions)
- Image embeddings from CLIP or SigLIP models

The collections are optimized for cosine similarity distance metric and include appropriate indexing configurations for fast retrieval.
