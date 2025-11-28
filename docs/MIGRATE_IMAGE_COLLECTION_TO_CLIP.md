# Migrating Image Collection from SigLIP (1024 dim) to CLIP (768 dim)

This guide walks you through migrating the `image_chunks` Qdrant collection from 1024 dimensions (SigLIP large) to 768 dimensions (CLIP large).

## Overview

**What changed:**

- Image embedding model: SigLIP large (1024 dim) → CLIP large (768 dim)
- Text-to-image search: Now uses unified CLIP model (same semantic space)
- Collection dimension: 1024 → 768

**Why:**

- Unified embedding space for text and images (better semantic alignment)
- No dimension mismatch issues
- Better text-to-image retrieval accuracy

## Step 1: Check Current Collection

First, verify the current state of your `image_chunks` collection:

```bash
# Check collection info
python scripts/init_qdrant.py --collection image_chunks
```

This will show:

- Current vector size (should be 1024)
- Number of points (images) in the collection

## Step 2: Backup (Optional but Recommended)

If you want to keep existing image embeddings, you can export them first:

```python
# Create a backup script (optional)
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Export all points
points = []
for point in client.scroll(collection_name="image_chunks", limit=1000):
    points.append(point)

# Save to file (you'll need to implement this based on your needs)
# This is optional - you can also just re-embed images after recreating the collection
```

**Note:** Since we're changing dimensions, you'll need to re-embed all images anyway. The backup is mainly for metadata preservation.

## Step 3: Recreate Collection with 768 Dimensions

### Option A: Using the init script (Recommended)

```bash
# Recreate the image_chunks collection with 768 dimensions
python scripts/init_qdrant.py \
    --collection image_chunks \
    --vector-size 768 \
    --recreate
```

This will:

1. Delete the existing `image_chunks` collection (1024 dim)
2. Create a new `image_chunks` collection (768 dim)

### Option B: Using Python directly

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# Delete existing collection
try:
    client.delete_collection("image_chunks")
    print("Deleted existing image_chunks collection")
except Exception as e:
    print(f"Error deleting collection: {e}")

# Create new collection with 768 dimensions
client.create_collection(
    collection_name="image_chunks",
    vectors_config=VectorParams(
        size=768,  # CLIP large dimension
        distance=Distance.COSINE,
    ),
)
print("Created new image_chunks collection with 768 dimensions")
```

## Step 4: Re-embed Existing Images (If You Have Existing Data)

If you have images already ingested, you'll need to re-embed them with the new CLIP model:

### Option A: Re-ingest Documents

The easiest way is to re-upload your documents. The ingestion pipeline will:

1. Extract images again
2. Generate new CLIP embeddings (768 dim)
3. Store them in the new collection

```bash
# Re-upload documents through your API or ingestion script
# The pipeline will automatically use CLIP (768 dim) now
```

### Option B: Re-embed Only Images (Advanced)

If you want to re-embed images without re-processing entire documents:

```python
# Example script to re-embed images
from app.services.embedding.image_embedder import ImageEmbedder
from app.services.storage.supabase_storage import SupabaseImageStorage
from app.repositories.vector_repository import VectorRepository
from app.core.database import get_supabase_client

# Initialize services
image_embedder = ImageEmbedder(model_type="clip")  # Now uses CLIP
image_storage = SupabaseImageStorage()
vector_repo = VectorRepository(
    collection_name="image_chunks",
    vector_size=768,  # New dimension
)
supabase = get_supabase_client()

# Get all images from database
images = supabase.table("images").select("*").execute()

# Re-embed each image
for image in images.data:
    image_path = image["image_path"]

    # Download image from Supabase storage
    image_bytes = image_storage.download_image(image_path)  # You may need to implement this

    # Generate new embedding
    embedding = image_embedder.embed_image(image_bytes)

    # Update in Qdrant
    vector_repo.store_vectors(
        vectors=[embedding],
        chunk_ids=[image["chunk_id"]],
        payloads=[{
            "chunk_id": image["chunk_id"],
            "document_id": image["document_id"],
            "image_path": image_path,
            "caption": image.get("caption"),
            "image_type": image.get("image_type", "photo"),
            # ... other metadata
        }]
    )
```

## Step 5: Verify Migration

After recreating the collection and re-embedding:

```bash
# Check collection info
python scripts/init_qdrant.py --collection image_chunks
```

You should see:

- Vector size: **768** (not 1024)
- Points count: Should match your image count (after re-embedding)

## Step 6: Test Text-to-Image Search

Test that text-to-image search works correctly:

```python
from app.services.retrieval.image_retriever import ImageRetriever

retriever = ImageRetriever()  # Now uses CLIP (768 dim)

# Test query
results = await retriever.retrieve_with_embedding(
    query_embedding=retriever.generate_query_embedding("chart showing revenue growth"),
    limit=5
)

print(f"Found {len(results)} images")
for result in results:
    print(f"  - {result['image_path']} (score: {result['score']:.3f})")
```

## Troubleshooting

### Error: "Collection dimension mismatch"

If you see this error, the collection still has 1024 dimensions:

- Make sure you ran `--recreate` flag
- Verify collection was deleted and recreated
- Check collection info: `python scripts/init_qdrant.py --collection image_chunks`

### Error: "No images found"

If images aren't being retrieved:

- Make sure images were re-embedded after collection recreation
- Check that `image_chunks` collection has points: `python scripts/init_qdrant.py --collection image_chunks`
- Verify image embeddings are being generated with CLIP (check logs)

### Images not displaying

If images aren't showing in the frontend:

- Verify `image_path` is in the Qdrant payload
- Check that Supabase storage URLs are being generated correctly
- Ensure `AnswerGenerator` is handling image chunks properly

## Summary

✅ **Code changes completed:**

- `ImageEmbedder` now defaults to CLIP (768 dim)
- `ImageRetriever` uses 768 dimensions
- `IngestionPipeline` uses CLIP

✅ **Next steps:**

1. Recreate `image_chunks` collection: `python scripts/init_qdrant.py --collection image_chunks --vector-size 768 --recreate`
2. Re-embed existing images (re-upload documents or use re-embedding script)
3. Test text-to-image search

## Benefits After Migration

- ✅ Unified embedding space (text and images both 768 dim)
- ✅ Better semantic alignment between text queries and images
- ✅ No dimension mismatch issues
- ✅ Simpler architecture (one model for both text and image encoding)
