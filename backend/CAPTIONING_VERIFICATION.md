# Captioning Verification Guide

## Did Captioning Work?

Based on the query logs you provided, **we cannot definitively tell if captioning worked** because:

1. ✅ **Image chunk was retrieved** - 1 image chunk in top 10 results
2. ✅ **Image URL was generated** - Supabase signed URLs created successfully
3. ❓ **No caption logs visible** - Query logs don't show caption values

## How Captioning Works

### During Ingestion (When Document is Uploaded)

1. **Initialization** (`pipeline.py` line 99):
   ```python
   self.vision_processor = VisionProcessorFactory.create_processor(mode="captioning")
   ```
   - Should log: `"Initialized captioning processor for image captioning during ingestion"`
   - If failed: `"Failed to initialize captioning processor: {error}"`

2. **Caption Generation** (`pipeline.py` lines 395-412):
   - For each extracted image:
     - Logs: `"Generating caption for image {index}"`
     - Processes image with BLIP model
     - Logs: `"✓ Generated caption for image {index}: {caption[:50]}..."`
     - If failed: `"Failed to generate caption for image {index}: {error}"`

3. **Storage**:
   - Caption stored in `images` table (`caption` column)
   - Caption also stored in chunk payload in Qdrant
   - Caption included in `chunk_text` field

### During Query (When Answer is Generated)

1. **Retrieval** (`image_retriever.py` line 144):
   - Caption extracted from Qdrant payload: `payload.get("caption", "")`

2. **Usage** (`answer_generator.py` line 351):
   ```python
   image_caption = chunk.get("caption") or metadata.get("caption") or chunk_text
   ```
   - Falls back to `chunk_text` if caption is missing

3. **Fallback Behavior**:
   - If caption is empty/null: Uses `chunk_text` (which may contain OCR text)
   - If caption is generic ("Image", "photo"): Tries to enhance from filename

## How to Verify Captioning Worked

### Method 1: Check Ingestion Logs

Look for these log messages when the document was ingested:

**Success indicators:**
```
INFO: Initialized captioning processor for image captioning during ingestion
DEBUG: Generating caption for image 1
DEBUG: ✓ Generated caption for image 1: a chart showing revenue growth...
```

**Failure indicators:**
```
WARNING: Failed to initialize captioning processor: {error}
WARNING: Failed to generate caption for image 1: {error}
```

### Method 2: Check Database

Query the `images` table in Supabase:

```sql
SELECT 
    id,
    image_path,
    caption,
    extracted_text,
    image_type
FROM images
WHERE document_id = 'your-document-id'
ORDER BY created_at;
```

**What to look for:**
- ✅ `caption` column has descriptive text (not NULL, not empty)
- ✅ Caption is meaningful (not just "Image" or "photo")
- ❌ `caption` is NULL or empty → Captioning failed

### Method 3: Check Qdrant Payload

Query Qdrant to see what's stored in the image chunk:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
results = client.scroll(
    collection_name="image_chunks",
    scroll_filter={
        "must": [{"key": "chunk_id", "match": {"value": "your-chunk-id"}}]
    }
)

for point in results[0]:
    payload = point.payload
    print(f"Caption: {payload.get('caption')}")
    print(f"Text: {payload.get('text')}")
```

### Method 4: Check Query Response

In the query response, check the sources:

```json
{
  "sources": [
    {
      "chunk_type": "image",
      "chunk_text": "...",  // Should contain caption
      "metadata": {
        "caption": "..."  // Should have caption here
      }
    }
  ]
}
```

## Common Issues

### Issue 1: Transformers Not Installed

**Symptom:**
```
WARNING: transformers library not available. Install with: pip install transformers torch
```

**Solution:**
```bash
pip install transformers torch torchvision
```

### Issue 2: Model Download Failed

**Symptom:**
```
Failed to initialize captioning processor: ConnectionError or TimeoutError
```

**Solution:**
- Check internet connection
- Model downloads on first use (Salesforce/blip-image-captioning-base)
- May take several minutes

### Issue 3: CUDA/GPU Issues

**Symptom:**
```
Failed to load captioning model: CUDA out of memory
```

**Solution:**
- Use CPU mode (default)
- Or reduce batch size
- Or use smaller model: `Salesforce/blip-image-captioning-base`

### Issue 4: Vision Processor Not Initialized

**Symptom:**
- No caption generated
- `caption` is NULL in database

**Check:**
- Look for initialization log: `"Initialized captioning processor..."`
- Check if `self.vision_processor` is None in pipeline

## What the Logs Tell Us

From your query logs:

```
81| {"results_count": 5, "event": "image_chunks_search_completed"}
91| {"results_count": 10, "text_chunks": 4, "table_chunks": 5, "image_chunks": 1}
95| HTTP Request: POST .../storage/v1/object/sign/... "HTTP/2 200 OK"
```

**Observations:**
1. ✅ Image chunk was found and retrieved
2. ✅ Image URL was generated successfully
3. ❓ No logs showing what caption was used
4. ❓ No errors about missing captions

**Conclusion:** The system is working, but we need to check:
- What caption value is actually stored
- Whether captioning ran during ingestion

## Next Steps to Verify

1. **Check ingestion logs** for the document that contains the image
2. **Query the database** to see the actual caption value
3. **Check the query response** sources to see what caption was used
4. **Test with a new document** and watch the ingestion logs

## Expected Caption Quality

Good captions should be:
- Descriptive: "A bar chart showing revenue growth from 2019 to 2021"
- Specific: "Line graph displaying tech sector revenue, with 2021 showing $2.5B"
- Not generic: Avoid "Image", "photo", "chart" alone

If you see generic captions, captioning may have:
- Failed silently
- Used OCR text instead
- Generated a poor caption (model limitation)

