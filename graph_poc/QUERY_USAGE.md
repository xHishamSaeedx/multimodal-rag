# Graph Query & Answer Generation Usage

This guide explains how to query the knowledge graph and generate answers using Groq LLM.

## Prerequisites

1. **Neo4j running** with ingested documents
2. **Groq API key** - Get one from https://console.groq.com/
3. **Groq model name** - e.g., `llama-3.1-70b-versatile`

## Setup

### 1. Install Dependencies

```bash
cd graph_poc
pip install -r requirements.txt
```

### 2. Set Environment Variables

**PowerShell:**
```powershell
$env:GROQ_API_KEY='your-groq-api-key-here'
$env:GROQ_MODEL='llama-3.1-70b-versatile'
```

**Bash/Linux:**
```bash
export GROQ_API_KEY='your-groq-api-key-here'
export GROQ_MODEL='llama-3.1-70b-versatile'
```

### 3. Available Groq Models

- `llama-3.1-70b-versatile` - Best quality (recommended)
- `llama-3.1-8b-instant` - Fast, smaller model
- `mixtral-8x7b-32768` - Alternative high-quality model

## Usage

### Basic Query

Query the graph and get an answer:

```bash
python query_graph.py "What is machine learning?"
```

### Query Specific Document

Query a specific document by ID:

```bash
python query_graph.py "What are the key metrics?" doc_f1074a9f
```

### Programmatic Usage

```python
from query_graph import query_and_answer

# Query with environment variables
result = query_and_answer("What is the main topic?")

# Or provide API key and model directly
result = query_and_answer(
    query="What are the key findings?",
    document_id="doc_f1074a9f",
    groq_api_key="your-api-key",
    groq_model="llama-3.1-70b-versatile"
)

print(result["answer"])
for source in result["sources"]:
    print(f"  - {source['filename']} (Chunk {source['chunk_index']})")
```

## How It Works

1. **Graph Query**: Searches Neo4j for chunks matching the query
   - Extracts keywords from query
   - Finds entities matching keywords
   - Retrieves chunks that mention those entities

2. **Chunk Retrieval**: Gets chunk content from graph
   - Retrieves chunk text stored in Neo4j
   - Formats chunks with citations

3. **Answer Generation**: Uses Groq LLM to generate answer
   - Builds context from retrieved chunks
   - Generates answer with citations
   - Returns formatted response

## Output Format

The query returns a dictionary with:

```python
{
    "answer": "Generated answer text with citations...",
    "sources": [
        {
            "chunk_id": "chunk_123",
            "document_id": "doc_456",
            "filename": "document.pdf",
            "chunk_index": 5,
            "chunk_text": "Preview of chunk text..."
        }
    ],
    "model": "llama-3.1-70b-versatile",
    "tokens_used": {
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "total_tokens": 700
    }
}
```

## Troubleshooting

### "Groq API key not configured"
- Set `GROQ_API_KEY` environment variable
- Or pass `groq_api_key` parameter

### "Groq model not configured"
- Set `GROQ_MODEL` environment variable
- Or pass `groq_model` parameter

### "No chunks found"
- Make sure documents are ingested
- Try a different query with different keywords
- Check that entities were extracted during ingestion

### "Content not stored in graph"
- Re-ingest documents (content is now stored during ingestion)
- Check that chunk content is being stored in `create_document_graph_batch`

## Next Steps

- Add semantic search (vector similarity)
- Improve entity extraction (use proper NER)
- Add multi-hop reasoning
- Integrate with backend answer generator

