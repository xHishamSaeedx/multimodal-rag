# Graph Query & Answer Generation - Quick Setup

## Quick Start

### 1. Create .env File

Create a `.env` file in the `graph_poc` folder:

```env
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.1-70b-versatile
```

### 2. Install Dependencies

```powershell
cd graph_poc
pip install -r requirements.txt
```

### 3. Query the Graph

```powershell
# Query all documents
python graph_poc\query_graph.py "What is machine learning?"

# Query specific document
python graph_poc\query_graph.py "What are the key metrics?" doc_f1074a9f
```

## What You Get

- **Answer**: LLM-generated answer based on graph content
- **Sources**: List of chunks used with citations
- **Token Usage**: API usage statistics

## Example Output

```
============================================================
ANSWER
============================================================
Machine learning is a subset of artificial intelligence that...

[Document: tech_sector_report.pdf, Chunk: 3]

============================================================
SOURCES
============================================================

[1] tech_sector_report.pdf - Chunk 3
    Machine learning algorithms enable systems to learn...
```

## Available Models

- `llama-3.1-70b-versatile` - Best quality (recommended)
- `llama-3.1-8b-instant` - Fast, smaller
- `mixtral-8x7b-32768` - Alternative

## Troubleshooting

**"GROQ_API_KEY not found in .env file"**
→ Create `.env` file in `graph_poc` folder with:
  ```
  GROQ_API_KEY=your-api-key
  GROQ_MODEL=llama-3.1-70b-versatile
  ```

**"No chunks found"**
→ Make sure documents are ingested first
→ Try different query keywords

**"python-dotenv not installed"**
→ Run: `pip install python-dotenv`

