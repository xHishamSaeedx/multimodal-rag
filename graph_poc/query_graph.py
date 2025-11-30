"""
Query the knowledge graph and generate answers using Groq LLM.

This script allows you to query the graph and get LLM-generated answers.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # dotenv not installed, will show warning later
        pass

from graph_querier import GraphQuerier
from neo4j_connection import get_neo4j_driver, close_neo4j_driver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import Groq (will fail gracefully if not installed)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq library not installed. Install with: pip install groq")


class SimpleAnswerGenerator:
    """
    Simplified answer generator for POC using Groq.
    Based on backend/app/services/generation/answer_generator.py
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a precise information retrieval assistant. Your role is to extract and present information exclusively from the provided context.

Core Principles:
1. **Strict Context Adherence**: Use ONLY information explicitly present in the provided context. Do not infer, assume, or add information beyond what is stated.
2. **Deterministic Extraction**: When the context contains specific facts, numbers, or data, quote or reference them directly.
3. **Citation Requirement**: Always cite sources using [Document: filename, Chunk: N] format when referencing specific information.
4. **Uncertainty Handling**: If the context lacks information needed to answer, explicitly state "I don't have that information in the provided context."
"""
    
    DEFAULT_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Task: Analyze the context above and provide a precise answer to the question. 
Base your answer entirely on the extracted information. Cite sources using [Document: filename, Chunk: N] format.

Format your answer in Markdown."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize answer generator.
        
        Args:
            api_key: Groq API key (from env var GROQ_API_KEY if not provided)
            model: Groq model name (from env var GROQ_MODEL if not provided)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL")
        
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        if not self.model:
            raise ValueError("Groq model required. Set GROQ_MODEL environment variable or pass model parameter.")
        
        self.client = Groq(api_key=self.api_key)
        logger.info(f"Initialized Groq client with model: {self.model}")
    
    def generate_answer(
        self,
        query: str,
        chunks: list,
        max_context_length: int = 8000,
    ) -> dict:
        """
        Generate answer from chunks.
        
        Args:
            query: User query
            chunks: List of chunk dictionaries with chunk_text
            max_context_length: Maximum context length in characters
        
        Returns:
            Dictionary with answer, sources, etc.
        """
        if not chunks:
            return {
                "answer": "I don't have enough information to answer this question. No relevant context was found.",
                "sources": [],
                "model": self.model,
            }
        
        # Format context
        context_parts = []
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.get("chunk_text", "")
            filename = chunk.get("filename", "Unknown Document")
            chunk_index = chunk.get("chunk_index", idx + 1)
            
            citation = f"[Document: {filename}, Chunk: {chunk_index}]"
            chunk_with_citation = f"{citation}\n{chunk_text}\n\n"
            
            if sum(len(p) for p in context_parts) + len(chunk_with_citation) > max_context_length:
                break
            
            context_parts.append(chunk_with_citation)
        
        context_text = "\n".join(context_parts)
        
        # Build prompt
        user_prompt = self.DEFAULT_USER_PROMPT_TEMPLATE.format(
            context=context_text,
            question=query,
        )
        
        # Call Groq API
        logger.info(f"Generating answer with model: {self.model}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract sources
        sources = []
        for chunk in chunks[:len(context_parts)]:
            sources.append({
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_text": chunk.get("chunk_text", "")[:200] + "..." if len(chunk.get("chunk_text", "")) > 200 else chunk.get("chunk_text", ""),
            })
        
        # Extract token usage
        tokens_used = None
        if response.usage:
            tokens_used = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return {
            "answer": answer,
            "sources": sources,
            "model": self.model,
            "tokens_used": tokens_used,
        }


def query_and_answer(
    query: str,
    document_id: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    groq_model: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Query the graph and generate an answer.
    
    Args:
        query: User query/question
        document_id: Optional document ID to limit search
        groq_api_key: Groq API key (uses env var if not provided)
        groq_model: Groq model name (uses env var if not provided)
        limit: Maximum number of chunks to retrieve
    
    Returns:
        Dictionary with answer and sources
    """
    if not GROQ_AVAILABLE:
        raise ImportError("Groq library not installed. Install with: pip install groq")
    
    logger.info("=" * 60)
    logger.info("Graph Query & Answer Generation")
    logger.info("=" * 60)
    logger.info(f"Query: {query}")
    
    # Query graph
    logger.info("\n[Step 1/3] Querying knowledge graph...")
    driver = get_neo4j_driver()
    
    try:
        with GraphQuerier(driver=driver) as querier:
            if document_id:
                # Query specific document
                chunks = querier.query_by_document(document_id, limit=limit)
                logger.info(f"Found {len(chunks)} chunks from document {document_id}")
            else:
                # Query by keywords extracted from query
                keywords = [word for word in query.split() if len(word) > 3]
                chunks = querier.query_by_keywords(keywords, limit=limit)
                logger.info(f"Found {len(chunks)} chunks matching keywords: {keywords}")
            
            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information in the knowledge graph to answer this question.",
                    "sources": [],
                    "error": "No chunks found",
                }
            
            # Enrich chunks with content
            logger.info(f"[Step 2/3] Enriching {len(chunks)} chunks...")
            enriched_chunks = querier.enrich_chunks_with_content(chunks)
            
            # Filter out chunks without content
            chunks_with_content = [c for c in enriched_chunks if c.get("chunk_text") and not c.get("chunk_text", "").startswith("[Chunk")]
            
            if not chunks_with_content:
                logger.warning("No chunks with content found. Document may need to be re-ingested.")
                return {
                    "answer": "I found chunks in the graph, but they don't have content stored. Please re-ingest the document to store chunk content.",
                    "sources": [],
                    "error": "No content in chunks",
                }
            
            # Use chunks with content
            enriched_chunks = chunks_with_content
            logger.info(f"Using {len(enriched_chunks)} chunks with content")
        
        # Generate answer
        logger.info(f"[Step 3/3] Generating answer with Groq...")
        answer_generator = SimpleAnswerGenerator(
            api_key=groq_api_key,
            model=groq_model
        )
        
        result = answer_generator.generate_answer(query, enriched_chunks)
        
        logger.info("\n" + "=" * 60)
        logger.info("[SUCCESS] Answer generated!")
        logger.info("=" * 60)
        
        return result
        
    finally:
        close_neo4j_driver()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python query_graph.py <query> [document_id]")
        print("\nExamples:")
        print("  python query_graph.py 'What is machine learning?'")
        print("  python query_graph.py 'What are the key metrics?' doc_83374995")
        print("\nEnvironment Variables:")
        print("  GROQ_API_KEY: Your Groq API key (required)")
        print("  GROQ_MODEL: Groq model name (e.g., 'llama-3.1-70b-versatile')")
        sys.exit(1)
    
    query = sys.argv[1]
    document_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Get API key and model from .env file or environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")
    
    # Check if .env file exists
    env_path = Path(__file__).parent / ".env"
    
    if not groq_api_key:
        if env_path.exists():
            print(f"\n[ERROR] GROQ_API_KEY not found in .env file: {env_path}")
            print("Please add GROQ_API_KEY=your-api-key to the .env file")
        else:
            print(f"\n[ERROR] GROQ_API_KEY not set and .env file not found: {env_path}")
            print("Create a .env file in graph_poc folder with:")
            print("  GROQ_API_KEY=your-api-key")
            print("  GROQ_MODEL=llama-3.1-70b-versatile")
        sys.exit(1)
    
    if not groq_model:
        if env_path.exists():
            print(f"\n[ERROR] GROQ_MODEL not found in .env file: {env_path}")
            print("Please add GROQ_MODEL=llama-3.1-70b-versatile to the .env file")
        else:
            print(f"\n[ERROR] GROQ_MODEL not set and .env file not found: {env_path}")
            print("Create a .env file in graph_poc folder with:")
            print("  GROQ_API_KEY=your-api-key")
            print("  GROQ_MODEL=llama-3.1-70b-versatile")
        print("\nAvailable models:")
        print("  - llama-3.1-70b-versatile")
        print("  - llama-3.1-8b-instant")
        print("  - mixtral-8x7b-32768")
        sys.exit(1)
    
    try:
        result = query_and_answer(query, document_id, groq_api_key, groq_model)
        
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["answer"])
        
        if result.get("sources"):
            print("\n" + "=" * 60)
            print("SOURCES")
            print("=" * 60)
            for i, source in enumerate(result["sources"], 1):
                print(f"\n[{i}] {source['filename']} - Chunk {source['chunk_index']}")
                print(f"    {source['chunk_text'][:150]}...")
        
        if result.get("tokens_used"):
            print("\n" + "=" * 60)
            print("TOKEN USAGE")
            print("=" * 60)
            tokens = result["tokens_used"]
            print(f"Prompt: {tokens['prompt_tokens']} tokens")
            print(f"Completion: {tokens['completion_tokens']} tokens")
            print(f"Total: {tokens['total_tokens']} tokens")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Failed to query and generate answer: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

