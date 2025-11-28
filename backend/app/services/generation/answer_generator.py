"""
Answer generation service.

Generates answers using LLM (Groq)
with retrieved context and citation extraction.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from groq import Groq
from app.core.config import settings
from app.services.storage.supabase_storage import SupabaseImageStorage
from app.services.vision import VisionProcessorFactory
from app.utils.exceptions import BaseAppException
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AnswerGeneratorError(BaseAppException):
    """Raised when answer generation operations fail."""
    pass


class AnswerGenerator:
    """
    Answer generation service using Groq LLM.
    
    Features:
    - Generates answers from retrieved chunks
    - Extracts and formats citations
    - Basic hallucination guardrails
    - Configurable prompt templates
    - Error handling and retry logic
    """
    
    # Default prompt template (following Phase 1 specification)
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context. 

Instructions:
- Answer based ONLY on the provided context
- Format your answer in Markdown format
- Use proper Markdown syntax for formatting (headers, lists, code blocks, tables, etc.)
- When presenting tabular data, use Markdown table format with proper alignment
- When images are referenced in the context, the image description/caption contains important information - USE IT to answer questions
- If an image is described as a chart, graph, or diagram, the description may contain data points, values, or trends mentioned in the context
- Extract and use numerical data, trends, and information from image descriptions when answering questions
- Cite sources using [Document: filename, Chunk: N] format
- If information is not in the context, say "I don't have that information"
- Be concise and accurate
- Do not make up information that is not in the context"""

    DEFAULT_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Please provide an answer based on the context above. Format your answer in Markdown. 

IMPORTANT:
- If the context contains tables, extract and present the data using Markdown table syntax
- If the context contains images (especially charts, graphs, or diagrams), carefully read the image description/caption - it may contain the data you need to answer the question
- Image descriptions often contain numerical data, trends, or visual information that answers questions about charts and graphs
- Extract specific numbers, values, and trends from image descriptions when they are provided
- Reference images naturally in your answer when they contain relevant information

If the context doesn't contain the answer, say "I don't have that information"."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,  # Lower temperature for more deterministic answers
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the answer generator.
        
        Args:
            api_key: Groq API key (defaults to settings.groq_api_key)
            model: Groq model name (defaults to settings.groq_model)
            temperature: Sampling temperature (0.0 for deterministic, higher for creative)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt (uses default if not provided)
        """
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.image_storage = SupabaseImageStorage()  # For generating image URLs
        
        # Initialize vision processor if in vision_llm mode (for query-time processing)
        try:
            vision_mode = getattr(settings, "vision_processing_mode", "captioning")
            if vision_mode == "vision_llm":
                self.vision_processor = VisionProcessorFactory.create_processor(mode="vision_llm")
                logger.info("Initialized Vision LLM processor for query-time image understanding")
            else:
                self.vision_processor = None
                logger.debug("Using captioning mode - captions generated during ingestion")
        except Exception as e:
            logger.warning(f"Failed to initialize vision processor: {e}. Will use stored captions.")
            self.vision_processor = None
        
        if not self.api_key:
            logger.warning(
                "answer_generator_config_missing",
                config_key="GROQ_API_KEY",
                message="Answer generation will fail. Please set GROQ_API_KEY in your .env file.",
            )
        if not self.model:
            logger.warning(
                "answer_generator_config_missing",
                config_key="GROQ_MODEL",
                message="Answer generation will fail. Please set GROQ_MODEL in your .env file.",
            )
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            logger.debug("answer_generator_initialized", model=self.model)
        else:
            self.client = None
    
    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        include_sources: bool = True,
        max_context_length: int = 8000,  # Approximate token limit for context
    ) -> Dict[str, Any]:
        """
        Generate an answer from retrieved chunks.
        
        Args:
            query: User query/question
            chunks: List of retrieved chunks, each containing:
                - chunk_id: UUID string
                - document_id: UUID string
                - chunk_text: Text content
                - filename: Document filename
                - metadata: Additional metadata (may contain chunk_index, page_number, etc.)
            include_sources: Whether to include source citations in response
            max_context_length: Maximum characters to include in context (approximate token limit)
        
        Returns:
            Dictionary containing:
                - answer: Generated answer text
                - sources: List of source citations (if include_sources=True)
                - chunks_used: List of chunk IDs used in answer generation
                - model: Model used for generation
                - tokens_used: Optional token usage information
        
        Raises:
            AnswerGeneratorError: If generation fails
        """
        if not self.api_key:
            raise AnswerGeneratorError(
                "Groq API key not configured. Please set GROQ_API_KEY in your .env file.",
                {"query": query[:100] if query else ""},
            )
        
        if not self.model:
            raise AnswerGeneratorError(
                "Groq model not configured. Please set GROQ_MODEL in your .env file.",
                {"query": query[:100] if query else ""},
            )
        
        if not query or not query.strip():
            raise AnswerGeneratorError(
                "Query cannot be empty",
                {"query": ""},
            )
        
        if not chunks:
            logger.warning("answer_generation_no_chunks")
            return {
                "answer": "I don't have enough information to answer this question. No relevant context was found.",
                "sources": [],
                "chunks_used": [],
                "model": self.model,
            }
        
        try:
            # Check for potential deduplication issues
            # Count chunk types
            text_chunks = [c for c in chunks if c.get("chunk_type", "text") == "text"]
            table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
            image_chunks = [c for c in chunks if c.get("chunk_type") == "image"]
            
            logger.debug(
                "answer_generation_chunk_analysis",
                total_chunks=len(chunks),
                text_chunks=len(text_chunks),
                table_chunks=len(table_chunks),
                image_chunks=len(image_chunks),
            )
            
            # Format context from chunks
            context_text, chunk_mapping = self._format_context(
                chunks=chunks,
                max_length=max_context_length,
                question=query,
            )
            
            # Log if we have both text and table chunks (potential deduplication check)
            if text_chunks and table_chunks:
                logger.info(
                    "answer_generation_mixed_chunks",
                    text_chunks_count=len(text_chunks),
                    table_chunks_count=len(table_chunks),
                    message="Both text and table chunks found - deduplication may need verification",
                )
            
            # Build prompt
            user_prompt = self.DEFAULT_USER_PROMPT_TEMPLATE.format(
                context=context_text,
                question=query,
            )
            
            logger.debug(
                "answer_generation_start",
                query_preview=query[:50] if len(query) > 50 else query,
                chunks_count=len(chunks),
                context_length=len(context_text),
            )
            
            # Call Groq API and measure TTFT
            request_start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            request_end_time = time.time()
            ttft = request_end_time - request_start_time  # Time to first token (response received)
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Extract sources if requested
            sources = []
            chunks_used = list(chunk_mapping.keys())
            
            if include_sources:
                sources = self._extract_sources(
                    answer=answer,
                    chunks=chunks,
                    chunk_mapping=chunk_mapping,
                )
            
            # Extract token usage if available
            tokens_used = None
            if response.usage:
                tokens_used = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            logger.info(
                "answer_generation_completed",
                answer_length=len(answer),
                sources_count=len(sources),
                total_tokens=tokens_used['total_tokens'] if tokens_used else None,
                ttft_seconds=round(ttft, 3),
                model=self.model,
            )
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": chunks_used,
                "model": self.model,
                "tokens_used": tokens_used,
                "ttft": ttft,  # Time to first token in seconds
            }
        
        except Exception as e:
            logger.error(
                "answer_generation_error",
                error_type=type(e).__name__,
                error_message=str(e),
                query_preview=query[:100] if query else "",
                exc_info=True,
            )
            raise AnswerGeneratorError(
                f"Failed to generate answer from Groq: {str(e)}",
                {"query": query[:100] if query else "", "error": str(e)},
            ) from e
    
    def _format_context(
        self,
        chunks: List[Dict[str, Any]],
        max_length: int = 8000,
        question: Optional[str] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Format chunks into context text with citations.
        
        Args:
            chunks: List of retrieved chunks
            max_length: Maximum context length (characters)
            question: Optional query/question text (used for Vision LLM processing)
        
        Returns:
            Tuple of (context_text, chunk_mapping) where:
                - context_text: Formatted context string
                - chunk_mapping: Dictionary mapping chunk_id to citation index
        """
        context_parts = []
        chunk_mapping = {}
        current_length = 0
        
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("chunk_text", "")
            filename = chunk.get("filename", "Unknown Document")
            metadata = chunk.get("metadata", {})
            chunk_type = chunk.get("chunk_type", "text")
            
            # Get chunk index from metadata or use sequence number
            chunk_index = metadata.get("chunk_index", idx + 1)
            
            # If this is a table chunk, prefer table_markdown if available
            if chunk_type == "table":
                # Check if table_markdown is available in payload or metadata
                table_markdown = chunk.get("table_markdown") or metadata.get("table_markdown")
                if table_markdown:
                    chunk_text = table_markdown
                # Otherwise, use chunk_text (which should be flattened text)
            
            # If this is an image chunk, add image reference with URL and enhanced description
            if chunk_type == "image":
                image_path = chunk.get("image_path") or metadata.get("image_path")
                image_type = metadata.get("image_type", "photo")
                
                if image_path:
                    try:
                        # Generate signed URL for the image (valid for 1 hour)
                        image_url = self.image_storage.get_image_url(image_path, expires_in=3600)
                        
                        # Get caption/description (from stored caption or chunk text)
                        image_caption = chunk.get("caption") or metadata.get("caption") or chunk_text
                        
                        # If using Vision LLM mode, process image with query for real-time understanding
                        if self.vision_processor and self.vision_processor.get_mode() == "vision_llm" and question:
                            try:
                                # Download image from Supabase
                                image_bytes = self.image_storage.download_image(image_path)
                                
                                # Process with Vision LLM using the query
                                vision_result = self.vision_processor.process_image(
                                    image_bytes=image_bytes,
                                    query=question,  # Use the actual query for better understanding
                                    context=chunk_text,  # Include surrounding context
                                )
                                
                                # Use Vision LLM's analysis as the description
                                image_caption = vision_result.description
                                logger.debug(f"Vision LLM analysis: {image_caption[:100]}...")
                            except Exception as e:
                                logger.warning(f"Failed to process image with Vision LLM: {e}. Using stored caption.")
                                # Fall back to stored caption
                        
                        # Enhance description based on image type and filename
                        enhanced_description = image_caption
                        if not enhanced_description or enhanced_description in ["Image: photo", "Image", "photo"]:
                            # Try to infer from filename (e.g., "revenue_growth.png" -> "Revenue growth chart")
                            filename_hint = ""
                            if image_path:
                                # Extract filename from path (e.g., "doc_id/image_1_20251128_182843_d0a0b4b8.png")
                                path_parts = image_path.split("/")
                                if len(path_parts) > 1:
                                    filename = path_parts[-1]
                                    # Remove common prefixes and extensions
                                    clean_name = filename.replace("image_", "").replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
                                    # If it's a chart/graph type, mention it
                                    if image_type in ["chart", "diagram", "graph"]:
                                        filename_hint = f" {image_type.capitalize()}"
                            
                            enhanced_description = f"Visual {image_type}" + filename_hint
                        
                        # Build comprehensive image context
                        image_context = f"[Image: {enhanced_description}]"
                        if image_caption and image_caption != enhanced_description and image_caption not in ["Image: photo", "Image", "photo"]:
                            image_context += f"\nDescription: {image_caption}"
                        
                        # Add surrounding text context if available (might contain chart data)
                        if chunk_text and chunk_text not in ["Image: photo", "Image", enhanced_description, "photo"]:
                            image_context += f"\nContext: {chunk_text}"
                        
                        image_context += f"\nImage URL: {image_url}"
                        
                        # Add explicit instruction for charts/graphs
                        if image_type in ["chart", "diagram", "graph"] or "chart" in enhanced_description.lower() or "graph" in enhanced_description.lower():
                            image_context += f"\n\nIMPORTANT: This is a {image_type} that likely contains numerical data, trends, or visual information. "
                            image_context += f"Look at the surrounding text chunks in the context above and below this image - they may describe what the chart shows, "
                            image_context += f"including specific numbers, years, values, or trends. Use that information to answer questions about the chart."
                        else:
                            image_context += f"\n\nNOTE: This image may contain charts, graphs, or data visualizations. "
                            image_context += f"If this is a chart or graph, look at surrounding text chunks in the context that may describe the data shown in the image."
                        
                        chunk_text = image_context
                    except Exception as e:
                        logger.warning(f"Failed to generate image URL for {image_path}: {e}")
                        # Still include the chunk text even if URL generation fails
                        enhanced_description = chunk.get("caption") or metadata.get("caption") or chunk_text or "Image"
                        chunk_text = f"[Image: {enhanced_description}]\n{chunk_text if chunk_text and chunk_text not in ['Image: photo', 'Image', 'photo'] else 'Visual content from document'}"
            
            # Format chunk with citation
            citation = f"[Document: {filename}, Chunk: {chunk_index}]"
            chunk_with_citation = f"{citation}\n{chunk_text}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_with_citation) > max_length and context_parts:
                logger.debug(
                    "context_length_limit_reached",
                    max_length=max_length,
                    chunks_included=idx + 1,
                    total_chunks=len(chunks),
                )
                break
            
            context_parts.append(chunk_with_citation)
            chunk_mapping[chunk_id] = idx + 1  # Citation number (1-indexed)
            current_length += len(chunk_with_citation)
        
        context_text = "\n".join(context_parts)
        return context_text, chunk_mapping
    
    def _extract_sources(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        chunk_mapping: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Extract source citations from answer and chunks.
        
        Identifies which chunks were used by looking for citation patterns in the answer
        or by including all chunks in the context (fallback).
        
        Args:
            answer: Generated answer text
            chunks: List of chunks used
            chunk_mapping: Mapping of chunk_id to citation index
        
        Returns:
            List of source dictionaries, each containing:
                - chunk_id: UUID string
                - document_id: UUID string
                - filename: Document filename
                - chunk_index: Chunk index in document
                - chunk_text: Chunk text (truncated)
                - citation: Citation string
        """
        sources = []
        used_chunk_ids = set()
        
        # Try to extract citations from answer
        # Look for patterns like [Document: filename, Chunk: N]
        import re
        citation_pattern = r'\[Document:\s*([^,]+),\s*Chunk:\s*(\d+)\]'
        matches = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        # Create a mapping of (filename, chunk_index) to chunk
        chunk_by_citation = {}
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            filename = chunk.get("filename", "Unknown Document")
            metadata = chunk.get("metadata", {})
            chunk_index = metadata.get("chunk_index", chunk_mapping.get(chunk_id, 0))
            chunk_by_citation[(filename.lower(), chunk_index)] = chunk
        
        # Extract sources from matches
        for filename_match, chunk_index_match in matches:
            filename = filename_match.strip()
            try:
                chunk_index = int(chunk_index_match)
                chunk = chunk_by_citation.get((filename.lower(), chunk_index))
                if chunk and chunk.get("chunk_id") not in used_chunk_ids:
                    sources.append(self._format_source(chunk, chunk_index))
                    used_chunk_ids.add(chunk.get("chunk_id"))
            except ValueError:
                continue
        
        # If no citations found in answer, include all chunks as sources (fallback)
        if not sources:
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in used_chunk_ids:
                    metadata = chunk.get("metadata", {})
                    chunk_index = metadata.get("chunk_index", chunk_mapping.get(chunk_id, 0))
                    sources.append(self._format_source(chunk, chunk_index))
                    used_chunk_ids.add(chunk_id)
        
        return sources
    
    def _format_source(
        self,
        chunk: Dict[str, Any],
        chunk_index: int,
    ) -> Dict[str, Any]:
        """
        Format a chunk into a source dictionary.
        
        Args:
            chunk: Chunk dictionary
            chunk_index: Chunk index for citation
        
        Returns:
            Source dictionary
        """
        chunk_text = chunk.get("chunk_text", "")
        filename = chunk.get("filename", "Unknown Document")
        citation = f"[Document: {filename}, Chunk: {chunk_index}]"
        chunk_type = chunk.get("chunk_type", "text")
        metadata = chunk.get("metadata", {})
        
        # Truncate chunk text for source display (first 200 chars)
        chunk_text_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        
        source_dict = {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "document_id": str(chunk.get("document_id", "")),
            "filename": filename,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text_preview,
            "full_chunk_text": chunk_text,  # Include full text for reference
            "citation": citation,
            "metadata": chunk.get("metadata", {}),
        }
        
        # Add image-specific fields if this is an image chunk
        if chunk_type == "image":
            image_path = chunk.get("image_path") or metadata.get("image_path")
            if image_path:
                source_dict["image_path"] = image_path
                try:
                    # Generate signed URL for the image (valid for 1 hour)
                    image_url = self.image_storage.get_image_url(image_path, expires_in=3600)
                    source_dict["image_url"] = image_url
                except Exception as e:
                    logger.warning(f"Failed to generate image URL for source {image_path}: {e}")
                    source_dict["image_url"] = None
        
        return source_dict
        
        # Add image-specific fields if this is an image chunk
        if chunk_type == "image":
            image_path = chunk.get("image_path") or metadata.get("image_path")
            if image_path:
                source_dict["image_path"] = image_path
                try:
                    # Generate signed URL for the image (valid for 1 hour)
                    image_url = self.image_storage.get_image_url(image_path, expires_in=3600)
                    source_dict["image_url"] = image_url
                except Exception as e:
                    logger.warning(f"Failed to generate image URL for source {image_path}: {e}")
                    source_dict["image_url"] = None
        
        return source_dict
    
    def validate_answer_groundedness(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Basic hallucination detection by checking if answer is grounded in chunks.
        
        This is a simple Phase 1 implementation using keyword matching.
        More advanced validation can be added in later phases.
        
        Args:
            answer: Generated answer text
            chunks: Retrieved chunks used for generation
            threshold: Minimum overlap threshold (0.0-1.0)
        
        Returns:
            Dictionary containing:
                - is_grounded: Boolean indicating if answer appears grounded
                - confidence: Confidence score (0.0-1.0)
                - matched_chunks: List of chunk IDs that match
        """
        # Simple keyword-based validation (Phase 1)
        # Extract keywords from answer (simple word splitting)
        answer_words = set(answer.lower().split())
        
        # Extract keywords from chunks
        chunk_keywords = []
        chunk_ids = []
        for chunk in chunks:
            chunk_text = chunk.get("chunk_text", "").lower()
            chunk_words = set(chunk_text.split())
            chunk_keywords.append(chunk_words)
            chunk_ids.append(chunk.get("chunk_id"))
        
        # Count matching keywords
        matched_chunks = []
        match_counts = []
        
        for idx, chunk_words in enumerate(chunk_keywords):
            overlap = len(answer_words.intersection(chunk_words))
            if overlap > 0:
                matched_chunks.append(chunk_ids[idx])
                match_counts.append(overlap)
        
        # Calculate confidence based on keyword overlap
        if matched_chunks:
            total_chunk_words = sum(len(words) for words in chunk_keywords)
            total_matches = sum(match_counts)
            confidence = min(1.0, total_matches / max(total_chunk_words / len(chunks), 1))
        else:
            confidence = 0.0
        
        is_grounded = confidence >= threshold
        
        return {
            "is_grounded": is_grounded,
            "confidence": confidence,
            "matched_chunks": matched_chunks,
        }
