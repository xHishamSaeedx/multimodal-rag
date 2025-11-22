"""
Text embedding service.

Generates embeddings using sentence-transformers models
(e.g., e5-base-v2, all-mpnet-base-v2, all-MiniLM-L6-v2).
"""

import logging
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from app.core.config import settings
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class EmbeddingError(BaseAppException):
    """Raised when embedding generation fails."""
    pass


class TextEmbedder:
    """
    Service for generating text embeddings using sentence-transformers.
    
    Supports various models:
    - intfloat/e5-base-v2 (768 dim, recommended)
    - sentence-transformers/all-mpnet-base-v2 (768 dim)
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, faster)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Model name (default: from config)
            device: Device to use ("cpu" or "cuda", default: from config)
        """
        if SentenceTransformer is None:
            raise EmbeddingError(
                "sentence-transformers is not installed. Install it with: pip install sentence-transformers",
                {},
            )
        
        self.model_name = model_name or settings.embedding_model
        requested_device = device or settings.embedding_device
        self.batch_size = settings.embedding_batch_size
        
        # Auto-detect CUDA availability and validate device
        self.device = self._validate_device(requested_device)
        
        # Initialize model
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded embedding model: {self.model_name} "
                f"(dimension: {self.embedding_dim}, device: {self.device})"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise EmbeddingError(
                f"Failed to load embedding model {self.model_name}: {str(e)}",
                {"model_name": self.model_name, "device": self.device, "error": str(e)},
            ) from e
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                # Return zero vector for empty text
                return [0.0] * self.embedding_dim
            
            # For e5-base-v2, add "query: " or "passage: " prefix
            # Since we're embedding chunks (passages), add "passage: " prefix
            if "e5" in self.model_name.lower():
                text = f"passage: {text}"
            
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                show_progress_bar=False,
            )
            
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise EmbeddingError(
                f"Failed to generate embedding: {str(e)}",
                {"text_length": len(text) if text else 0, "error": str(e)},
            ) from e
    
    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors
        
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not texts:
                logger.warning("Empty texts list provided for embedding")
                return []
            
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text and text.strip()]
            
            if not non_empty_texts:
                logger.warning("All texts are empty")
                return [[0.0] * self.embedding_dim] * len(texts)
            
            # For e5-base-v2, add "passage: " prefix to all texts
            if "e5" in self.model_name.lower():
                non_empty_texts = [f"passage: {text}" for text in non_empty_texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            # Insert zero vectors for empty texts at original positions
            result = []
            non_empty_idx = 0
            for text in texts:
                if text and text.strip():
                    result.append(embeddings_list[non_empty_idx])
                    non_empty_idx += 1
                else:
                    result.append([0.0] * self.embedding_dim)
            
            logger.debug(f"Generated {len(embeddings_list)} embeddings")
            return result
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}", exc_info=True)
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {str(e)}",
                {"text_count": len(texts), "error": str(e)},
            ) from e
    
    def _validate_device(self, device: str) -> str:
        """
        Validate and auto-correct device selection.
        
        If CUDA is requested but not available, automatically falls back to CPU.
        
        Args:
            device: Requested device ("cpu" or "cuda")
        
        Returns:
            Validated device string
        """
        device_lower = device.lower().strip()
        
        # Normalize device name
        if device_lower in ("cuda", "gpu"):
            device_lower = "cuda"
        elif device_lower == "cpu":
            device_lower = "cpu"
        else:
            logger.warning(f"Unknown device '{device}', defaulting to 'cpu'")
            return "cpu"
        
        # Check CUDA availability if requested
        if device_lower == "cuda":
            if not TORCH_AVAILABLE:
                logger.warning(
                    "PyTorch is not available. Cannot check CUDA support. Falling back to CPU."
                )
                return "cpu"
            
            if not torch.cuda.is_available():
                logger.warning(
                    f"CUDA was requested but is not available. "
                    f"This could be because:\n"
                    f"  1. PyTorch was installed without CUDA support (CPU-only version)\n"
                    f"  2. No NVIDIA GPU is available\n"
                    f"  3. CUDA drivers are not installed\n"
                    f"\n"
                    f"Falling back to CPU. To use CUDA, install PyTorch with CUDA support:\n"
                    f"  Visit: https://pytorch.org/get-started/locally/\n"
                    f"  Or run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
                    f"\n"
                    f"Note: Embeddings will be slower on CPU, but still functional."
                )
                return "cpu"
            
            # CUDA is available
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "Unknown"
            logger.info(
                f"CUDA is available: {cuda_device_count} GPU(s) detected. "
                f"Using GPU: {cuda_device_name}"
            )
            return "cuda"
        
        # CPU requested
        if device_lower == "cpu":
            logger.info("Using CPU for embeddings")
            return "cpu"
        
        # Default fallback (should not reach here, but just in case)
        logger.warning(f"Unexpected device value '{device}', defaulting to 'cpu'")
        return "cpu"
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
