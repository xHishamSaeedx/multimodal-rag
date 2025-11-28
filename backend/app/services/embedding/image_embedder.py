"""
Image embedding service.

Generates image embeddings using CLIP or SigLIP models.
Supports both CLIP (sentence-transformers) and SigLIP (timm) models.
"""

import logging
from typing import List, Optional
import numpy as np
from io import BytesIO

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    timm = None
    TIMM_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

from app.core.config import settings
from app.utils.exceptions import BaseAppException

logger = logging.getLogger(__name__)


class EmbeddingError(BaseAppException):
    """Raised when embedding generation fails."""
    pass


class ImageEmbedder:
    """
    Service for generating image embeddings using CLIP or SigLIP.
    
    Supports:
    - CLIP: via sentence-transformers (512 or 768 dim)
    - SigLIP: via timm (768 or 1024 dim)
    
    Model Options:
    - CLIP: 'sentence-transformers/clip-ViT-B-32' (512 dim) or 'sentence-transformers/clip-ViT-L-14' (768 dim)
    - SigLIP: 'vit_base_patch16_siglip_224' (768 dim) or 'vit_large_patch16_siglip_384' (1024 dim)
    """
    
    def __init__(
        self,
        model_type: str = "siglip",  # "clip" or "siglip" (default: "siglip" to match POC)
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the image embedder.
        
        Args:
            model_type: Model type - "clip" or "siglip" (default: "siglip")
            model_name: Specific model name (default: based on model_type)
            device: Device to use ("cpu" or "cuda", default: from config)
        """
        self.model_type = model_type.lower()
        
        if self.model_type not in ("clip", "siglip"):
            raise EmbeddingError(
                f"Invalid model_type: {model_type}. Must be 'clip' or 'siglip'",
                {"model_type": model_type},
            )
        
        # Set default model names (matching extract_images.py POC)
        if model_name is None:
            if self.model_type == "clip":
                # Use CLIP base model (512 dim) for balance of speed and quality
                model_name = "sentence-transformers/clip-ViT-B-32"
            else:  # siglip
                # Use SigLIP large model (1024 dim) for better quality (matching POC)
                model_name = "vit_large_patch16_siglip_384"  # 1024 dimensions
        
        self.model_name = model_name
        requested_device = device or settings.embedding_device
        self.device = self._validate_device(requested_device)
        
        # Initialize model
        try:
            logger.info(f"Loading {self.model_type.upper()} image embedding model: {self.model_name} on {self.device}")
            
            if self.model_type == "clip":
                self._init_clip_model()
            else:  # siglip
                self._init_siglip_model()
            
            logger.info(
                f"Loaded {self.model_type.upper()} model: {self.model_name} "
                f"(dimension: {self.embedding_dim}, device: {self.device})"
            )
        except Exception as e:
            logger.error(f"Failed to load image embedding model: {str(e)}")
            raise EmbeddingError(
                f"Failed to load image embedding model {self.model_name}: {str(e)}",
                {"model_name": self.model_name, "device": self.device, "error": str(e)},
            ) from e
    
    def _init_clip_model(self):
        """Initialize CLIP model via sentence-transformers."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise EmbeddingError(
                "sentence-transformers is not installed. Install it with: pip install sentence-transformers",
                {},
            )
        
        if not PIL_AVAILABLE:
            raise EmbeddingError(
                "Pillow is not installed. Install it with: pip install pillow",
                {},
            )
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def _init_siglip_model(self):
        """Initialize SigLIP model via timm."""
        if not TIMM_AVAILABLE:
            raise EmbeddingError(
                "timm is not installed. Install it with: pip install timm torch torchvision",
                {},
            )
        
        if not PIL_AVAILABLE:
            raise EmbeddingError(
                "Pillow is not installed. Install it with: pip install pillow",
                {},
            )
        
        if not TORCH_AVAILABLE:
            raise EmbeddingError(
                "PyTorch is not installed. Install it with: pip install torch",
                {},
            )
        
        # Load SigLIP model from timm
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)  # num_classes=0 for features only
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create image preprocessing transform
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        
        # Get embedding dimension
        # For vit_large_patch16_siglip_384: 1024 dimensions
        # For vit_base_patch16_siglip_224: 768 dimensions
        # Default fallback based on model name
        if 'large' in self.model_name.lower():
            default_dim = 1024
        else:
            default_dim = 768
        self.embedding_dim = getattr(self.model, 'num_features', default_dim)
    
    def embed_image(self, image_bytes: bytes) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image_bytes: Image file content as bytes
        
        Returns:
            Embedding vector as list of floats
        
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not image_bytes:
                logger.warning("Empty image bytes provided for embedding")
                return [0.0] * self.embedding_dim
            
            if self.model_type == "clip":
                return self._embed_clip(image_bytes)
            else:  # siglip
                return self._embed_siglip(image_bytes)
                
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}", exc_info=True)
            raise EmbeddingError(
                f"Failed to generate image embedding: {str(e)}",
                {"image_size": len(image_bytes) if image_bytes else 0, "error": str(e)},
            ) from e
    
    def _embed_clip(self, image_bytes: bytes) -> List[float]:
        """Generate embedding using CLIP."""
        # CLIP via sentence-transformers can encode directly from bytes or file path
        # But we need to use PIL Image for consistency
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        embedding = self.model.encode(
            image,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,
        )
        
        return embedding.tolist()
    
    def _embed_siglip(self, image_bytes: bytes) -> List[float]:
        """Generate embedding using SigLIP."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Apply transform
        if hasattr(self, 'transform'):
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            # Fallback transform
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            # timm models return features directly
            features = self.model.forward_features(input_tensor)
            
            # Handle different feature formats
            if isinstance(features, (list, tuple)):
                features = features[-1]  # Get last layer
            elif isinstance(features, dict):
                # Get the main feature tensor
                if 'x_norm_clstoken' in features:
                    embedding = features['x_norm_clstoken']
                elif 'x_norm_patchtokens' in features:
                    embedding = features['x_norm_patchtokens'].mean(dim=1)
                else:
                    embedding = list(features.values())[0]
                    if len(embedding.shape) > 2:
                        embedding = embedding.mean(dim=1)
            else:
                # Tensor output - use CLS token (first token) or global pool
                if len(features.shape) == 3:
                    # [batch, seq_len, dim] - use CLS token (first token)
                    embedding = features[:, 0, :]
                else:
                    embedding = features
            
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding_np = embedding.cpu().numpy().flatten()
        
        return embedding_np.tolist()
    
    def embed_batch(
        self,
        image_bytes_list: List[bytes],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple images in batch.
        
        Args:
            image_bytes_list: List of image file contents as bytes
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors
        
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not image_bytes_list:
                logger.warning("Empty images list provided for embedding")
                return []
            
            if self.model_type == "clip":
                return self._embed_batch_clip(image_bytes_list, show_progress)
            else:  # siglip
                return self._embed_batch_siglip(image_bytes_list, show_progress)
                
        except Exception as e:
            logger.error(f"Error generating batch image embeddings: {str(e)}", exc_info=True)
            raise EmbeddingError(
                f"Failed to generate batch image embeddings: {str(e)}",
                {"image_count": len(image_bytes_list), "error": str(e)},
            ) from e
    
    def _embed_batch_clip(self, image_bytes_list: List[bytes], show_progress: bool) -> List[List[float]]:
        """Generate embeddings using CLIP in batch."""
        # Convert bytes to PIL Images
        images = [Image.open(BytesIO(img_bytes)).convert("RGB") for img_bytes in image_bytes_list]
        
        # Generate embeddings in batch
        embeddings = self.model.encode(
            images,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        
        return embeddings.tolist()
    
    def _embed_batch_siglip(self, image_bytes_list: List[bytes], show_progress: bool) -> List[List[float]]:
        """Generate embeddings using SigLIP in batch."""
        # Convert bytes to PIL Images and apply transform
        images = [Image.open(BytesIO(img_bytes)).convert("RGB") for img_bytes in image_bytes_list]
        
        if hasattr(self, 'transform'):
            input_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        else:
            # Fallback transform
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_tensors = torch.stack([transform(img) for img in images]).to(self.device)
        
        # Generate embeddings in batch
        embeddings_list = []
        batch_size = settings.embedding_batch_size
        
        for i in range(0, len(input_tensors), batch_size):
            batch = input_tensors[i:i + batch_size]
            
            with torch.no_grad():
                features = self.model.forward_features(batch)
                
                # Handle different feature formats (same as single image)
                if isinstance(features, (list, tuple)):
                    features = features[-1]
                elif isinstance(features, dict):
                    if 'x_norm_clstoken' in features:
                        embedding = features['x_norm_clstoken']
                    elif 'x_norm_patchtokens' in features:
                        embedding = features['x_norm_patchtokens'].mean(dim=1)
                    else:
                        embedding = list(features.values())[0]
                        if len(embedding.shape) > 2:
                            embedding = embedding.mean(dim=1)
                else:
                    if len(features.shape) == 3:
                        embedding = features[:, 0, :]
                    else:
                        embedding = features
                
                # Normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding_np = embedding.cpu().numpy()
                
                embeddings_list.extend(embedding_np.tolist())
        
        return embeddings_list
    
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
                    f"CUDA was requested but is not available. Falling back to CPU."
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
            logger.info("Using CPU for image embeddings")
            return "cpu"
        
        # Default fallback
        logger.warning(f"Unexpected device value '{device}', defaulting to 'cpu'")
        return "cpu"
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim

