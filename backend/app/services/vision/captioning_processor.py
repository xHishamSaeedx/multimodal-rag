"""
Captioning processor for generating image descriptions.

Uses BLIP-2 or similar models to generate captions for images.
This is a cost-effective approach that runs locally.
"""

import logging
import io
from typing import Optional
from PIL import Image

from app.services.vision.processor import VisionProcessor, VisionResult
from app.core.config import settings
from app.utils.exceptions import VisionProcessingError

logger = logging.getLogger(__name__)

# Try to import transformers (optional dependency)
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    BlipProcessor = None
    BlipForConditionalGeneration = None
    torch = None


class CaptioningProcessor(VisionProcessor):
    """
    Generate captions using BLIP-2 or similar models.
    
    This processor runs locally and generates captions during ingestion,
    storing them for later use by the LLM.
    
    Models supported:
    - Salesforce/blip-image-captioning-base (small, fast)
    - Salesforce/blip-image-captioning-large (better quality)
    - Salesforce/blip2-opt-2.7b (best quality, larger)
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the captioning processor.
        
        Args:
            model_name: HuggingFace model name. If None, uses config default.
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "transformers library not available. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name or getattr(
            settings, "captioning_model", "Salesforce/blip-image-captioning-base"
        )
        self._processor = None
        self._model = None
        self._device = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the model (only when first needed)."""
        if self._model_loaded:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise VisionProcessingError(
                "transformers library not available. "
                "Install with: pip install transformers torch",
                {"model_name": self.model_name}
            )
        
        try:
            logger.info(f"Loading captioning model: {self.model_name}")
            
            # Determine device
            if torch and torch.cuda.is_available():
                self._device = "cuda"
                logger.info("Using GPU for captioning")
            else:
                self._device = "cpu"
                logger.info("Using CPU for captioning (slower)")
            
            # Load processor and model
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            )
            
            if self._device == "cuda":
                self._model = self._model.to(self._device)
            
            self._model.eval()  # Set to evaluation mode
            self._model_loaded = True
            
            logger.info(f"✓ Loaded captioning model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load captioning model: {e}", exc_info=True)
            raise VisionProcessingError(
                f"Failed to load captioning model {self.model_name}: {str(e)}",
                {"model_name": self.model_name, "error": str(e)}
            ) from e
    
    def process_image(
        self,
        image_bytes: bytes,
        query: Optional[str] = None,
        context: Optional[str] = None,
    ) -> VisionResult:
        """
        Generate a caption for the image.
        
        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)
            query: Optional query/question (not used in captioning mode)
            context: Optional context (not used in captioning mode)
        
        Returns:
            VisionResult with generated caption
        
        Raises:
            VisionProcessingError: If processing fails
        """
        if not self.is_available():
            raise VisionProcessingError(
                "Captioning processor is not available. "
                "Install transformers and torch: pip install transformers torch",
                {}
            )
        
        try:
            self._load_model()
            
            # Load image from bytes
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if necessary
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception as e:
                raise VisionProcessingError(
                    f"Failed to load image: {str(e)}",
                    {"error": str(e)}
                ) from e
            
            # Generate caption
            try:
                inputs = self._processor(image, return_tensors="pt")
                
                # Move inputs to device
                if self._device == "cuda":
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                # Generate caption
                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        max_length=100,
                        num_beams=3,
                        early_stopping=True
                    )
                
                # Decode caption
                caption = self._processor.decode(out[0], skip_special_tokens=True)
                
                logger.debug(f"Generated caption: {caption[:100]}...")
                
                return VisionResult(
                    description=caption,
                    metadata={
                        "model": self.model_name,
                        "mode": "captioning",
                        "device": self._device,
                    },
                    confidence=None,  # BLIP doesn't provide confidence scores
                )
                
            except Exception as e:
                raise VisionProcessingError(
                    f"Failed to generate caption: {str(e)}",
                    {"model_name": self.model_name, "error": str(e)}
                ) from e
        
        except VisionProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in captioning: {e}", exc_info=True)
            raise VisionProcessingError(
                f"Unexpected error in captioning: {str(e)}",
                {"error": str(e)}
            ) from e
    
    def get_mode(self) -> str:
        """Return processing mode name."""
        return "captioning"
    
    def is_available(self) -> bool:
        """
        Check if captioning processor is available.
        
        Returns:
            True if transformers and torch are available
        """
        return TRANSFORMERS_AVAILABLE
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            return {"status": "not_loaded", "model_name": self.model_name}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self._device,
            "model_type": "BLIP",
        }

