"""
Vision LLM processor for real-time image understanding.

Uses Vision LLM APIs (GPT-4V, Claude 3.5 Sonnet, etc.) to understand images
at query time. This provides better understanding but requires API calls.
"""

import logging
import base64
from typing import Optional

from app.services.vision.processor import VisionProcessor, VisionResult
from app.core.config import settings
from app.utils.exceptions import VisionProcessingError

logger = logging.getLogger(__name__)

# Try to import OpenAI (optional dependency)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Try to import Anthropic (optional dependency)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


class VisionLLMProcessor(VisionProcessor):
    """
    Use Vision LLM API for real-time image understanding.
    
    Supports:
    - OpenAI GPT-4V (gpt-4-vision-preview)
    - Anthropic Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)
    
    This processor is used at query time to analyze images
    based on the user's question.
    """
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the Vision LLM processor.
        
        Args:
            provider: API provider ("openai" or "anthropic")
            model: Model name. If None, uses config default.
        """
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self._client = None
        self._api_key = None
        
        # Validate provider
        if self.provider not in ["openai", "anthropic"]:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                f"Supported providers: 'openai', 'anthropic'"
            )
        
        # Check availability
        if self.provider == "openai" and not OPENAI_AVAILABLE:
            logger.warning(
                "OpenAI library not available. "
                "Install with: pip install openai"
            )
        elif self.provider == "anthropic" and not ANTHROPIC_AVAILABLE:
            logger.warning(
                "Anthropic library not available. "
                "Install with: pip install anthropic"
            )
    
    def _get_default_model(self) -> str:
        """Get default model name based on provider."""
        if self.provider == "openai":
            return getattr(settings, "vision_llm_model", "gpt-4-vision-preview")
        elif self.provider == "anthropic":
            return "claude-3-5-sonnet-20241022"
        return "gpt-4-vision-preview"
    
    def _get_client(self):
        """Lazy load API client."""
        if self._client is None:
            if self.provider == "openai":
                if not OPENAI_AVAILABLE:
                    raise VisionProcessingError(
                        "OpenAI library not available. "
                        "Install with: pip install openai",
                        {"provider": self.provider}
                    )
                
                self._api_key = getattr(settings, "openai_api_key", None)
                if not self._api_key:
                    raise VisionProcessingError(
                        "OpenAI API key not configured. "
                        "Set OPENAI_API_KEY in your .env file.",
                        {"provider": self.provider}
                    )
                
                self._client = OpenAI(api_key=self._api_key)
                logger.info("Initialized OpenAI client for Vision LLM")
            
            elif self.provider == "anthropic":
                if not ANTHROPIC_AVAILABLE:
                    raise VisionProcessingError(
                        "Anthropic library not available. "
                        "Install with: pip install anthropic",
                        {"provider": self.provider}
                    )
                
                self._api_key = getattr(settings, "anthropic_api_key", None)
                if not self._api_key:
                    raise VisionProcessingError(
                        "Anthropic API key not configured. "
                        "Set ANTHROPIC_API_KEY in your .env file.",
                        {"provider": self.provider}
                    )
                
                self._client = Anthropic(api_key=self._api_key)
                logger.info("Initialized Anthropic client for Vision LLM")
        
        return self._client
    
    def process_image(
        self,
        image_bytes: bytes,
        query: Optional[str] = None,
        context: Optional[str] = None,
    ) -> VisionResult:
        """
        Process image with Vision LLM based on query.
        
        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)
            query: Question about the image (required for best results)
            context: Optional surrounding context (document text, etc.)
        
        Returns:
            VisionResult with description/answer
        
        Raises:
            VisionProcessingError: If processing fails
        """
        if not self.is_available():
            raise VisionProcessingError(
                f"{self.provider} library not available. "
                f"Install with: pip install {self.provider}",
                {"provider": self.provider}
            )
        
        try:
            client = self._get_client()
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine image MIME type (simplified - assumes JPEG/PNG)
            image_mime = "image/jpeg"  # Default
            if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                image_mime = "image/png"
            elif image_bytes[:2] == b'\xff\xd8':
                image_mime = "image/jpeg"
            
            # Build prompt
            if query:
                prompt = query
                if context:
                    prompt = f"Context: {context}\n\nQuestion: {query}"
            else:
                prompt = "Describe this image in detail, including any text, data, charts, or visual elements."
            
            if self.provider == "openai":
                return self._process_openai(client, image_b64, image_mime, prompt)
            elif self.provider == "anthropic":
                return self._process_anthropic(client, image_b64, image_mime, prompt)
            else:
                raise VisionProcessingError(
                    f"Unknown provider: {self.provider}",
                    {"provider": self.provider}
                )
        
        except VisionProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Vision LLM processing: {e}", exc_info=True)
            raise VisionProcessingError(
                f"Failed to process image with Vision LLM: {str(e)}",
                {"provider": self.provider, "model": self.model, "error": str(e)}
            ) from e
    
    def _process_openai(self, client, image_b64: str, image_mime: str, prompt: str) -> VisionResult:
        """Process image with OpenAI GPT-4V."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_mime};base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.0,  # Deterministic for factual answers
            )
            
            description = response.choices[0].message.content
            
            return VisionResult(
                description=description,
                metadata={
                    "provider": "openai",
                    "model": self.model,
                    "mode": "vision_llm",
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None,
                },
                confidence=None,  # OpenAI doesn't provide confidence scores
            )
        
        except Exception as e:
            raise VisionProcessingError(
                f"OpenAI API error: {str(e)}",
                {"provider": "openai", "model": self.model, "error": str(e)}
            ) from e
    
    def _process_anthropic(self, client, image_b64: str, image_mime: str, prompt: str) -> VisionResult:
        """Process image with Anthropic Claude."""
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_mime,
                                    "data": image_b64,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            description = response.content[0].text
            
            return VisionResult(
                description=description,
                metadata={
                    "provider": "anthropic",
                    "model": self.model,
                    "mode": "vision_llm",
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None,
                },
                confidence=None,  # Anthropic doesn't provide confidence scores
            )
        
        except Exception as e:
            raise VisionProcessingError(
                f"Anthropic API error: {str(e)}",
                {"provider": "anthropic", "model": self.model, "error": str(e)}
            ) from e
    
    def get_mode(self) -> str:
        """Return processing mode name."""
        return "vision_llm"
    
    def is_available(self) -> bool:
        """
        Check if Vision LLM processor is available.
        
        Returns:
            True if required library and API key are available
        """
        if self.provider == "openai":
            return OPENAI_AVAILABLE and bool(getattr(settings, "openai_api_key", None))
        elif self.provider == "anthropic":
            return ANTHROPIC_AVAILABLE and bool(getattr(settings, "anthropic_api_key", None))
        return False

