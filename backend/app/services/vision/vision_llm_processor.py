"""
Vision LLM processor for real-time image understanding.

Uses Vision LLM APIs (GPT-4V, Google Gemini, etc.) to understand images
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

# Try to import Google Generative AI (optional dependency)
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None


class VisionLLMProcessor(VisionProcessor):
    """
    Use Vision LLM API for real-time image understanding.
    
    Supports:
    - OpenAI GPT-4o (gpt-4o, gpt-4o-mini)
    - Google Gemini (gemini-1.5-pro, gemini-1.5-flash)
    
    This processor is used at query time to analyze images
    based on the user's question.
    """
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the Vision LLM processor.
        
        Args:
            provider: API provider ("openai" or "google")
            model: Model name. If None, uses config default.
        """
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self._client = None
        self._api_key = None
        
        # Validate provider
        if self.provider not in ["openai", "google"]:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                f"Supported providers: 'openai', 'google'"
            )
        
        # Check availability
        if self.provider == "openai" and not OPENAI_AVAILABLE:
            logger.warning(
                "OpenAI library not available. "
                "Install with: pip install openai"
            )
        elif self.provider == "google" and not GOOGLE_AVAILABLE:
            logger.warning(
                "Google Generative AI library not available. "
                "Install with: pip install google-generativeai"
            )
        
        # Pre-warm the client at initialization (not lazy)
        # This ensures the API client is ready immediately, not on first use
        try:
            if self.is_available():
                self._get_client()  # Initialize client immediately
                logger.info(f"Pre-warmed {self.provider} Vision LLM client (model: {self.model})")
            else:
                logger.warning(
                    f"{self.provider} Vision LLM not available - missing library or API key. "
                    f"Vision processing will fail at runtime."
                )
        except Exception as e:
            logger.warning(
                f"Failed to pre-warm {self.provider} Vision LLM client: {e}. "
                f"It will be initialized on first use (slower)."
            )
    
    def _get_default_model(self) -> str:
        """Get default model name based on provider."""
        if self.provider == "openai":
            return getattr(settings, "vision_llm_model", "gpt-4o")
        elif self.provider == "google":
            return getattr(settings, "vision_llm_model", "gemini-1.5-pro")
        return "gpt-4o"
    
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
            
            elif self.provider == "google":
                if not GOOGLE_AVAILABLE:
                    raise VisionProcessingError(
                        "Google Generative AI library not available. "
                        "Install with: pip install google-generativeai",
                        {"provider": self.provider}
                    )
                
                self._api_key = getattr(settings, "google_api_key", None)
                if not self._api_key:
                    raise VisionProcessingError(
                        "Google API key not configured. "
                        "Set GOOGLE_API_KEY in your .env file.",
                        {"provider": self.provider}
                    )
                
                genai.configure(api_key=self._api_key)
                self._client = genai.GenerativeModel(self.model)
                logger.info(f"Initialized Google Gemini client for Vision LLM (model: {self.model})")
        
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
            elif self.provider == "google":
                return self._process_google(client, image_b64, image_mime, prompt)
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
    
    def _process_google(self, client, image_b64: str, image_mime: str, prompt: str) -> VisionResult:
        """Process image with Google Gemini."""
        try:
            from PIL import Image
            import io
            
            # Decode base64 image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Generate content with image and prompt
            response = client.generate_content([prompt, image])
            
            description = response.text
            
            # Extract token usage if available
            tokens_used = None
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                tokens_used = usage.prompt_token_count + usage.candidates_token_count if usage else None
            
            return VisionResult(
                description=description,
                metadata={
                    "provider": "google",
                    "model": self.model,
                    "mode": "vision_llm",
                    "tokens_used": tokens_used,
                },
                confidence=None,  # Google doesn't provide confidence scores
            )
        
        except Exception as e:
            raise VisionProcessingError(
                f"Google Gemini API error: {str(e)}",
                {"provider": "google", "model": self.model, "error": str(e)}
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
        elif self.provider == "google":
            return GOOGLE_AVAILABLE and bool(getattr(settings, "google_api_key", None))
        return False

