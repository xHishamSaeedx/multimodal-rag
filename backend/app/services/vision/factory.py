"""
Factory for creating vision processors.

Implements the Factory pattern to create appropriate vision processors
based on configuration.
"""

import logging
from typing import Optional

from app.services.vision.processor import VisionProcessor
from app.core.config import settings

logger = logging.getLogger(__name__)


class VisionProcessorFactory:
    """
    Factory for creating vision processors.
    
    Supports multiple processing modes:
    - "captioning": Generate captions using local models (BLIP-2, etc.)
    - "vision_llm": Use Vision LLM APIs (GPT-4V, Claude, etc.)
    """
    
    @staticmethod
    def create_processor(mode: Optional[str] = None) -> VisionProcessor:
        """
        Create vision processor based on configuration.
        
        Args:
            mode: Processing mode ("captioning" or "vision_llm").
                  If None, uses vision_processing_mode from settings.
        
        Returns:
            VisionProcessor instance
        
        Raises:
            ValueError: If mode is unknown
            ImportError: If required dependencies are missing
        """
        mode = mode or getattr(settings, "vision_processing_mode", "captioning")
        
        logger.info(f"Creating vision processor with mode: {mode}")
        
        if mode == "captioning":
            from app.services.vision.captioning_processor import CaptioningProcessor
            return CaptioningProcessor()
        
        elif mode == "vision_llm":
            from app.services.vision.vision_llm_processor import VisionLLMProcessor
            provider = getattr(settings, "vision_llm_provider", "openai")
            model = getattr(settings, "vision_llm_model", "gpt-4-vision-preview")
            return VisionLLMProcessor(provider=provider, model=model)
        
        else:
            raise ValueError(
                f"Unknown vision processing mode: {mode}. "
                f"Supported modes: 'captioning', 'vision_llm'"
            )
    
    @staticmethod
    def get_available_modes() -> list:
        """
        Get list of available processing modes.
        
        Returns:
            List of mode names
        """
        modes = ["captioning"]
        
        # Check if Vision LLM is available
        try:
            import openai
            modes.append("vision_llm")
        except ImportError:
            pass
        
        return modes

