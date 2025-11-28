"""
Base interface for vision processing strategies.

Defines the contract that all vision processors must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class VisionResult:
    """
    Result from vision processing.
    
    Attributes:
        description: Generated description or answer about the image
        metadata: Additional metadata about the processing
        confidence: Optional confidence score (0.0 to 1.0)
    """
    description: str
    metadata: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None


class VisionProcessor(ABC):
    """
    Base class for vision processing strategies.
    
    All vision processors must implement this interface to ensure
    consistent behavior across different processing modes.
    """
    
    @abstractmethod
    def process_image(
        self,
        image_bytes: bytes,
        query: Optional[str] = None,
        context: Optional[str] = None,
    ) -> VisionResult:
        """
        Process an image and return a description or answer.
        
        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)
            query: Optional query/question about the image
            context: Optional surrounding context (e.g., document text)
        
        Returns:
            VisionResult containing description and metadata
        
        Raises:
            VisionProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def get_mode(self) -> str:
        """
        Return the processing mode name.
        
        Returns:
            Mode name (e.g., "captioning", "vision_llm")
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the processor is available and ready to use.
        
        Returns:
            True if processor can be used, False otherwise
        """
        pass

