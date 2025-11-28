"""
Custom exceptions.

Application-specific exception classes.
"""


class BaseAppException(Exception):
    """Base exception for all application-specific exceptions."""
    
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ExtractionError(BaseAppException):
    """Raised when document extraction fails."""
    pass


class UnsupportedFileTypeError(BaseAppException):
    """Raised when attempting to extract from an unsupported file type."""
    pass


class FileReadError(BaseAppException):
    """Raised when a file cannot be read."""
    pass


class StorageError(BaseAppException):
    """Raised when storage operations fail."""
    pass


class VisionProcessingError(BaseAppException):
    """Raised when vision processing fails."""
    pass