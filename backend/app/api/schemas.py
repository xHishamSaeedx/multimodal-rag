"""
Pydantic schemas for request/response models.

This module contains all API request and response models.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    
    success: bool
    message: str
    document_id: Optional[str] = None
    object_key: Optional[str] = None  # MinIO object key (path) where document is stored
    file_name: str
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    extracted_text_length: int
    metadata: Optional[Dict[str, Any]] = None
    extracted_at: datetime


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None


class DocumentInfo(BaseModel):
    """Document information model."""
    
    object_key: str
    filename: str
    size: int
    last_modified: Optional[str] = None
    content_type: str
    source: str
    document_type: str


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    
    success: bool = True
    documents: List[DocumentInfo]
    count: int


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""
    
    success: bool = True
    message: str
    object_key: str