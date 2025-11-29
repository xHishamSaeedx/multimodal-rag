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


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str
    limit: Optional[int] = 10  # Number of chunks to retrieve
    include_sources: bool = True  # Whether to include source citations
    filter_conditions: Optional[Dict[str, Any]] = None  # Optional filters (document_id, document_type, etc.)
    enable_sparse: Optional[bool] = True  # Whether to use BM25 sparse retriever
    enable_dense: Optional[bool] = True  # Whether to use dense text retriever
    enable_table: Optional[bool] = True  # Whether to use table retriever
    enable_image: Optional[bool] = True  # Whether to use image retriever


class SourceInfo(BaseModel):
    """Source citation information."""
    
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    chunk_text: str  # Truncated preview
    full_chunk_text: str  # Full chunk text
    citation: str
    metadata: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None  # Supabase storage path for image chunks
    image_url: Optional[str] = None  # Signed URL for image access (temporary)


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    success: bool = True
    query: str
    answer: str
    sources: List[SourceInfo] = []
    chunks_used: List[str] = []  # List of chunk IDs used
    model: str  # Model used for generation
    tokens_used: Optional[Dict[str, int]] = None  # Token usage (prompt_tokens, completion_tokens, total_tokens)
    retrieval_stats: Optional[Dict[str, Any]] = None  # Retrieval statistics (chunks_found, etc.)