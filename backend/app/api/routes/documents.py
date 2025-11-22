"""
Document CRUD endpoints.

GET /api/v1/documents - List all documents
GET /api/v1/documents/{object_key} - Download a document
DELETE /api/v1/documents/{object_key} - Delete a document
"""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, Query, Path as FastAPIPath
from fastapi.responses import Response
from typing import Optional
from urllib.parse import unquote

from app.api.schemas import (
    DocumentListResponse,
    DocumentInfo,
    DocumentDeleteResponse,
    ErrorResponse,
)
from app.services.storage import MinIOStorage
from app.utils.exceptions import StorageError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def list_documents(
    prefix: Optional[str] = Query(
        default="raw_documents/",
        description="Prefix to filter documents (e.g., 'raw_documents/api/')",
    ),
    source: Optional[str] = Query(
        default=None,
        description="Filter by source (e.g., 'api', 'upload')",
    ),
    document_type: Optional[str] = Query(
        default=None,
        description="Filter by document type (e.g., 'pdf', 'docx')",
    ),
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of documents to return",
    ),
) -> DocumentListResponse:
    """
    List all documents in storage.
    
    Args:
        prefix: Prefix to filter documents
        source: Optional source filter
        document_type: Optional document type filter
        limit: Maximum number of documents to return
    
    Returns:
        DocumentListResponse with list of documents
    """
    try:
        storage = MinIOStorage()
        
        # Build prefix if source or document_type filters are provided
        filter_prefix = prefix
        if source:
            filter_prefix = f"raw_documents/{source}/"
            if document_type:
                filter_prefix = f"raw_documents/{source}/{document_type}/"
        elif document_type:
            # If only document_type is provided, we need to search through all sources
            # For now, we'll just use the default prefix and filter client-side
            filter_prefix = prefix
        
        # List documents
        documents = storage.list_documents(
            prefix=filter_prefix,
            recursive=True,
            limit=limit,
        )
        
        # Apply client-side filtering if needed
        if source or document_type:
            filtered_docs = []
            for doc in documents:
                if source and doc["source"] != source:
                    continue
                if document_type and doc["document_type"] != document_type:
                    continue
                filtered_docs.append(doc)
            documents = filtered_docs
        
        # Convert to DocumentInfo models
        document_infos = [
            DocumentInfo(
                object_key=doc["object_key"],
                filename=doc["filename"],
                size=doc["size"],
                last_modified=doc["last_modified"],
                content_type=doc["content_type"],
                source=doc["source"],
                document_type=doc["document_type"],
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=document_infos,
            count=len(document_infos),
        )
    
    except StorageError as e:
        logger.error(f"StorageError: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Unexpected error listing documents: {str(e)}",
                "details": {},
            },
        )


@router.get(
    "/{object_key:path}",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_document(
    object_key: str = FastAPIPath(
        ...,
        description="Object key (path) of the document to download",
    ),
    download: bool = Query(
        default=False,
        description="Force download (adds Content-Disposition header)",
    ),
) -> Response:
    """
    Download a document by object key.
    
    Args:
        object_key: Object key (path) of the document
        download: Whether to force download or display inline
    
    Returns:
        File response with document content
    """
    try:
        # URL decode the object key
        object_key = unquote(object_key)
        
        storage = MinIOStorage()
        
        # Check if document exists
        if not storage.document_exists(object_key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "success": False,
                    "error": f"Document not found: {object_key}",
                    "details": {"object_key": object_key},
                },
            )
        
        # Download document
        file_bytes = storage.download_document(object_key)
        
        # Get filename from object key
        filename = Path(object_key).name
        
        # Get content type
        content_type = storage._infer_content_type(filename)
        
        # Prepare headers
        headers = {}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        else:
            headers["Content-Disposition"] = f'inline; filename="{filename}"'
        
        return Response(
            content=file_bytes,
            media_type=content_type,
            headers=headers,
        )
    
    except HTTPException:
        raise
    except StorageError as e:
        logger.error(f"StorageError: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Unexpected error downloading document: {str(e)}",
                "details": {"object_key": object_key},
            },
        )


@router.delete(
    "/{object_key:path}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def delete_document(
    object_key: str = FastAPIPath(
        ...,
        description="Object key (path) of the document to delete",
    ),
) -> DocumentDeleteResponse:
    """
    Delete a document by object key.
    
    Args:
        object_key: Object key (path) of the document to delete
    
    Returns:
        DocumentDeleteResponse confirming deletion
    """
    try:
        # URL decode the object key
        object_key = unquote(object_key)
        
        storage = MinIOStorage()
        
        # Check if document exists
        if not storage.document_exists(object_key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "success": False,
                    "error": f"Document not found: {object_key}",
                    "details": {"object_key": object_key},
                },
            )
        
        # Delete document
        storage.delete_document(object_key)
        
        return DocumentDeleteResponse(
            message=f"Document deleted successfully: {object_key}",
            object_key=object_key,
        )
    
    except HTTPException:
        raise
    except StorageError as e:
        logger.error(f"StorageError: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Unexpected error deleting document: {str(e)}",
                "details": {"object_key": object_key},
            },
        )
