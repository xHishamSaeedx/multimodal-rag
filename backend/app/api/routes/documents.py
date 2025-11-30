"""
Document CRUD endpoints.

GET /api/v1/documents - List all documents
GET /api/v1/documents/{object_key} - Download a document
DELETE /api/v1/documents/{object_key} - Delete a document
"""
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
from app.repositories.document_repository import DocumentRepository, RepositoryError
from app.repositories.vector_repository import VectorRepository, VectorRepositoryError
from app.repositories.sparse_repository import SparseRepository, SparseRepositoryError
from app.repositories.graph_repository import GraphRepository
from app.core.config import settings
from app.utils.exceptions import StorageError
from app.utils.logging import get_logger

logger = get_logger(__name__)

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
    Delete a document by object key with cascade deletion.
    
    This performs cascade deletion across all storage systems:
    1. Deletes vectors from Qdrant (text chunks)
    1.5. Deletes table vectors from Qdrant (table_chunks collection)
    1.6. Deletes image vectors from Qdrant (image_chunks collection)
    2. Deletes chunks from Elasticsearch (BM25 index, includes text, table, and image chunks)
    3. Deletes images from Supabase Storage (document-images bucket)
    4. Deletes document, chunks, tables, and images from Supabase database
    5. Deletes document graph from Neo4j (knowledge graph) - all related nodes and relationships
    6. Deletes file from MinIO (data lake)
    
    Args:
        object_key: Object key (path) of the document to delete
    
    Returns:
        DocumentDeleteResponse confirming deletion
    """
    try:
        # URL decode the object key
        object_key = unquote(object_key)
        
        storage = MinIOStorage()
        doc_repo = DocumentRepository()
        vector_repo = VectorRepository()
        sparse_repo = SparseRepository()
        from app.services.storage.supabase_storage import SupabaseImageStorage
        image_storage = SupabaseImageStorage()
        
        # Check if document exists in MinIO
        document_exists_in_storage = storage.document_exists(object_key)
        
        # Find document in Supabase by source_path
        document = doc_repo.get_document_by_source_path(object_key)
        
        # If document doesn't exist in either place, return 404
        if not document_exists_in_storage and not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "success": False,
                    "error": f"Document not found: {object_key}",
                    "details": {"object_key": object_key},
                },
            )
        
        deletion_errors = []
        deletion_success = []
        
        # 1. Delete from Qdrant (vectors) if document exists in Supabase
        if document:
            try:
                vector_repo.delete_vectors_by_document(document.id)
                deletion_success.append("Qdrant vectors")
                logger.info(f"Deleted vectors from Qdrant for document: {document.id}")
            except VectorRepositoryError as e:
                deletion_errors.append(f"Qdrant: {e.message}")
                logger.error(f"Failed to delete vectors from Qdrant: {e.message}", exc_info=True)
            except Exception as e:
                deletion_errors.append(f"Qdrant: {str(e)}")
                logger.error(f"Unexpected error deleting vectors from Qdrant: {str(e)}", exc_info=True)
            
            # 1.5. Delete table vectors from Qdrant (table_chunks collection)
            try:
                vector_repo.delete_table_vectors_by_document(document.id)
                deletion_success.append("Qdrant table vectors")
                logger.info(f"Deleted table vectors from Qdrant for document: {document.id}")
            except VectorRepositoryError as e:
                deletion_errors.append(f"Qdrant table_chunks: {e.message}")
                logger.error(f"Failed to delete table vectors from Qdrant: {e.message}", exc_info=True)
            except Exception as e:
                deletion_errors.append(f"Qdrant table_chunks: {str(e)}")
                logger.error(f"Unexpected error deleting table vectors from Qdrant: {str(e)}", exc_info=True)
            
            # 1.6. Delete image vectors from Qdrant (image_chunks collection)
            try:
                vector_repo.delete_image_vectors_by_document(document.id)
                deletion_success.append("Qdrant image vectors")
                logger.info(f"Deleted image vectors from Qdrant for document: {document.id}")
            except VectorRepositoryError as e:
                deletion_errors.append(f"Qdrant image_chunks: {e.message}")
                logger.error(f"Failed to delete image vectors from Qdrant: {e.message}", exc_info=True)
            except Exception as e:
                deletion_errors.append(f"Qdrant image_chunks: {str(e)}")
                logger.error(f"Unexpected error deleting image vectors from Qdrant: {str(e)}", exc_info=True)
            
            # 2. Delete from Elasticsearch (BM25 index)
            try:
                sparse_repo.delete_chunks_by_document(document.id)
                deletion_success.append("Elasticsearch chunks")
                logger.info(f"Deleted chunks from Elasticsearch for document: {document.id}")
            except SparseRepositoryError as e:
                deletion_errors.append(f"Elasticsearch: {e.message}")
                logger.error(f"Failed to delete chunks from Elasticsearch: {e.message}", exc_info=True)
            except Exception as e:
                deletion_errors.append(f"Elasticsearch: {str(e)}")
                logger.error(f"Unexpected error deleting chunks from Elasticsearch: {str(e)}", exc_info=True)
            
            # 3. Delete images from Supabase Storage (document-images bucket)
            try:
                image_storage.delete_images_by_document(document.id)
                deletion_success.append("Supabase Storage images")
                logger.info(f"Deleted images from Supabase Storage for document: {document.id}")
            except Exception as e:
                deletion_errors.append(f"Supabase Storage: {str(e)}")
                logger.error(f"Failed to delete images from Supabase Storage: {str(e)}", exc_info=True)
            
            # 4. Delete from Supabase (document, chunks, tables, and images)
            try:
                doc_repo.delete_document(document.id)
                deletion_success.append("Supabase document, chunks, tables, and images")
                logger.info(f"Deleted document, chunks, tables, and images from Supabase: {document.id}")
            except RepositoryError as e:
                deletion_errors.append(f"Supabase: {e.message}")
                logger.error(f"Failed to delete from Supabase: {e.message}", exc_info=True)
            except Exception as e:
                deletion_errors.append(f"Supabase: {str(e)}")
                logger.error(f"Unexpected error deleting from Supabase: {str(e)}", exc_info=True)
            
            # 5. Delete from Neo4j (knowledge graph) if enabled
            if settings.neo4j_enabled:
                try:
                    graph_repo = GraphRepository()
                    graph_repo.delete_document_graph(str(document.id))
                    deletion_success.append("Neo4j knowledge graph")
                    logger.info(f"Deleted document graph from Neo4j: {document.id}")
                except Exception as e:
                    deletion_errors.append(f"Neo4j: {str(e)}")
                    logger.error(f"Failed to delete from Neo4j: {str(e)}", exc_info=True)
        
        # 6. Delete from MinIO (data lake)
        if document_exists_in_storage:
            try:
                storage.delete_document(object_key)
                deletion_success.append("MinIO data lake")
                logger.info(f"Deleted document from MinIO: {object_key}")
            except StorageError as e:
                deletion_errors.append(f"MinIO: {e.message}")
                logger.error(f"Failed to delete from MinIO: {e.message}", exc_info=True)
            except Exception as e:
                deletion_errors.append(f"MinIO: {str(e)}")
                logger.error(f"Unexpected error deleting from MinIO: {str(e)}", exc_info=True)
        
        # If there were any errors, log them but still return success if at least one deletion succeeded
        if deletion_errors:
            logger.warning(
                f"Partial deletion completed for {object_key}. "
                f"Success: {', '.join(deletion_success)}. "
                f"Errors: {'; '.join(deletion_errors)}"
            )
        
        # If all deletions failed, return error
        if not deletion_success and deletion_errors:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "success": False,
                    "error": f"Failed to delete document from all storage systems: {'; '.join(deletion_errors)}",
                    "details": {"object_key": object_key, "errors": deletion_errors},
                },
            )
        
        # Build success message
        success_message = f"Document deleted successfully: {object_key}"
        if deletion_success:
            success_message += f" (deleted from: {', '.join(deletion_success)})"
        if deletion_errors:
            success_message += f" (warnings: {'; '.join(deletion_errors)})"
        
        return DocumentDeleteResponse(
            message=success_message,
            object_key=object_key,
        )
    
    except HTTPException:
        raise
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
