"""
Document ingestion endpoint.

POST /api/v1/ingest - Upload and process documents
"""
import logging
import traceback
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
from datetime import datetime

from app.api.schemas import IngestResponse, ErrorResponse
from app.services.ingestion import TextExtractor
from app.services.ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from app.services.storage import MinIOStorage
from app.utils.exceptions import (
    ExtractionError,
    UnsupportedFileTypeError,
    FileReadError,
    StorageError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload and extract text from a document.
    
    Supported file types:
    - PDF (.pdf)
    - DOCX (.docx)
    - TXT (.txt)
    - Markdown (.md, .markdown)
    
    Args:
        file: The file to upload and process
    
    Returns:
        IngestResponse with extracted content and metadata
    
    Raises:
        HTTPException: If file type is unsupported or extraction fails
    """
    # Get filename with validation
    file_name = file.filename
    if not file_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Filename is required",
                "details": {},
            },
        )
    
    try:
        # Initialize extractor
        logger.info(f"Initializing TextExtractor for file: {file_name}")
        extractor = TextExtractor()
        
        # Check if file type is supported
        if not extractor.is_supported(file_name):
            logger.warning(f"Unsupported file type: {file_name}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": f"Unsupported file type: {file_name}",
                    "details": {
                        "file_name": file_name,
                        "supported_types": list(extractor.SUPPORTED_EXTENSIONS.keys()),
                    },
                },
            )
        
        # Read file content
        logger.info(f"Reading file content: {file_name}")
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            logger.warning(f"Empty file: {file_name}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": "File is empty",
                    "details": {"file_name": file_name},
                },
            )
        
        # Execute complete ingestion pipeline
        logger.info(f"Starting ingestion pipeline for: {file_name}")
        pipeline = IngestionPipeline()
        
        result = pipeline.ingest_document(
            file_bytes=file_bytes,
            filename=file_name,
            source="api",
        )
        
        extracted_content = result["extracted_content"]
        
        # Return response
        return IngestResponse(
            success=True,
            message=f"Successfully ingested document: {file_name} ({result['chunks_count']} chunks created)",
            document_id=result["document_id"],
            object_key=result["object_key"],
            file_name=extracted_content.file_name,
            file_type=extracted_content.file_type,
            file_size=extracted_content.file_size or 0,
            page_count=extracted_content.page_count,
            extracted_text_length=len(extracted_content.text),
            metadata={
                **(extracted_content.metadata or {}),
                "chunks_count": result["chunks_count"],
                "chunking_stats": result["stats"],
            },
            extracted_at=extracted_content.extracted_at,
        )
    
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    
    except UnsupportedFileTypeError as e:
        logger.error(f"UnsupportedFileTypeError: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    
    except FileReadError as e:
        logger.error(f"FileReadError: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    
    except IngestionPipelineError as e:
        logger.error(f"IngestionPipelineError: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": str(e),
                "details": {"file_name": file_name},
            },
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
    
    except ExtractionError as e:
        logger.error(f"ExtractionError: {e.message}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error during ingestion: {str(e)}", exc_info=True)
        logger.error(f"Traceback: {error_traceback}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Unexpected error during ingestion: {str(e)}",
                "details": {
                    "file_name": file_name if 'file_name' in locals() else (file.filename if file else "unknown"),
                    "error_type": type(e).__name__,
                },
            },
        )


@router.post(
    "/batch",
    response_model=List[IngestResponse],
    status_code=status.HTTP_200_OK,
)
async def ingest_documents(
    files: List[UploadFile] = File(...),
) -> List[IngestResponse]:
    """
    Upload and extract text from multiple documents.
    
    Supported file types:
    - PDF (.pdf)
    - DOCX (.docx)
    - TXT (.txt)
    - Markdown (.md, .markdown)
    
    Args:
        files: List of files to upload and process
    
    Returns:
        List of IngestResponse objects with extracted content and metadata
    """
    results = []
    extractor = TextExtractor()
    
    for file in files:
        try:
            # Check if file type is supported
            if not extractor.is_supported(file.filename):
                results.append(
                    IngestResponse(
                        success=False,
                        message=f"Unsupported file type: {file.filename}",
                        file_name=file.filename or "unknown",
                        file_type="unknown",
                        file_size=0,
                        extracted_text_length=0,
                        extracted_at=datetime.utcnow(),
                    )
                )
                continue
            
            # Read file content
            file_bytes = await file.read()
            
            if len(file_bytes) == 0:
                results.append(
                    IngestResponse(
                        success=False,
                        message=f"File is empty: {file.filename}",
                        file_name=file.filename or "unknown",
                        file_type="unknown",
                        file_size=0,
                        extracted_text_length=0,
                        extracted_at=datetime.utcnow(),
                    )
                )
                continue
            
            # Extract text
            extracted_content = extractor.extract_from_bytes(
                file_bytes=file_bytes,
                file_name=file.filename or "unknown",
            )
            
            results.append(
                IngestResponse(
                    success=True,
                    message=f"Successfully extracted text from {file.filename}",
                    file_name=extracted_content.file_name,
                    file_type=extracted_content.file_type,
                    file_size=extracted_content.file_size or 0,
                    page_count=extracted_content.page_count,
                    extracted_text_length=len(extracted_content.text),
                    metadata=extracted_content.metadata,
                    extracted_at=extracted_content.extracted_at,
                )
            )
        
        except Exception as e:
            results.append(
                IngestResponse(
                    success=False,
                    message=f"Error processing {file.filename}: {str(e)}",
                    file_name=file.filename or "unknown",
                    file_type="unknown",
                    file_size=0,
                    extracted_text_length=0,
                    extracted_at=datetime.utcnow(),
                )
            )
    
    return results
