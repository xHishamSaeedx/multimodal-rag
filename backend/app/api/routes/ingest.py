"""
Document ingestion endpoint.

POST /api/v1/ingest - Upload and process documents
"""
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request, Form
from typing import List, Optional
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
from app.utils.logging import get_logger
from app.utils.metrics import (
    documents_ingested_total,
    document_ingestion_duration_seconds,
    chunks_created_total,
    chunks_created_per_document,
    document_processing_errors_total,
)

logger = get_logger(__name__)

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
async def ingest_document(
    http_request: Request,  # FastAPI will inject this automatically
    file: UploadFile = File(...),
    enable_text: str = Form("true"),
    enable_tables: str = Form("true"),
    enable_images: str = Form("true"),
) -> IngestResponse:
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
    
    # Start timer for ingestion duration
    ingestion_start_time = time.time()
    file_type = "unknown"  # Will be set after extraction
    
    try:
        # Initialize extractor
        logger.info("ingestion_start", file_name=file_name, file_size=file.size if hasattr(file, 'size') else None)
        extractor = TextExtractor()
        
        # Check if file type is supported
        if not extractor.is_supported(file_name):
            logger.warning(
                "ingestion_unsupported_file_type",
                file_name=file_name,
                supported_types=list(extractor.SUPPORTED_EXTENSIONS.keys()),
            )
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
        logger.debug("ingestion_reading_file", file_name=file_name)
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            logger.warning("ingestion_empty_file", file_name=file_name)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": "File is empty",
                    "details": {"file_name": file_name},
                },
            )
        
        # Execute complete ingestion pipeline
        logger.info("ingestion_pipeline_start", file_name=file_name, file_size_bytes=len(file_bytes))
        
        # Log raw values received from form
        logger.info(
            "processor_configuration_raw",
            enable_text_raw=enable_text,
            enable_tables_raw=enable_tables,
            enable_images_raw=enable_images,
            enable_text_type=type(enable_text).__name__,
            enable_tables_type=type(enable_tables).__name__,
            enable_images_type=type(enable_images).__name__,
        )
        
        # Convert form string values to proper booleans
        # FormData sends strings, so we need to explicitly convert them
        enable_text = enable_text.lower() in ('true', '1', 'yes', 'on') if isinstance(enable_text, str) else bool(enable_text)
        enable_tables = enable_tables.lower() in ('true', '1', 'yes', 'on') if isinstance(enable_tables, str) else bool(enable_tables)
        enable_images = enable_images.lower() in ('true', '1', 'yes', 'on') if isinstance(enable_images, str) else bool(enable_images)
        
        # Log processor configuration (after conversion)
        logger.info(
            "processor_configuration",
            enable_text=enable_text,
            enable_tables=enable_tables,
            enable_images=enable_images,
            file_name=file_name,
        )
        
        # Get pre-warmed services from app state (if available)
        # These are pre-initialized in the lifespan function to avoid model loading delays
        text_embedder = getattr(http_request.app.state, "text_embedder", None)
        image_embedder = getattr(http_request.app.state, "image_embedder", None)
        vision_processor = getattr(http_request.app.state, "captioning_processor", None)
        
        if text_embedder:
            logger.debug("Using pre-warmed TextEmbedder from app state")
        if image_embedder:
            logger.debug("Using pre-warmed ImageEmbedder from app state")
        if vision_processor:
            logger.debug("Using pre-warmed CaptioningProcessor from app state")
        
        # Create pipeline with pre-warmed services (if available)
        # If services are not pre-warmed, pipeline will create new instances
        pipeline = IngestionPipeline(
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            vision_processor=vision_processor,
            enable_text=enable_text,
            enable_tables=enable_tables,
            enable_images=enable_images,
        )
        
        result = pipeline.ingest_document(
            file_bytes=file_bytes,
            filename=file_name,
            source="api",
        )
        
        extracted_content = result["extracted_content"]
        file_type = extracted_content.file_type or "unknown"
        
        # Calculate ingestion duration
        ingestion_duration = time.time() - ingestion_start_time
        
        # Record ingestion metrics
        documents_ingested_total.labels(file_type=file_type, status="success").inc()
        document_ingestion_duration_seconds.labels(file_type=file_type).observe(ingestion_duration)
        
        # Record chunks created metrics
        chunks_count = result["chunks_count"]
        chunks_created_total.labels(document_type=file_type).inc(chunks_count)
        chunks_created_per_document.labels(file_type=file_type).observe(chunks_count)
        
        logger.info(
            "ingestion_completed",
            file_name=file_name,
            document_id=result["document_id"],
            chunks_count=chunks_count,
            extracted_text_length=len(extracted_content.text),
            page_count=extracted_content.page_count,
        )
        
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
        # Record error metrics
        file_type = file_name.split('.')[-1] if '.' in file_name else "unknown"
        documents_ingested_total.labels(file_type=file_type, status="failure").inc()
        document_processing_errors_total.labels(error_type="HTTPException", file_type=file_type).inc()
        # Re-raise HTTPException as-is
        raise
    
    except UnsupportedFileTypeError as e:
        file_type = file_name.split('.')[-1] if '.' in file_name else "unknown"
        documents_ingested_total.labels(file_type=file_type, status="failure").inc()
        document_processing_errors_total.labels(error_type="UnsupportedFileTypeError", file_type=file_type).inc()
        logger.error(
            "ingestion_error",
            error_type="UnsupportedFileTypeError",
            error_message=e.message,
            file_name=file_name,
            details=e.details,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    
    except FileReadError as e:
        file_type = file_name.split('.')[-1] if '.' in file_name else "unknown"
        documents_ingested_total.labels(file_type=file_type, status="failure").inc()
        document_processing_errors_total.labels(error_type="FileReadError", file_type=file_type).inc()
        logger.error(
            "ingestion_error",
            error_type="FileReadError",
            error_message=e.message,
            file_name=file_name,
            details=e.details,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    
    except IngestionPipelineError as e:
        file_type = file_name.split('.')[-1] if '.' in file_name else "unknown"
        documents_ingested_total.labels(file_type=file_type, status="failure").inc()
        document_processing_errors_total.labels(error_type="IngestionPipelineError", file_type=file_type).inc()
        logger.error(
            "ingestion_error",
            error_type="IngestionPipelineError",
            error_message=str(e),
            file_name=file_name,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": str(e),
                "details": {"file_name": file_name},
            },
        )
    
    except StorageError as e:
        file_type = file_name.split('.')[-1] if '.' in file_name else "unknown"
        documents_ingested_total.labels(file_type=file_type, status="failure").inc()
        document_processing_errors_total.labels(error_type="StorageError", file_type=file_type).inc()
        logger.error(
            "ingestion_error",
            error_type="StorageError",
            error_message=e.message,
            file_name=file_name,
            details=e.details,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    
    except ExtractionError as e:
        file_type = file_name.split('.')[-1] if '.' in file_name else "unknown"
        documents_ingested_total.labels(file_type=file_type, status="failure").inc()
        document_processing_errors_total.labels(error_type="ExtractionError", file_type=file_type).inc()
        logger.error(
            "ingestion_error",
            error_type="ExtractionError",
            error_message=e.message,
            file_name=file_name,
            details=e.details,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details,
            },
        )
    
    except Exception as e:
        file_type = file_name.split('.')[-1] if '.' in file_name else "unknown" if 'file_name' in locals() else "unknown"
        documents_ingested_total.labels(file_type=file_type, status="failure").inc()
        document_processing_errors_total.labels(error_type=type(e).__name__, file_type=file_type).inc()
        logger.error(
            "ingestion_unexpected_error",
            error_type=type(e).__name__,
            error_message=str(e),
            file_name=file_name if 'file_name' in locals() else (file.filename if file else "unknown"),
            exc_info=True,
        )
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
