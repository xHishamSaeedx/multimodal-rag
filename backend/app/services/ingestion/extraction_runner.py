"""
Parallel extraction runner.

Runs text, table, and image extraction in parallel for better performance.
Supports both synchronous (ProcessPoolExecutor/ThreadPoolExecutor) and asynchronous (asyncio) execution.
Uses ProcessPoolExecutor for CPU-bound extraction tasks to avoid GIL contention.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Tuple
from pathlib import Path
from datetime import datetime

from app.services.ingestion.extractor import TextExtractor, ExtractedContent
from app.services.ingestion.table_extractor import TableExtractor, ExtractedTable
from app.services.ingestion.image_extractor import ImageExtractor, ExtractedImage
from app.services.ingestion.table_processor import TableProcessor, ProcessedTable
from app.utils.exceptions import ExtractionError
from app.utils.metrics import (
    text_extraction_duration_seconds,
    table_extraction_duration_seconds,
    tables_extracted_total,
    image_extraction_duration_seconds,
    images_extracted_total,
)

logger = logging.getLogger(__name__)


class ExtractionRunner:
    """
    Service for running text, table, and image extraction in parallel.
    
    Features:
    - Parallel execution of text, table, and image extraction (synchronous and async)
    - Proper error handling for all extractions
    - Supports both file paths and bytes
    - Table and image extraction failures are non-fatal (continues with text only)
    """
    
    def __init__(
        self,
        max_workers: int = 3,  # Increased to 3 for text, table, and image
        extract_images: bool = True,
        extract_ocr: bool = False,
        enable_text: bool = True,
        enable_tables: bool = True,
        enable_image_extraction: bool = True,
        use_process_pool: bool = True,  # Use ProcessPoolExecutor for CPU-bound tasks
    ):
        """
        Initialize the extraction runner.
        
        Args:
            max_workers: Maximum number of worker threads/processes for parallel extraction
            extract_images: Whether to extract images (default: True)
            extract_ocr: Whether to extract OCR text from images (default: False)
            enable_text: Whether to extract text (default: True)
            enable_tables: Whether to extract tables (default: True)
            enable_image_extraction: Whether to extract images (default: True)
            use_process_pool: Whether to use ProcessPoolExecutor for CPU-bound tasks (default: True)
        """
        self.text_extractor = TextExtractor() if enable_text else None
        self.table_extractor = TableExtractor() if enable_tables else None
        self.image_extractor = ImageExtractor(extract_ocr=extract_ocr) if (extract_images and enable_image_extraction) else None
        self.table_processor = TableProcessor() if enable_tables else None
        self.max_workers = max_workers
        self.extract_images = extract_images and enable_image_extraction
        self.enable_text = enable_text
        self.enable_tables = enable_tables
        self.enable_image_extraction = enable_image_extraction
        self.use_process_pool = use_process_pool
    
    def extract_parallel_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        file_type: str | None = None,
    ) -> Tuple[ExtractedContent, list[ExtractedTable], list[ExtractedImage]]:
        """
        Extract text, tables, and images from file bytes in parallel (synchronous).
        
        Uses ProcessPoolExecutor (for CPU-bound tasks) or ThreadPoolExecutor to run all extractions concurrently.
        ProcessPoolExecutor avoids GIL contention for CPU-intensive extraction operations.
        
        Args:
            file_bytes: File content as bytes
            file_name: Name of the file
            file_type: Optional file type hint
        
        Returns:
            Tuple of (ExtractedContent, List[ExtractedTable], List[ExtractedImage])
        
        Raises:
            ExtractionError: If text extraction fails (table/image extraction failures are non-fatal)
        """
        try:
            logger.info(f"Starting parallel extraction ({'ProcessPool' if self.use_process_pool else 'ThreadPool'}): {file_name}")
            
            # Infer file type for metrics
            if file_type is None:
                file_type = self.text_extractor._infer_file_type(Path(file_name)) if self.text_extractor else "unknown"
            
            # Choose executor based on configuration
            # ProcessPoolExecutor is better for CPU-bound tasks (table/image extraction)
            # but requires picklable objects
            ExecutorClass = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
            
            try:
                with ExecutorClass(max_workers=self.max_workers) as executor:
                    # Submit all extraction tasks with timing wrappers
                    futures = {}
                    future_start_times = {}
                    
                    text_future = None
                    if self.enable_text and self.text_extractor:
                        text_future = executor.submit(
                            self.text_extractor.extract_from_bytes,
                            file_bytes=file_bytes,
                            file_name=file_name,
                            file_type=file_type,
                        )
                        futures[text_future] = "text"
                        future_start_times[text_future] = time.time()
                    
                    table_future = None
                    if self.enable_tables and self.table_extractor:
                        table_future = executor.submit(
                            self.table_extractor.extract_from_bytes,
                            file_bytes=file_bytes,
                            file_name=file_name,
                            file_type=file_type,
                        )
                        futures[table_future] = "table"
                        future_start_times[table_future] = time.time()
                    
                    image_future = None
                    if self.extract_images and self.image_extractor:
                        image_future = executor.submit(
                            self.image_extractor.extract_from_bytes,
                            file_bytes=file_bytes,
                            file_name=file_name,
                            file_type=file_type,
                        )
                        futures[image_future] = "image"
                        future_start_times[image_future] = time.time()
                    
                    # Wait for all to complete and collect results
                    extracted_content = None
                    extracted_tables = []
                    extracted_images = []
                    
                    for future in as_completed(futures.keys()):
                        try:
                            result = future.result()
                            extraction_type = futures[future]
                            extraction_duration = time.time() - future_start_times[future]
                            
                            # Determine which extraction this is based on result type
                            if extraction_type == "text":
                                extracted_content = result
                                # Record text extraction metrics
                                text_extraction_duration_seconds.labels(file_type=file_type).observe(extraction_duration)
                                logger.info(f"✓ Text extraction completed in {extraction_duration:.2f}s")
                            elif extraction_type == "table":
                                extracted_tables = result
                                # Record table extraction metrics (per document, not per table)
                                table_extraction_duration_seconds.labels(file_type=file_type).observe(extraction_duration)
                                tables_extracted_total.labels(file_type=file_type).inc(len(extracted_tables))
                                logger.info(f"✓ Table extraction completed in {extraction_duration:.2f}s ({len(extracted_tables)} tables)")
                            elif extraction_type == "image":
                                extracted_images = result
                                # Record image extraction metrics (per document, not per image)
                                image_extraction_duration_seconds.labels(file_type=file_type).observe(extraction_duration)
                                images_extracted_total.labels(file_type=file_type).inc(len(extracted_images))
                                logger.info(f"✓ Image extraction completed in {extraction_duration:.2f}s ({len(extracted_images)} images)")
                        
                        except Exception as e:
                            # Determine which extraction failed
                            extraction_type = futures[future]
                            if extraction_type == "text":
                                logger.error(
                                    f"Text extraction failed: {e}",
                                    exc_info=e,
                                )
                                raise ExtractionError(
                                    f"Text extraction failed: {str(e)}",
                                    {"file_name": file_name, "error": str(e)},
                                ) from e
                            elif extraction_type == "table":
                                # Table extraction failure is non-fatal
                                logger.warning(
                                    f"Table extraction failed (continuing with text only): {e}",
                                    exc_info=e,
                                )
                                extracted_tables = []
                            elif extraction_type == "image":
                                # Image extraction failure is non-fatal
                                logger.warning(
                                    f"Image extraction failed (continuing without images): {e}",
                                    exc_info=e,
                                )
                                extracted_images = []
            
            except (RuntimeError, OSError) as pool_error:
                # ProcessPoolExecutor might fail with pickling issues or on Windows
                # Fall back to ThreadPoolExecutor
                if self.use_process_pool:
                    logger.warning(
                        f"ProcessPoolExecutor failed ({pool_error}), falling back to ThreadPoolExecutor. "
                        "This may result in slower extraction due to GIL contention."
                    )
                    # Retry with ThreadPoolExecutor
                    self.use_process_pool = False
                    return self.extract_parallel_from_bytes(file_bytes, file_name, file_type)
                else:
                    raise
            
            # Text extraction is required if enabled
            if self.enable_text and extracted_content is None:
                raise ExtractionError(
                    "Text extraction did not complete",
                    {"file_name": file_name},
                )
            
            # If text extraction is disabled, create empty ExtractedContent
            if not self.enable_text:
                from app.services.ingestion.extractor import ExtractedContent
                extracted_content = ExtractedContent(
                    text="",
                    file_name=file_name,
                    file_type=file_type or "unknown",
                    file_size=len(file_bytes),
                    page_count=0,
                    extracted_at=datetime.now(),
                    metadata={},
                )
            
            # Log table extraction results
            if extracted_tables:
                logger.info(
                    f"✓ Extracted {len(extracted_tables)} table(s) from {file_name}"
                )
                for i, table in enumerate(extracted_tables, 1):
                    logger.info(
                        f"  Table {i}: {len(table.rows)} rows × {len(table.headers)} columns "
                        f"(page: {table.page or 'N/A'}, method: {table.method})"
                    )
            else:
                logger.info(f"No tables found in {file_name}")
            
            # Log image extraction results
            if extracted_images:
                logger.info(
                    f"✓ Extracted {len(extracted_images)} image(s) from {file_name}"
                )
                for i, image in enumerate(extracted_images, 1):
                    logger.info(
                        f"  Image {i}: {image.width}×{image.height}px, type: {image.image_type} "
                        f"(page: {image.page or 'N/A'}, format: {image.image_ext})"
                    )
            else:
                logger.info(f"No images found in {file_name}")
            
            logger.info(
                f"Parallel extraction complete: "
                f"{len(extracted_content.text)} chars, "
                f"{len(extracted_tables)} table(s), "
                f"{len(extracted_images)} image(s)"
            )
            
            return extracted_content, extracted_tables, extracted_images
        
        except Exception as e:
            if isinstance(e, ExtractionError):
                raise
            logger.error(f"Unexpected error in parallel extraction: {str(e)}", exc_info=True)
            raise ExtractionError(
                f"Failed to extract content: {str(e)}",
                {"file_name": file_name, "error": str(e)},
            ) from e
    
    def extract_parallel_from_file(
        self,
        file_path: str | Path,
        file_type: str | None = None,
    ) -> Tuple[ExtractedContent, list[ExtractedTable], list[ExtractedImage]]:
        """
        Extract text and tables from file path in parallel (synchronous).
        
        Uses ThreadPoolExecutor to run both extractions concurrently.
        
        Args:
            file_path: Path to the file
            file_type: Optional file type hint
        
        Returns:
            Tuple of (ExtractedContent, List[ExtractedTable])
        
        Raises:
            ExtractionError: If text extraction fails (table extraction failure is non-fatal)
        """
        try:
            file_path = Path(file_path)
            logger.info(f"Starting parallel extraction from file (sync): {file_path}")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit both extraction tasks
                text_future = executor.submit(
                    self.text_extractor.extract_from_file,
                    file_path=file_path,
                    file_type=file_type,
                )
                
                table_future = executor.submit(
                    self.table_extractor.extract_from_file,
                    file_path=file_path,
                    file_type=file_type,
                )
                
                # Wait for both to complete and collect results
                extracted_content = None
                extracted_tables = []
                
                for future in as_completed([text_future, table_future]):
                    try:
                        result = future.result()
                        
                        # Determine which extraction this is based on result type
                        if isinstance(result, ExtractedContent):
                            extracted_content = result
                        elif isinstance(result, list):
                            # Table extraction returns list
                            extracted_tables = result
                    
                    except Exception as e:
                        # Determine which extraction failed
                        if future == text_future:
                            logger.error(
                                f"Text extraction failed: {e}",
                                exc_info=e,
                            )
                            raise ExtractionError(
                                f"Text extraction failed: {str(e)}",
                                {"file_path": str(file_path), "error": str(e)},
                            ) from e
                        else:
                            # Table extraction failure is non-fatal
                            logger.warning(
                                f"Table extraction failed (continuing with text only): {e}",
                                exc_info=e,
                            )
                            extracted_tables = []
            
            if extracted_content is None:
                raise ExtractionError(
                    "Text extraction did not complete",
                    {"file_path": str(file_path)},
                )
            
            # Log table extraction results
            if extracted_tables:
                logger.info(
                    f"✓ Extracted {len(extracted_tables)} table(s) from {file_path.name}"
                )
                for i, table in enumerate(extracted_tables, 1):
                    logger.info(
                        f"  Table {i}: {len(table.rows)} rows × {len(table.headers)} columns "
                        f"(page: {table.page or 'N/A'}, method: {table.method})"
                    )
            else:
                logger.info(f"No tables found in {file_path.name}")
            
            # Note: extract_parallel_from_file doesn't extract images (file-based extraction)
            # Return empty list for images to maintain consistent return signature
            extracted_images = []
            
            logger.info(
                f"Parallel extraction complete: "
                f"{len(extracted_content.text)} chars, "
                f"{len(extracted_tables)} table(s), "
                f"{len(extracted_images)} image(s)"
            )
            
            return extracted_content, extracted_tables, extracted_images
        
        except Exception as e:
            if isinstance(e, ExtractionError):
                raise
            logger.error(f"Unexpected error in parallel extraction: {str(e)}", exc_info=True)
            raise ExtractionError(
                f"Failed to extract content: {str(e)}",
                {"file_path": str(file_path), "error": str(e)},
            ) from e
    
    async def extract_parallel_from_bytes_async(
        self,
        file_bytes: bytes,
        file_name: str,
        file_type: str | None = None,
    ) -> Tuple[ExtractedContent, list[ExtractedTable], list[ExtractedImage]]:
        """
        Extract text and tables from file bytes in parallel (asynchronous).
        
        Uses asyncio.to_thread to run both extractions concurrently.
        
        Args:
            file_bytes: File content as bytes
            file_name: Name of the file
            file_type: Optional file type hint
        
        Returns:
            Tuple of (ExtractedContent, List[ExtractedTable])
        
        Raises:
            ExtractionError: If text extraction fails (table extraction failure is non-fatal)
        """
        try:
            logger.info(f"Starting parallel extraction (async): {file_name}")
            
            # Run both extractions in parallel
            text_task = asyncio.to_thread(
                self.text_extractor.extract_from_bytes,
                file_bytes=file_bytes,
                file_name=file_name,
                file_type=file_type,
            )
            
            table_task = asyncio.to_thread(
                self.table_extractor.extract_from_bytes,
                file_bytes=file_bytes,
                file_name=file_name,
                file_type=file_type,
            )
            
            # Wait for both to complete
            extracted_content, extracted_tables = await asyncio.gather(
                text_task,
                table_task,
                return_exceptions=True,
            )
            
            # Handle exceptions
            if isinstance(extracted_content, Exception):
                logger.error(
                    f"Text extraction failed: {extracted_content}",
                    exc_info=extracted_content,
                )
                raise ExtractionError(
                    f"Text extraction failed: {str(extracted_content)}",
                    {"file_name": file_name, "error": str(extracted_content)},
                ) from extracted_content
            
            if isinstance(extracted_tables, Exception):
                logger.warning(
                    f"Table extraction failed (continuing with text only): {extracted_tables}",
                    exc_info=extracted_tables,
                )
                # Table extraction failure is not fatal - continue with empty list
                extracted_tables = []
            
            # Log table extraction results
            if extracted_tables:
                logger.info(
                    f"✓ Extracted {len(extracted_tables)} table(s) from {file_name}"
                )
                for i, table in enumerate(extracted_tables, 1):
                    logger.info(
                        f"  Table {i}: {len(table.rows)} rows × {len(table.headers)} columns "
                        f"(page: {table.page or 'N/A'}, method: {table.method})"
                    )
            else:
                logger.info(f"No tables found in {file_name}")
            
            # Note: async method doesn't extract images yet
            # Return empty list for images to maintain consistent return signature
            extracted_images = []
            
            logger.info(
                f"Parallel extraction complete: "
                f"{len(extracted_content.text)} chars, "
                f"{len(extracted_tables)} table(s), "
                f"{len(extracted_images)} image(s)"
            )
            
            return extracted_content, extracted_tables, extracted_images
        
        except Exception as e:
            if isinstance(e, ExtractionError):
                raise
            logger.error(f"Unexpected error in parallel extraction: {str(e)}", exc_info=True)
            raise ExtractionError(
                f"Failed to extract content: {str(e)}",
                {"file_name": file_name, "error": str(e)},
            ) from e

