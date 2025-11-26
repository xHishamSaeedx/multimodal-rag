"""
Parallel extraction runner.

Runs text and table extraction in parallel for better performance.
Supports both synchronous (ThreadPoolExecutor) and asynchronous (asyncio) execution.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
from pathlib import Path

from app.services.ingestion.extractor import TextExtractor, ExtractedContent
from app.services.ingestion.table_extractor import TableExtractor, ExtractedTable
from app.services.ingestion.table_processor import TableProcessor, ProcessedTable
from app.services.ingestion.table_deduplicator import TableDeduplicator
from app.utils.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class ExtractionRunner:
    """
    Service for running text and table extraction in parallel.
    
    Features:
    - Parallel execution of text and table extraction (synchronous and async)
    - Proper error handling for both extractions
    - Supports both file paths and bytes
    - Table extraction failures are non-fatal (continues with text only)
    """
    
    def __init__(self, max_workers: int = 2, enable_deduplication: bool = True):
        """
        Initialize the extraction runner.
        
        Args:
            max_workers: Maximum number of worker threads for parallel extraction
            enable_deduplication: Whether to remove table text from extracted text
        """
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor()
        self.table_processor = TableProcessor()
        self.table_deduplicator = TableDeduplicator() if enable_deduplication else None
        self.max_workers = max_workers
        self.enable_deduplication = enable_deduplication
    
    def extract_parallel_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        file_type: str | None = None,
    ) -> Tuple[ExtractedContent, list[ExtractedTable]]:
        """
        Extract text and tables from file bytes in parallel (synchronous).
        
        Uses ThreadPoolExecutor to run both extractions concurrently.
        
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
            logger.info(f"Starting parallel extraction (sync): {file_name}")
            
            # Create a copy of file_bytes for each extraction (some libraries may consume it)
            # In practice, most libraries work with BytesIO which doesn't consume the bytes
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit both extraction tasks
                text_future = executor.submit(
                    self.text_extractor.extract_from_bytes,
                    file_bytes=file_bytes,
                    file_name=file_name,
                    file_type=file_type,
                )
                
                table_future = executor.submit(
                    self.table_extractor.extract_from_bytes,
                    file_bytes=file_bytes,
                    file_name=file_name,
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
                                {"file_name": file_name, "error": str(e)},
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
                    {"file_name": file_name},
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
            
            # Process tables and remove table text from extracted text
            original_text_length = len(extracted_content.text)
            processed_tables = []
            
            if extracted_tables and self.enable_deduplication and self.table_deduplicator:
                # Process tables
                processed_tables = [
                    self.table_processor.process_table(table) for table in extracted_tables
                ]
                
                # Remove table text from extracted text
                cleaned_text = self.table_deduplicator.remove_table_text(
                    extracted_content.text,
                    processed_tables,
                )
                
                removed_chars = original_text_length - len(cleaned_text)
                
                if removed_chars > 0:
                    logger.info(
                        f"✓ Removed {removed_chars} characters of table content from text "
                        f"({removed_chars / original_text_length * 100:.1f}% of original text)"
                    )
                    logger.info(
                        f"  Text length: {original_text_length} → {len(cleaned_text)} characters"
                    )
                    
                    # Update extracted content with cleaned text
                    extracted_content.text = cleaned_text
                else:
                    logger.info(
                        f"No table content found in extracted text (tables may be in image format)"
                    )
            elif extracted_tables:
                # Tables extracted but deduplication disabled
                processed_tables = [
                    self.table_processor.process_table(table) for table in extracted_tables
                ]
                logger.info(
                    f"Table deduplication disabled - table content may appear in text chunks"
                )
            
            logger.info(
                f"Parallel extraction complete: "
                f"{len(extracted_content.text)} chars, "
                f"{len(extracted_tables)} table(s)"
            )
            
            return extracted_content, extracted_tables
        
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
    ) -> Tuple[ExtractedContent, list[ExtractedTable]]:
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
            
            # Process tables and remove table text from extracted text
            original_text_length = len(extracted_content.text)
            processed_tables = []
            
            if extracted_tables and self.enable_deduplication and self.table_deduplicator:
                # Process tables
                processed_tables = [
                    self.table_processor.process_table(table) for table in extracted_tables
                ]
                
                # Remove table text from extracted text
                cleaned_text = self.table_deduplicator.remove_table_text(
                    extracted_content.text,
                    processed_tables,
                )
                
                removed_chars = original_text_length - len(cleaned_text)
                
                if removed_chars > 0:
                    logger.info(
                        f"✓ Removed {removed_chars} characters of table content from text "
                        f"({removed_chars / original_text_length * 100:.1f}% of original text)"
                    )
                    logger.info(
                        f"  Text length: {original_text_length} → {len(cleaned_text)} characters"
                    )
                    
                    # Update extracted content with cleaned text
                    extracted_content.text = cleaned_text
                else:
                    logger.info(
                        f"No table content found in extracted text (tables may be in image format)"
                    )
            elif extracted_tables:
                # Tables extracted but deduplication disabled
                processed_tables = [
                    self.table_processor.process_table(table) for table in extracted_tables
                ]
                logger.info(
                    f"Table deduplication disabled - table content may appear in text chunks"
                )
            
            logger.info(
                f"Parallel extraction complete: "
                f"{len(extracted_content.text)} chars, "
                f"{len(extracted_tables)} table(s)"
            )
            
            return extracted_content, extracted_tables
        
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
    ) -> Tuple[ExtractedContent, list[ExtractedTable]]:
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
            
            # Process tables and remove table text from extracted text
            original_text_length = len(extracted_content.text)
            processed_tables = []
            
            if extracted_tables and self.enable_deduplication and self.table_deduplicator:
                # Process tables
                processed_tables = [
                    self.table_processor.process_table(table) for table in extracted_tables
                ]
                
                # Remove table text from extracted text
                cleaned_text = self.table_deduplicator.remove_table_text(
                    extracted_content.text,
                    processed_tables,
                )
                
                removed_chars = original_text_length - len(cleaned_text)
                
                if removed_chars > 0:
                    logger.info(
                        f"✓ Removed {removed_chars} characters of table content from text "
                        f"({removed_chars / original_text_length * 100:.1f}% of original text)"
                    )
                    logger.info(
                        f"  Text length: {original_text_length} → {len(cleaned_text)} characters"
                    )
                    
                    # Update extracted content with cleaned text
                    extracted_content.text = cleaned_text
                else:
                    logger.info(
                        f"No table content found in extracted text (tables may be in image format)"
                    )
            elif extracted_tables:
                # Tables extracted but deduplication disabled
                processed_tables = [
                    self.table_processor.process_table(table) for table in extracted_tables
                ]
                logger.info(
                    f"Table deduplication disabled - table content may appear in text chunks"
                )
            
            logger.info(
                f"Parallel extraction complete: "
                f"{len(extracted_content.text)} chars, "
                f"{len(extracted_tables)} table(s)"
            )
            
            return extracted_content, extracted_tables
        
        except Exception as e:
            if isinstance(e, ExtractionError):
                raise
            logger.error(f"Unexpected error in parallel extraction: {str(e)}", exc_info=True)
            raise ExtractionError(
                f"Failed to extract content: {str(e)}",
                {"file_name": file_name, "error": str(e)},
            ) from e

