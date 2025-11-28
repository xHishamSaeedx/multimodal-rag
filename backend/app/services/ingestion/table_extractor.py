"""
Table extraction service.

Extracts tables from PDF and DOCX files.
Supports multiple extraction methods with fallback strategies.
"""

import io
import logging
from pathlib import Path
from typing import BinaryIO, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# PDF table extraction libraries
try:
    import camelot
except ImportError:
    camelot = None

try:
    import tabula
except ImportError:
    tabula = None

# DOCX table extraction
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

from app.utils.exceptions import (
    ExtractionError,
    UnsupportedFileTypeError,
    FileReadError,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    """
    Represents an extracted table with metadata.
    
    Attributes:
        table_index: Index of the table within the document (1-based)
        page: Page number where table was found (None for DOCX)
        data: Raw table data as list of rows (each row is list of cells)
        headers: First row as headers (if available)
        rows: Data rows (excluding header row)
        method: Extraction method used ('camelot-lattice', 'camelot-stream', 'tabula', 'docx')
        accuracy: Extraction accuracy score (0-100, None if not available)
        metadata: Additional metadata (position, dimensions, etc.)
    """
    
    table_index: int
    page: Optional[int]
    data: List[List[str]]
    headers: List[str]
    rows: List[List[str]]
    method: str
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set default values."""
        if not self.data:
            self.data = []
        if not self.headers and self.data:
            # If no headers specified, use first row as headers
            self.headers = self.data[0] if self.data else []
            self.rows = self.data[1:] if len(self.data) > 1 else []
        elif not self.rows and self.data:
            # If headers specified but no rows, extract rows
            self.rows = self.data[1:] if len(self.data) > 1 else []


class TableExtractor:
    """
    Service for extracting tables from various document formats.
    
    Supports:
    - PDF: Using camelot-py (lattice/stream) or tabula-py (fallback)
    - DOCX: Using python-docx (native support)
    
    Features:
    - Multiple extraction methods with automatic fallback
    - Empty table filtering
    - Data cleaning and normalization
    """
    
    # Supported file types for table extraction
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
    }
    
    def __init__(
        self,
        pdf_method: str = "auto",  # auto, camelot-lattice, camelot-stream, tabula
    ):
        """
        Initialize the table extractor.
        
        Args:
            pdf_method: Preferred PDF extraction method
                       'auto' tries all methods in order
        """
        self.pdf_method = pdf_method
        
        # Check for required libraries
        if camelot is None and tabula is None:
            logger.warning(
                "No PDF table extraction libraries available. "
                "Install camelot-py[cv] or tabula-py for PDF support."
            )
        
        if DocxDocument is None:
            logger.warning(
                "python-docx is not installed. DOCX table extraction will not work."
            )
    
    def extract_from_file(
        self,
        file_path: str | Path,
        file_type: str | None = None,
    ) -> List[ExtractedTable]:
        """
        Extract tables from a file path.
        
        Args:
            file_path: Path to the file to extract tables from
            file_type: Optional file type hint (pdf, docx)
                      If not provided, will be inferred from file extension
        
        Returns:
            List of ExtractedTable objects
        
        Raises:
            UnsupportedFileTypeError: If file type is not supported
            FileReadError: If file cannot be read
            ExtractionError: If extraction fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileReadError(
                f"File not found: {file_path}",
                {"file_path": str(file_path)},
            )
        
        # Determine file type
        if file_type is None:
            file_type = self._infer_file_type(file_path)
        
        if file_type not in self.SUPPORTED_EXTENSIONS.values():
            raise UnsupportedFileTypeError(
                f"Unsupported file type for table extraction: {file_type}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS.values())}",
                {"file_path": str(file_path), "file_type": file_type},
            )
        
        # Extract based on file type
        try:
            if file_type == "pdf":
                tables = self._extract_pdf(str(file_path))
            elif file_type == "docx":
                tables = self._extract_docx(str(file_path))
            else:
                raise UnsupportedFileTypeError(
                    f"Unsupported file type: {file_type}",
                    {"file_path": str(file_path), "file_type": file_type},
                )
            
            logger.info(f"Extracted {len(tables)} table(s) from {file_path.name}")
            return tables
        
        except Exception as e:
            if isinstance(e, (UnsupportedFileTypeError, FileReadError, ExtractionError)):
                raise
            raise ExtractionError(
                f"Failed to extract tables from {file_path}: {str(e)}",
                {"file_path": str(file_path), "file_type": file_type, "error": str(e)},
            ) from e
    
    def extract_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        file_type: str | None = None,
    ) -> List[ExtractedTable]:
        """
        Extract tables from file bytes.
        
        Args:
            file_bytes: File content as bytes
            file_name: Name of the file (used to infer type)
            file_type: Optional file type hint (pdf, docx)
                      If not provided, will be inferred from file name
        
        Returns:
            List of ExtractedTable objects
        
        Raises:
            UnsupportedFileTypeError: If file type is not supported
            ExtractionError: If extraction fails
        """
        # Determine file type
        if file_type is None:
            file_path = Path(file_name)
            file_type = self._infer_file_type(file_path)
        
        if file_type not in self.SUPPORTED_EXTENSIONS.values():
            raise UnsupportedFileTypeError(
                f"Unsupported file type for table extraction: {file_type}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS.values())}",
                {"file_name": file_name, "file_type": file_type},
            )
        
        # Extract based on file type
        try:
            file_io = io.BytesIO(file_bytes)
            
            if file_type == "pdf":
                # PDF libraries typically need file paths, so we'll write to temp file
                # For now, we'll use a workaround with BytesIO
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    tables = self._extract_pdf(tmp_path)
                finally:
                    # Clean up temp file
                    import os
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            elif file_type == "docx":
                tables = self._extract_docx(file_io)
            else:
                raise UnsupportedFileTypeError(
                    f"Unsupported file type: {file_type}",
                    {"file_name": file_name, "file_type": file_type},
                )
            
            logger.info(f"Extracted {len(tables)} table(s) from {file_name}")
            return tables
        
        except Exception as e:
            if isinstance(e, (UnsupportedFileTypeError, ExtractionError)):
                raise
            raise ExtractionError(
                f"Failed to extract tables from {file_name}: {str(e)}",
                {"file_name": file_name, "file_type": file_type, "error": str(e)},
            ) from e
    
    def _infer_file_type(self, file_path: Path) -> str:
        """
        Infer file type from file extension.
        
        Args:
            file_path: Path to the file
        
        Returns:
            File type string (pdf, docx)
        """
        ext = file_path.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext)
        
        if file_type is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension for table extraction: {ext}. "
                f"Supported extensions: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}",
                {"file_path": str(file_path), "extension": ext},
            )
        
        return file_type
    
    def _extract_pdf(self, pdf_path: str) -> List[ExtractedTable]:
        """
        Extract tables from a PDF file using multiple methods.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of ExtractedTable objects
        """
        all_tables = []
        
        # Determine which methods to try
        methods_to_try = []
        if self.pdf_method == "auto":
            # Try in order: camelot-lattice, camelot-stream, tabula
            methods_to_try = ["camelot-lattice", "camelot-stream", "tabula"]
        else:
            methods_to_try = [self.pdf_method]
        
        # Try each method
        for method in methods_to_try:
            if method.startswith("camelot"):
                flavor = method.split("-")[1]  # lattice or stream
                tables = self._extract_pdf_camelot(pdf_path, flavor=flavor)
                if tables:
                    all_tables.extend(tables)
                    logger.info(f"Successfully extracted {len(tables)} table(s) using {method}")
                    break  # Success, no need to try other methods
            elif method == "tabula":
                tables = self._extract_pdf_tabula(pdf_path)
                if tables:
                    all_tables.extend(tables)
                    logger.info(f"Successfully extracted {len(tables)} table(s) using {method}")
                    break  # Success, no need to try other methods
        
        if not all_tables:
            logger.warning(
                f"No tables found in PDF: {pdf_path}. "
                "The PDF might not contain extractable tables, "
                "or the tables might be in image format (requiring OCR)."
            )
        
        return all_tables
    
    def _extract_pdf_camelot(
        self, pdf_path: str, flavor: str = "lattice"
    ) -> List[ExtractedTable]:
        """
        Extract tables from PDF using camelot with specified flavor.
        
        Args:
            pdf_path: Path to PDF file
            flavor: Extraction flavor ('lattice' for bordered tables, 'stream' for borderless)
        
        Returns:
            List of ExtractedTable objects
        """
        if camelot is None:
            return []
        
        try:
            logger.debug(f"Trying camelot with '{flavor}' method for {pdf_path}")
            tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
            
            extracted_tables = []
            for i, table in enumerate(tables):
                # Convert table to list of lists (rows)
                table_data = table.df.values.tolist()
                
                # Filter out empty tables
                if TableExtractor._is_table_empty(table_data):
                    logger.debug(f"Skipping table {i+1} on page {table.page}: empty table")
                    continue
                
                # Clean up the data - convert to strings and strip
                cleaned_data = []
                for row in table_data:
                    cleaned_row = [
                        str(cell).strip() if cell is not None else "" for cell in row
                    ]
                    # Only add row if it has at least one non-empty cell
                    if any(cell for cell in cleaned_row):
                        cleaned_data.append(cleaned_row)
                
                if not cleaned_data:
                    logger.debug(f"Skipping table {i+1} on page {table.page}: no data after cleaning")
                    continue
                
                # Validate that this is actually a table (not a false positive like chart labels)
                if not self._is_valid_table(cleaned_data):
                    logger.debug(
                        f"Skipping table {i+1} on page {table.page}: "
                        f"does not meet minimum table requirements (likely false positive - chart labels, list, etc.)"
                    )
                    continue
                
                # Extract headers and rows
                headers = cleaned_data[0] if cleaned_data else []
                rows = cleaned_data[1:] if len(cleaned_data) > 1 else []
                
                extracted_table = ExtractedTable(
                    table_index=len(extracted_tables) + 1,
                    page=table.page,
                    data=cleaned_data,
                    headers=headers,
                    rows=rows,
                    method=f"camelot-{flavor}",
                    accuracy=float(table.accuracy) if hasattr(table, "accuracy") else None,
                    metadata={
                        "page": table.page,
                        "accuracy": float(table.accuracy) if hasattr(table, "accuracy") else None,
                    },
                )
                extracted_tables.append(extracted_table)
            
            return extracted_tables
        
        except Exception as e:
            logger.warning(f"Error with camelot-{flavor}: {e}")
            return []
    
    def _extract_pdf_tabula(self, pdf_path: str) -> List[ExtractedTable]:
        """
        Extract tables from PDF using tabula-py as fallback.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of ExtractedTable objects
        """
        if tabula is None:
            return []
        
        try:
            logger.debug(f"Trying tabula-py for {pdf_path}")
            # Extract tables from all pages
            dfs = tabula.read_pdf(
                pdf_path, pages="all", multiple_tables=True, pandas_options={"header": 0}
            )
            
            extracted_tables = []
            for i, df in enumerate(dfs):
                if df is None or df.empty:
                    continue
                
                # Convert to list of lists
                table_data = df.values.tolist()
                
                # Add headers
                headers = [str(col).strip() for col in df.columns.tolist()]
                full_data = [headers] + table_data
                
                # Filter out empty tables
                if TableExtractor._is_table_empty(full_data):
                    continue
                
                # Validate that this is actually a table (not a false positive)
                if not self._is_valid_table(full_data):
                    logger.debug(f"Skipping tabula table {i+1}: does not meet minimum table requirements")
                    continue
                
                # Clean up the data
                cleaned_data = [headers]
                for row in table_data:
                    cleaned_row = [
                        str(cell).strip() if cell is not None else "" for cell in row
                    ]
                    if any(cell for cell in cleaned_row):
                        cleaned_data.append(cleaned_row)
                
                if len(cleaned_data) <= 1:  # Only headers, no data
                    continue
                
                rows = cleaned_data[1:]
                
                extracted_table = ExtractedTable(
                    table_index=len(extracted_tables) + 1,
                    page=None,  # Tabula doesn't provide page info easily
                    data=cleaned_data,
                    headers=headers,
                    rows=rows,
                    method="tabula",
                    accuracy=None,
                    metadata={},
                )
                extracted_tables.append(extracted_table)
            
            return extracted_tables
        
        except Exception as e:
            logger.warning(f"Error with tabula: {e}")
            return []
    
    def _extract_docx(self, source: str | BinaryIO) -> List[ExtractedTable]:
        """
        Extract tables from DOCX file using python-docx.
        
        Args:
            source: Path to DOCX file or BytesIO object
        
        Returns:
            List of ExtractedTable objects
        """
        if DocxDocument is None:
            raise ImportError(
                "python-docx is not installed. Install it with: pip install python-docx"
            )
        
        try:
            doc = DocxDocument(source)
            extracted_tables = []
            
            for table_idx, table in enumerate(doc.tables):
                table_data = []
                
                # Extract all rows
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                
                # Filter out empty tables
                if TableExtractor._is_table_empty(table_data):
                    continue
                
                # Validate that this is actually a table (not a false positive)
                if not self._is_valid_table(table_data):
                    logger.debug(f"Skipping DOCX table {table_idx+1}: does not meet minimum table requirements")
                    continue
                
                # Extract headers and rows
                headers = table_data[0] if table_data else []
                rows = table_data[1:] if len(table_data) > 1 else []
                
                extracted_table = ExtractedTable(
                    table_index=table_idx + 1,
                    page=None,  # DOCX doesn't have explicit pages
                    data=table_data,
                    headers=headers,
                    rows=rows,
                    method="docx",
                    accuracy=None,
                    metadata={},
                )
                extracted_tables.append(extracted_table)
            
            return extracted_tables
        
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract tables from DOCX: {str(e)}",
                {"source": str(source), "error": str(e)},
            ) from e
    
    def _is_valid_table(self, table_data: List[List[str]], min_rows: int = 3, min_cols: int = 2) -> bool:
        """
        Validate if extracted data is actually a real table (not a false positive).
        
        Filters out:
        - Single-column data (lists, not tables)
        - Too small tables (likely chart labels or structured text)
        - Empty tables
        
        Args:
            table_data: Table data as list of rows (each row is list of cells)
            min_rows: Minimum number of rows to be considered a valid table (default: 3)
            min_cols: Minimum number of columns to be considered a valid table (default: 2)
        
        Returns:
            True if table is valid, False otherwise
        """
        if not table_data:
            return False
        
        # Check if empty
        if TableExtractor._is_table_empty(table_data):
            return False
        
        # Check minimum dimensions
        num_rows = len(table_data)
        if num_rows < min_rows:
            logger.debug(f"Table rejected: too few rows ({num_rows} < {min_rows})")
            return False
        
        # Check column count (use first row as reference)
        if not table_data[0]:
            return False
        
        num_cols = len(table_data[0])
        if num_cols < min_cols:
            logger.debug(f"Table rejected: too few columns ({num_cols} < {min_cols}) - likely a list, not a table")
            return False
        
        # Check if all rows have same number of columns (basic table structure)
        for i, row in enumerate(table_data):
            if len(row) != num_cols:
                # Allow some variation (merged cells, etc.) but log it
                if abs(len(row) - num_cols) > 1:
                    logger.debug(f"Table rejected: inconsistent column count (row {i} has {len(row)} cols, expected {num_cols})")
                    return False
        
        return True
    
    def _is_table_empty(table_data: List[List[str]]) -> bool:
        """
        Check if a table is empty (all cells are empty or whitespace).
        
        Args:
            table_data: Table data as list of rows
        
        Returns:
            True if table is empty, False otherwise
        """
        if not table_data:
            return True
        
        for row in table_data:
            for cell in row:
                if cell and str(cell).strip():
                    return False
        
        return True
    
    def is_supported(self, file_path: str | Path) -> bool:
        """
        Check if a file type is supported for table extraction.
        
        Args:
            file_path: Path to the file
        
        Returns:
            True if file type is supported, False otherwise
        """
        try:
            file_path = Path(file_path)
            ext = file_path.suffix.lower()
            return ext in self.SUPPORTED_EXTENSIONS
        except Exception:
            return False

