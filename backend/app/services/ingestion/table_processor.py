"""
Table processing service.

Converts extracted tables into multiple formats:
- JSON (structured data)
- Markdown (human-readable)
- Flattened text (for embeddings)
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field

from app.services.ingestion.table_extractor import ExtractedTable

logger = logging.getLogger(__name__)


@dataclass
class ProcessedTable:
    """
    Represents a processed table with multiple format representations.
    
    Attributes:
        table_index: Index of the table within the document
        page: Page number where table was found (None for DOCX)
        table_data: Structured JSON format (dict with headers and rows)
        table_markdown: Markdown representation
        table_text: Flattened text representation for embeddings
        metadata: Additional metadata (row_count, col_count, headers, etc.)
        original_table: Reference to the original ExtractedTable
    """
    
    table_index: int
    page: int | None
    table_data: Dict[str, Any]
    table_markdown: str
    table_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_table: ExtractedTable | None = None


class TableProcessor:
    """
    Service for processing extracted tables into multiple formats.
    
    Converts tables to:
    - JSON format: Structured data for programmatic access
    - Markdown format: Human-readable format for LLM context
    - Flattened text: For embedding generation
    """
    
    def process_table(self, table: ExtractedTable) -> ProcessedTable:
        """
        Process an extracted table into all required formats.
        
        Args:
            table: ExtractedTable object to process
        
        Returns:
            ProcessedTable object with all format representations
        """
        # Generate all formats
        table_data = self.to_json(table)
        table_markdown = self.to_markdown(table)
        table_text = self.to_flattened_text(table)
        
        # Generate metadata
        metadata = {
            "row_count": len(table.rows),
            "col_count": len(table.headers) if table.headers else 0,
            "headers": table.headers,
            "method": table.method,
            "accuracy": table.accuracy,
            "page": table.page,
            **(table.metadata or {}),
        }
        
        return ProcessedTable(
            table_index=table.table_index,
            page=table.page,
            table_data=table_data,
            table_markdown=table_markdown,
            table_text=table_text,
            metadata=metadata,
            original_table=table,
        )
    
    def to_json(self, table: ExtractedTable) -> Dict[str, Any]:
        """
        Convert table to JSON format (structured data).
        
        Args:
            table: ExtractedTable to convert
        
        Returns:
            Dictionary with 'headers' and 'rows' keys
        """
        return {
            "headers": table.headers,
            "rows": table.rows,
        }
    
    def to_markdown(self, table: ExtractedTable) -> str:
        """
        Convert table to markdown format.
        
        Args:
            table: ExtractedTable to convert
        
        Returns:
            Markdown table string
        """
        if not table.data:
            return ""
        
        markdown_lines = []
        
        # Header row
        if table.headers:
            header = "| " + " | ".join(str(cell) for cell in table.headers) + " |"
            markdown_lines.append(header)
            
            # Separator row
            separator = "| " + " | ".join("---" for _ in table.headers) + " |"
            markdown_lines.append(separator)
        
        # Data rows
        for row in table.rows:
            row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
            markdown_lines.append(row_str)
        
        return "\n".join(markdown_lines)
    
    def to_flattened_text(self, table: ExtractedTable) -> str:
        """
        Convert table to flattened text format for embeddings.
        
        Format: "Header1: Value1, Header2: Value2, Header3: Value3"
        Each row becomes a line.
        
        Args:
            table: ExtractedTable to convert
        
        Returns:
            Flattened text string
        """
        if not table.data:
            return ""
        
        text_lines = []
        
        # If we have headers, use them for key-value pairs
        if table.headers:
            for row in table.rows:
                # Match headers with row values
                row_parts = []
                for i, header in enumerate(table.headers):
                    value = row[i] if i < len(row) else ""
                    if header.strip() and value.strip():
                        row_parts.append(f"{header}: {value}")
                
                if row_parts:
                    text_lines.append(", ".join(row_parts))
        else:
            # No headers, just join row values
            for row in table.rows:
                row_text = ", ".join(str(cell).strip() for cell in row if str(cell).strip())
                if row_text:
                    text_lines.append(row_text)
        
        return "\n".join(text_lines)

