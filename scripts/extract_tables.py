#!/usr/bin/env python3
"""
Standalone script to extract tables from PDF and DOCX files.

Usage:
    python extract_tables.py <file_path>

Requirements:
    - For PDF: camelot-py[cv] (pip install camelot-py[cv])
    - Optional: tabula-py (pip install tabula-py) as fallback
    - For DOCX: python-docx (already in requirements.txt)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def is_table_empty(table_data: List[List[str]]) -> bool:
    """Check if a table is empty (all cells are empty or whitespace)."""
    if not table_data:
        return True
    for row in table_data:
        for cell in row:
            if cell and str(cell).strip():
                return False
    return True


def extract_tables_from_pdf_camelot(pdf_path: str, flavor: str = 'lattice') -> List[Dict[str, Any]]:
    """Extract tables from PDF using camelot with specified flavor."""
    try:
        import camelot
    except ImportError:
        return []
    
    try:
        print(f"  Trying camelot with '{flavor}' method...")
        tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
        
        extracted_tables = []
        for i, table in enumerate(tables):
            # Convert table to list of lists (rows)
            table_data = table.df.values.tolist()
            
            # Filter out empty tables
            if is_table_empty(table_data):
                continue
            
            # Clean up the data - convert to strings and strip
            cleaned_data = []
            for row in table_data:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                # Only add row if it has at least one non-empty cell
                if any(cell for cell in cleaned_row):
                    cleaned_data.append(cleaned_row)
            
            if not cleaned_data:
                continue
            
            extracted_tables.append({
                'table_index': len(extracted_tables) + 1,
                'page': table.page,
                'accuracy': float(table.accuracy),
                'method': f'camelot-{flavor}',
                'data': cleaned_data,
                'headers': cleaned_data[0] if cleaned_data else [],
                'rows': cleaned_data[1:] if len(cleaned_data) > 1 else []
            })
        
        if extracted_tables:
            print(f"  Found {len(extracted_tables)} table(s) with camelot-{flavor}")
        return extracted_tables
    except Exception as e:
        print(f"  Error with camelot-{flavor}: {e}")
        return []


def extract_tables_from_pdf_tabula(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using tabula-py as fallback."""
    try:
        import tabula
    except ImportError:
        return []
    
    try:
        print("  Trying tabula-py as fallback...")
        # Extract tables from all pages
        dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, pandas_options={'header': 0})
        
        extracted_tables = []
        for i, df in enumerate(dfs):
            if df is None or df.empty:
                continue
            
            # Convert to list of lists
            table_data = df.values.tolist()
            
            # Add headers
            headers = df.columns.tolist()
            full_data = [headers] + table_data
            
            # Filter out empty tables
            if is_table_empty(full_data):
                continue
            
            # Clean up the data
            cleaned_data = []
            for row in full_data:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cell for cell in cleaned_row):
                    cleaned_data.append(cleaned_row)
            
            if not cleaned_data:
                continue
            
            extracted_tables.append({
                'table_index': len(extracted_tables) + 1,
                'page': None,  # Tabula doesn't provide page info easily
                'accuracy': None,
                'method': 'tabula',
                'data': cleaned_data,
                'headers': cleaned_data[0] if cleaned_data else [],
                'rows': cleaned_data[1:] if len(cleaned_data) > 1 else []
            })
        
        if extracted_tables:
            print(f"  Found {len(extracted_tables)} table(s) with tabula")
        return extracted_tables
    except Exception as e:
        print(f"  Error with tabula: {e}")
        return []


def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using multiple methods."""
    print(f"Extracting tables from PDF: {pdf_path}")
    
    all_tables = []
    
    # Try camelot with lattice method (for tables with borders)
    tables = extract_tables_from_pdf_camelot(pdf_path, flavor='lattice')
    if tables:
        all_tables.extend(tables)
    
    # Try camelot with stream method (for tables without borders)
    if not all_tables:
        tables = extract_tables_from_pdf_camelot(pdf_path, flavor='stream')
        if tables:
            all_tables.extend(tables)
    
    # Try tabula as fallback
    if not all_tables:
        tables = extract_tables_from_pdf_tabula(pdf_path)
        if tables:
            all_tables.extend(tables)
    
    if not all_tables:
        print("  Warning: No tables found. The PDF might not contain extractable tables,")
        print("           or the tables might be in image format (requiring OCR).")
    
    return all_tables


def extract_tables_from_docx(docx_path: str) -> List[Dict[str, Any]]:
    """Extract tables from DOCX using python-docx."""
    try:
        from docx import Document
    except ImportError:
        print("Error: python-docx is not installed. Install it with: pip install python-docx")
        sys.exit(1)
    
    try:
        doc = Document(docx_path)
        extracted_tables = []
        
        for table_idx, table in enumerate(doc.tables):
            table_data = []
            
            # Extract all rows
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            
            extracted_tables.append({
                'table_index': table_idx + 1,
                'data': table_data,
                'headers': table_data[0] if table_data else [],
                'rows': table_data[1:] if len(table_data) > 1 else []
            })
        
        return extracted_tables
    except Exception as e:
        print(f"Error extracting tables from DOCX: {e}")
        return []


def format_table_markdown(table: Dict[str, Any]) -> str:
    """Format a table as markdown."""
    if not table['data']:
        return "Empty table\n"
    
    markdown_lines = []
    markdown_lines.append(f"\n## Table {table['table_index']}")
    
    metadata = []
    if 'page' in table and table['page'] is not None:
        metadata.append(f"Page: {table['page']}")
    if 'accuracy' in table and table['accuracy'] is not None:
        metadata.append(f"Accuracy: {table['accuracy']:.2f}%")
    if 'method' in table:
        metadata.append(f"Method: {table['method']}")
    
    if metadata:
        markdown_lines.append(" | ".join(metadata))
    
    markdown_lines.append("")
    
    # Create markdown table
    data = table['data']
    if not data:
        return "\n".join(markdown_lines) + "\n(Empty table)\n"
    
    # Header row
    header = "| " + " | ".join(str(cell) for cell in data[0]) + " |"
    markdown_lines.append(header)
    
    # Separator row
    separator = "| " + " | ".join("---" for _ in data[0]) + " |"
    markdown_lines.append(separator)
    
    # Data rows
    for row in data[1:]:
        row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
        markdown_lines.append(row_str)
    
    return "\n".join(markdown_lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description='Extract tables from PDF and DOCX files'
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the PDF or DOCX file'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file path (JSON format). If not specified, prints to stdout'
    )
    parser.add_argument(
        '--format',
        '-f',
        choices=['json', 'markdown'],
        default='json',
        help='Output format: json or markdown (default: json)'
    )
    parser.add_argument(
        '--method',
        '-m',
        choices=['auto', 'camelot-lattice', 'camelot-stream', 'tabula'],
        default='auto',
        help='Extraction method for PDFs: auto (tries all), camelot-lattice, camelot-stream, or tabula (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Determine file type and extract tables
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        if args.method == 'auto':
            tables = extract_tables_from_pdf(str(file_path))
        elif args.method == 'camelot-lattice':
            tables = extract_tables_from_pdf_camelot(str(file_path), flavor='lattice')
        elif args.method == 'camelot-stream':
            tables = extract_tables_from_pdf_camelot(str(file_path), flavor='stream')
        elif args.method == 'tabula':
            tables = extract_tables_from_pdf_tabula(str(file_path))
    elif file_ext in ['.docx', '.doc']:
        print(f"Extracting tables from DOCX: {args.file_path}")
        tables = extract_tables_from_docx(str(file_path))
    else:
        print(f"Error: Unsupported file type: {file_ext}")
        print("Supported formats: .pdf, .docx")
        sys.exit(1)
    
    # Output results
    if args.format == 'json':
        output = json.dumps({
            'file_path': str(file_path),
            'file_type': file_ext,
            'table_count': len(tables),
            'tables': tables
        }, indent=2, ensure_ascii=False)
    else:  # markdown
        output_lines = [
            f"# Tables extracted from: {file_path.name}",
            f"**File type:** {file_ext}",
            f"**Total tables:** {len(tables)}\n"
        ]
        
        for table in tables:
            output_lines.append(format_table_markdown(table))
        
        output = "\n".join(output_lines)
    
    # Write to file or stdout
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "="*80)
        print(output)
    
    print(f"\nExtracted {len(tables)} table(s) from {file_path.name}")


if __name__ == '__main__':
    main()

