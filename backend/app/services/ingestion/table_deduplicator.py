"""
Table deduplication service.

Removes table text from extracted document text to prevent duplicate content
in text chunks and table chunks.
"""

import logging
import re
from typing import List, Tuple, Optional
from difflib import SequenceMatcher

from app.services.ingestion.table_processor import ProcessedTable

logger = logging.getLogger(__name__)


class TableRegion:
    """
    Represents a region in text that contains table content.
    
    Attributes:
        start_pos: Start character position in text
        end_pos: End character position in text
        table: Reference to the ProcessedTable
        confidence: Confidence score (0.0-1.0) that this region is table content
    """
    
    def __init__(
        self,
        start_pos: int,
        end_pos: int,
        table: ProcessedTable,
        confidence: float = 1.0,
    ):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.table = table
        self.confidence = confidence


class TableDeduplicator:
    """
    Service for identifying and removing table text from extracted document text.
    
    Strategy:
    1. Extract key phrases from each table (headers + sample rows)
    2. Search for these phrases in the document text
    3. Identify table regions
    4. Remove or mark table regions in text
    
    This prevents duplicate content where:
    - Text extraction picks up table content (especially in PDFs)
    - Table extraction provides more accurate structured representation
    """
    
    def __init__(
        self,
        min_match_ratio: float = 0.5,  # Lowered from 0.7 for better detection
        max_search_length: int = 5000,
    ):
        """
        Initialize the table deduplicator.
        
        Args:
            min_match_ratio: Minimum similarity ratio (0.0-1.0) to consider a match
            max_search_length: Maximum length of text to search for each table
        """
        self.min_match_ratio = min_match_ratio
        self.max_search_length = max_search_length
    
    def identify_table_regions(
        self,
        text: str,
        tables: List[ProcessedTable],
    ) -> List[TableRegion]:
        """
        Identify regions in text that contain table content.
        
        Args:
            text: Full document text
            tables: List of processed tables to search for
        
        Returns:
            List of TableRegion objects representing table content locations
        """
        regions = []
        detected_tables = []
        undetected_tables = []
        
        logger.info(
            f"Starting table region identification: {len(tables)} table(s) to check "
            f"in {len(text)} characters of text"
        )
        
        for table in tables:
            # Extract key phrases from table
            key_phrases = self._extract_key_phrases(table)
            
            if not key_phrases:
                logger.warning(
                    f"Table {table.table_index} (page {table.page}): "
                    f"No key phrases extracted - cannot match in text"
                )
                undetected_tables.append({
                    "table_index": table.table_index,
                    "page": table.page,
                    "reason": "no_key_phrases",
                    "row_count": table.metadata.get("row_count", 0),
                    "col_count": table.metadata.get("col_count", 0),
                })
                continue
            
            logger.debug(
                f"Table {table.table_index} (page {table.page}): "
                f"Extracted {len(key_phrases)} key phrase(s) for matching"
            )
            
            # Search for table content in text
            region = self._find_table_region(text, table, key_phrases)
            
            if region:
                regions.append(region)
                detected_tables.append({
                    "table_index": table.table_index,
                    "page": table.page,
                    "confidence": region.confidence,
                    "start_pos": region.start_pos,
                    "end_pos": region.end_pos,
                    "region_size": region.end_pos - region.start_pos,
                })
                logger.info(
                    f"✓ Table {table.table_index} (page {table.page}) DETECTED in text: "
                    f"positions {region.start_pos}-{region.end_pos} "
                    f"(confidence: {region.confidence:.2f}, size: {region.end_pos - region.start_pos} chars)"
                )
            else:
                undetected_tables.append({
                    "table_index": table.table_index,
                    "page": table.page,
                    "reason": "no_match_found",
                    "row_count": table.metadata.get("row_count", 0),
                    "col_count": table.metadata.get("col_count", 0),
                    "key_phrases_count": len(key_phrases),
                })
                logger.info(
                    f"✗ Table {table.table_index} (page {table.page}) NOT DETECTED in text: "
                    f"No matching region found (tried {len(key_phrases)} key phrases, "
                    f"min_match_ratio: {self.min_match_ratio})"
                )
        
        # Sort regions by start position
        regions.sort(key=lambda r: r.start_pos)
        
        # Merge overlapping regions
        merged_regions = self._merge_overlapping_regions(regions)
        
        # Summary logging
        logger.info(
            f"Table region identification complete: "
            f"{len(merged_regions)} region(s) identified from {len(tables)} table(s)"
        )
        logger.info(
            f"Detection summary: "
            f"{len(detected_tables)} table(s) detected, "
            f"{len(undetected_tables)} table(s) not detected"
        )
        
        if detected_tables:
            detected_str = ', '.join(
                f"Table {t['table_index']} (page {t['page']}, conf: {t['confidence']:.2f})"
                for t in detected_tables
            )
            logger.info(f"Detected tables: {detected_str}")
        
        if undetected_tables:
            reasons = {}
            for t in undetected_tables:
                reason = t['reason']
                reasons[reason] = reasons.get(reason, 0) + 1
            
            undetected_str = ', '.join(
                f"Table {t['table_index']} (page {t['page']}, reason: {t['reason']})"
                for t in undetected_tables
            )
            logger.info(f"Undetected tables: {undetected_str}")
            logger.info(
                f"Undetection reasons: {dict(reasons)}"
            )
            logger.info(
                f"Note: Undetected tables may be image-based (not in text) or matching failed. "
                f"Check if these tables appear as text in the document."
            )
        
        return merged_regions
    
    def remove_table_text(
        self,
        text: str,
        tables: List[ProcessedTable],
    ) -> str:
        """
        Remove table text from document text.
        
        Args:
            text: Full document text
            tables: List of processed tables to remove
        
        Returns:
            Text with table content removed
        """
        if not tables:
            return text
        
        # Identify table regions
        regions = self.identify_table_regions(text, tables)
        
        if not regions:
            logger.debug("No table regions found to remove")
            return text
        
        # Remove regions from text (in reverse order to maintain positions)
        cleaned_text = text
        for region in reversed(regions):
            # Remove the region, replace with placeholder or just remove
            before = cleaned_text[: region.start_pos]
            after = cleaned_text[region.end_pos :]
            
            # Add a small marker to indicate table was removed (optional)
            # This helps maintain text flow
            marker = " [Table content removed] " if region.confidence > 0.8 else " "
            
            cleaned_text = before + marker + after
        
        removed_chars = len(text) - len(cleaned_text)
        logger.info(
            f"Removed {removed_chars} characters of table content "
            f"from {len(regions)} region(s)"
        )
        
        return cleaned_text
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching by removing extra whitespace,
        normalizing punctuation, and converting to lowercase.
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace (multiple spaces/tabs/newlines -> single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common punctuation variations
        text = text.replace('|', ' ')  # Remove pipe separators
        text = text.replace(':', ' ')  # Remove colons
        text = text.replace(',', ' ')  # Remove commas
        text = text.replace(';', ' ')  # Remove semicolons
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_key_phrases(self, table: ProcessedTable) -> List[str]:
        """
        Extract key phrases from a table for matching.
        
        Uses headers and first few rows to create searchable phrases.
        Multiple variations are created to handle formatting differences.
        
        Args:
            table: ProcessedTable to extract phrases from
        
        Returns:
            List of key phrase strings (both original and normalized)
        """
        phrases = []
        
        # Add headers as phrases
        if table.metadata.get("headers"):
            headers = table.metadata["headers"]
            # Add individual headers (both original and normalized)
            for header in headers:
                header_clean = header.strip()
                if header_clean:
                    phrases.append(header_clean)  # Original
                    phrases.append(self._normalize_text(header_clean))  # Normalized
            logger.debug(
                f"Table {table.table_index}: Extracted {len(headers)} header phrase(s) "
                f"(with normalized variants: {len(headers) * 2})"
            )
            
            # Add header combinations (multiple formats)
            if len(headers) > 1:
                # Format 1: Pipe-separated
                header_phrase_pipe = " | ".join(h.strip() for h in headers if h.strip())
                phrases.append(header_phrase_pipe)
                phrases.append(self._normalize_text(header_phrase_pipe))
                
                # Format 2: Space-separated
                header_phrase_space = " ".join(h.strip() for h in headers if h.strip())
                phrases.append(header_phrase_space)
                phrases.append(self._normalize_text(header_phrase_space))
                
                logger.debug(
                    f"Table {table.table_index}: Added header combination phrases "
                    f"(pipe and space formats, normalized)"
                )
        
        # Add first few rows as phrases (sample of table content)
        table_data = table.table_data
        if table_data and "rows" in table_data:
            rows = table_data["rows"][:5]  # Increased to 5 rows for better matching
            row_phrases = []
            for i, row in enumerate(rows):
                if row:
                    # Format 1: Pipe-separated (as extracted)
                    row_phrase_pipe = " | ".join(str(cell).strip() for cell in row if str(cell).strip())
                    if row_phrase_pipe:
                        phrases.append(row_phrase_pipe)
                        phrases.append(self._normalize_text(row_phrase_pipe))
                    
                    # Format 2: Space-separated (as might appear in text)
                    row_phrase_space = " ".join(str(cell).strip() for cell in row if str(cell).strip())
                    if row_phrase_space and row_phrase_space != row_phrase_pipe:
                        phrases.append(row_phrase_space)
                        phrases.append(self._normalize_text(row_phrase_space))
                    
                    # Format 3: Key-value pairs (as in flattened text)
                    if table.metadata.get("headers") and len(table.metadata["headers"]) == len(row):
                        kv_parts = []
                        for j, header in enumerate(table.metadata["headers"]):
                            if j < len(row) and header.strip() and str(row[j]).strip():
                                kv_parts.append(f"{header}: {row[j]}")
                        if kv_parts:
                            kv_phrase = ", ".join(kv_parts)
                            phrases.append(kv_phrase)
                            phrases.append(self._normalize_text(kv_phrase))
                    
                    row_phrases.append(f"row_{i+1}")
            logger.debug(
                f"Table {table.table_index}: Extracted {len(row_phrases)} row phrase(s) "
                f"(from first {len(rows)} rows, multiple formats)"
            )
        
        # Also use flattened text as phrases (multiple samples)
        if table.table_text:
            lines = table.table_text.split("\n")[:3]  # First 3 lines
            for line in lines:
                if line.strip():
                    phrases.append(line.strip())  # Original
                    phrases.append(self._normalize_text(line.strip()))  # Normalized
            logger.debug(
                f"Table {table.table_index}: Added flattened text phrases "
                f"(from first {len(lines)} lines, normalized)"
            )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            if phrase and phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)
        
        logger.debug(
            f"Table {table.table_index}: Total {len(unique_phrases)} unique key phrase(s) extracted "
            f"(after deduplication from {len(phrases)} total)"
        )
        
        return unique_phrases
    
    def _find_table_region(
        self,
        text: str,
        table: ProcessedTable,
        key_phrases: List[str],
    ) -> Optional[TableRegion]:
        """
        Find the region in text that contains table content.
        
        Args:
            text: Document text to search
            table: ProcessedTable to find
            key_phrases: Key phrases extracted from table
        
        Returns:
            TableRegion if found, None otherwise
        """
        if not key_phrases:
            return None
        
        # Try to find table content using multiple strategies
        best_match = None
        best_confidence = 0.0
        best_strategy = None
        
        # Strategy 1: Find exact or fuzzy match of flattened table text
        if table.table_text:
            logger.debug(
                f"Table {table.table_index}: Trying Strategy 1 (fuzzy match on flattened text, "
                f"length: {len(table.table_text)} chars)"
            )
            match_pos = self._fuzzy_match_table_in_text(table.table_text, text)
            if match_pos:
                start, end, confidence = match_pos
                logger.debug(
                    f"Table {table.table_index}: Strategy 1 found match "
                    f"(pos: {start}-{end}, confidence: {confidence:.2f})"
                )
                if confidence > best_confidence:
                    best_match = (start, end, confidence)
                    best_confidence = confidence
                    best_strategy = "fuzzy_text_match"
            else:
                logger.debug(
                    f"Table {table.table_index}: Strategy 1 - no match found "
                    f"(min_match_ratio: {self.min_match_ratio})"
                )
        
        # Strategy 2: Find key phrases using fuzzy matching
        logger.debug(
            f"Table {table.table_index}: Trying Strategy 2 (fuzzy key phrase matching, "
            f"{len(key_phrases)} phrases)"
        )
        phrase_matches = []
        matched_phrases = []
        normalized_text = self._normalize_text(text)  # Normalize once for efficiency
        
        for phrase in key_phrases:
            if len(phrase) < 5:  # Lowered threshold from 10
                logger.debug(
                    f"Table {table.table_index}: Skipping very short phrase (length: {len(phrase)})"
                )
                continue
            
            # Try exact match first (case-insensitive)
            pattern = re.escape(phrase)
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if matches:
                phrase_matches.extend([(m.start(), m.end()) for m in matches])
                matched_phrases.append(phrase[:50])
                logger.debug(
                    f"Table {table.table_index}: Found exact phrase match: '{phrase[:50]}...' "
                    f"({len(matches)} occurrence(s))"
                )
            else:
                # Try normalized matching
                normalized_phrase = self._normalize_text(phrase)
                if normalized_phrase and len(normalized_phrase) >= 5:
                    # Search in normalized text
                    pattern_norm = re.escape(normalized_phrase)
                    matches_norm = list(re.finditer(pattern_norm, normalized_text))
                    
                    if matches_norm:
                        # Convert back to original text positions (approximate)
                        # Since normalization changes length, we need to find in original text
                        # Use fuzzy matching on original text
                        fuzzy_matches = self._fuzzy_find_phrase(phrase, text)
                        if fuzzy_matches:
                            phrase_matches.extend(fuzzy_matches)
                            matched_phrases.append(f"{phrase[:50]}... (fuzzy)")
                            logger.debug(
                                f"Table {table.table_index}: Found fuzzy phrase match: '{phrase[:50]}...' "
                                f"({len(fuzzy_matches)} occurrence(s))"
                            )
                        else:
                            logger.debug(
                                f"Table {table.table_index}: Phrase not found (exact or fuzzy): '{phrase[:50]}...'"
                            )
                    else:
                        logger.debug(
                            f"Table {table.table_index}: Phrase not found: '{phrase[:50]}...'"
                        )
                else:
                    logger.debug(
                        f"Table {table.table_index}: Phrase too short after normalization: '{phrase[:50]}...'"
                    )
        
        if phrase_matches:
            # Find the region that contains most phrase matches
            region_start = min(pos[0] for pos in phrase_matches)
            region_end = max(pos[1] for pos in phrase_matches)
            
            # Expand region to include surrounding context
            # Estimate table size based on row count
            row_count = table.metadata.get("row_count", 0)
            estimated_length = row_count * 50  # Rough estimate: 50 chars per row
            
            # Expand region
            expanded_start = max(0, region_start - 50)
            expanded_end = min(len(text), region_end + estimated_length)
            
            # Calculate confidence based on number of matches
            confidence = min(1.0, len(phrase_matches) / max(1, len(key_phrases)))
            
            logger.debug(
                f"Table {table.table_index}: Strategy 2 found region "
                f"(pos: {expanded_start}-{expanded_end}, "
                f"confidence: {confidence:.2f}, "
                f"matched {len(matched_phrases)}/{len(key_phrases)} phrases)"
            )
            
            if confidence > best_confidence:
                best_match = (expanded_start, expanded_end, confidence)
                best_confidence = confidence
                best_strategy = "key_phrase_matching"
        else:
            logger.debug(
                f"Table {table.table_index}: Strategy 2 - no phrase matches found "
                f"(tried {len([p for p in key_phrases if len(p) >= 10])} phrases)"
            )
        
        if best_match and best_confidence >= self.min_match_ratio:
            start, end, confidence = best_match
            logger.debug(
                f"Table {table.table_index}: Best match found using {best_strategy}: "
                f"confidence {confidence:.2f} >= {self.min_match_ratio} (threshold)"
            )
            return TableRegion(start, end, table, confidence)
        else:
            if best_match:
                logger.debug(
                    f"Table {table.table_index}: Match found but confidence too low: "
                    f"{best_confidence:.2f} < {self.min_match_ratio} (threshold)"
                )
            else:
                logger.debug(
                    f"Table {table.table_index}: No match found with any strategy"
                )
        
        return None
    
    def _fuzzy_find_phrase(
        self,
        phrase: str,
        text: str,
        min_similarity: float = 0.7,
    ) -> List[Tuple[int, int]]:
        """
        Find phrase in text using fuzzy matching.
        
        Args:
            phrase: Phrase to search for
            text: Text to search in
            min_similarity: Minimum similarity ratio (0.0-1.0)
        
        Returns:
            List of (start_pos, end_pos) tuples for matches
        """
        if not phrase or len(phrase) < 5:
            return []
        
        matches = []
        phrase_len = len(phrase)
        normalized_phrase = self._normalize_text(phrase)
        
        # Try sliding window approach
        window_size = phrase_len
        step_size = max(10, phrase_len // 4)
        
        for i in range(0, len(text) - window_size + 1, step_size):
            window = text[i : i + window_size]
            normalized_window = self._normalize_text(window)
            
            # Calculate similarity
            ratio = SequenceMatcher(None, normalized_phrase, normalized_window).ratio()
            
            if ratio >= min_similarity:
                matches.append((i, i + window_size))
        
        return matches
    
    def _fuzzy_match_table_in_text(
        self,
        table_text: str,
        document_text: str,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find fuzzy match of table text in document text.
        
        Uses multiple strategies:
        1. Exact match (normalized)
        2. Sliding window with fuzzy matching
        3. Multiple samples from table (not just first part)
        
        Args:
            table_text: Flattened table text to search for
            document_text: Full document text
        
        Returns:
            Tuple of (start_pos, end_pos, confidence) if found, None otherwise
        """
        if not table_text or len(table_text) < 20:
            return None
        
        # Normalize both texts for better matching
        normalized_table_text = self._normalize_text(table_text)
        normalized_doc_text = self._normalize_text(document_text)
        
        # Strategy 1: Try exact match on normalized text
        search_text = normalized_table_text[: self.max_search_length]
        start_pos = normalized_doc_text.find(search_text)
        if start_pos != -1:
            # Find corresponding position in original text (approximate)
            # Since normalization changes length, we search around the position
            approx_pos = min(start_pos, len(document_text) - len(search_text))
            return (approx_pos, approx_pos + len(search_text), 1.0)
        
        # Strategy 2: Try multiple samples from table (not just first part)
        samples = [
            normalized_table_text[: self.max_search_length],  # First part
        ]
        
        # Add middle part if table is long enough
        if len(normalized_table_text) > self.max_search_length * 2:
            mid_start = len(normalized_table_text) // 2
            mid_end = min(mid_start + self.max_search_length, len(normalized_table_text))
            samples.append(normalized_table_text[mid_start:mid_end])
        
        # Add last part if table is long enough
        if len(normalized_table_text) > self.max_search_length:
            samples.append(normalized_table_text[-self.max_search_length:])
        
        best_match = None
        best_ratio = 0.0
        
        for sample in samples:
            if len(sample) < 20:
                continue
            
            # Try fuzzy matching using SequenceMatcher with sliding window
            chunk_size = len(sample)
            step_size = max(50, chunk_size // 8)  # Smaller steps for better coverage
            
            for i in range(0, len(normalized_doc_text) - chunk_size + 1, step_size):
                chunk = normalized_doc_text[i : i + chunk_size]
                ratio = SequenceMatcher(None, sample, chunk).ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    # Convert back to approximate position in original text
                    approx_pos = min(i, len(document_text) - chunk_size)
                    best_match = (approx_pos, approx_pos + chunk_size, ratio)
        
        if best_match and best_ratio >= self.min_match_ratio:
            return best_match
        
        return None
    
    def _merge_overlapping_regions(
        self, regions: List[TableRegion]
    ) -> List[TableRegion]:
        """
        Merge overlapping table regions.
        
        Args:
            regions: List of TableRegion objects (should be sorted by start_pos)
        
        Returns:
            List of merged regions
        """
        if not regions:
            return []
        
        merged = []
        current = regions[0]
        
        for next_region in regions[1:]:
            # Check if regions overlap
            if current.end_pos >= next_region.start_pos:
                # Merge regions
                current = TableRegion(
                    start_pos=current.start_pos,
                    end_pos=max(current.end_pos, next_region.end_pos),
                    table=current.table,  # Keep first table reference
                    confidence=max(current.confidence, next_region.confidence),
                )
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_region
        
        # Add the last region
        merged.append(current)
        
        return merged

