"""
Graph query system for retrieving chunks from Neo4j knowledge graph.

This module queries the graph to find relevant chunks based on entities,
sections, and document structure.
"""
import logging
from typing import List, Dict, Any, Optional
from neo4j import Driver

from graph_schema import NODE_LABELS, PROPERTY_KEYS, RELATIONSHIP_TYPES
from neo4j_connection import get_neo4j_driver, close_neo4j_driver, get_database_name

logger = logging.getLogger(__name__)


class GraphQuerier:
    """
    Query the Neo4j knowledge graph to retrieve relevant chunks.
    """
    
    def __init__(self, driver: Optional[Driver] = None, database: Optional[str] = None):
        """
        Initialize graph querier.
        
        Args:
            driver: Neo4j driver instance (creates new if not provided)
            database: Database name (uses default if not provided)
        """
        self.driver = driver
        self.database = database or get_database_name()
        self._own_driver = driver is None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.driver:
            self.driver = get_neo4j_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._own_driver and self.driver:
            close_neo4j_driver()
            self.driver = None
    
    def query_by_entity(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query chunks that mention a specific entity.
        
        Args:
            entity_name: Name of the entity to search for
            limit: Maximum number of chunks to return
        
        Returns:
            List of chunk dictionaries
        """
        query = f"""
        MATCH (e:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_NAME']}: $entity_name}})
              <-[:{RELATIONSHIP_TYPES['MENTIONS']}]-(c:{NODE_LABELS['CHUNK']})
              <-[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]-(s:{NODE_LABELS['SECTION']})
              <-[:{RELATIONSHIP_TYPES['HAS_SECTION']}]-(d:{NODE_LABELS['DOCUMENT']})
        RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
        ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        session = self.driver.session(database=self.database)
        try:
            result = session.run(query, entity_name=entity_name, limit=limit)
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "filename": record.get("filename", "Unknown Document"),
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                    }
                })
            return chunks
        finally:
            session.close()
    
    def query_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query chunks by searching for keywords in entity names.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of chunks to return
        
        Returns:
            List of chunk dictionaries
        """
        # Find entities that match keywords
        entity_query = f"""
        MATCH (e:{NODE_LABELS['ENTITY']})
        WHERE ANY(keyword IN $keywords WHERE toLower(e.{PROPERTY_KEYS['ENTITY_NAME']}) CONTAINS toLower(keyword))
        RETURN DISTINCT e.{PROPERTY_KEYS['ENTITY_NAME']} AS entity_name
        LIMIT 20
        """
        
        session = self.driver.session(database=self.database)
        try:
            # Find matching entities
            entity_result = session.run(entity_query, keywords=keywords)
            entity_names = [record["entity_name"] for record in entity_result]
            
            if not entity_names:
                logger.warning(f"No entities found matching keywords: {keywords}")
                return []
            
            # Query chunks for these entities
            chunks = []
            seen_chunk_ids = set()
            
            for entity_name in entity_names[:5]:  # Limit to top 5 entities
                entity_chunks = self.query_by_entity(entity_name, limit=limit)
                for chunk in entity_chunks:
                    if chunk["chunk_id"] not in seen_chunk_ids:
                        chunks.append(chunk)
                        seen_chunk_ids.add(chunk["chunk_id"])
                    if len(chunks) >= limit:
                        break
                if len(chunks) >= limit:
                    break
            
            return chunks[:limit]
        finally:
            session.close()
    
    def query_by_document(self, document_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Query all chunks from a specific document.
        
        Args:
            document_id: Document ID
            limit: Maximum number of chunks to return
        
        Returns:
            List of chunk dictionaries
        """
        query = f"""
        MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $document_id}})
              -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
              -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
        RETURN c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
               c.{PROPERTY_KEYS['CHUNK_INDEX']} AS chunk_index,
               c.{PROPERTY_KEYS['CHUNK_TYPE']} AS chunk_type,
               d.{PROPERTY_KEYS['DOCUMENT_ID']} AS document_id,
               d.{PROPERTY_KEYS['TITLE']} AS filename,
               s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
               s.{PROPERTY_KEYS['SECTION_INDEX']} AS section_index
        ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
        LIMIT $limit
        """
        
        session = self.driver.session(database=self.database)
        try:
            result = session.run(query, document_id=document_id, limit=limit)
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "chunk_type": record.get("chunk_type", "text"),
                    "filename": record.get("filename", "Unknown Document"),
                    "metadata": {
                        "section_title": record.get("section_title"),
                        "section_index": record.get("section_index"),
                    }
                })
            return chunks
        finally:
            session.close()
    
    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        Get chunk content from Neo4j.
        
        Note: In production, chunk content should be retrieved from Supabase/Qdrant.
        For POC, we store content directly in the graph.
        
        Args:
            chunk_id: Chunk ID
        
        Returns:
            Chunk text content or None
        """
        # Try to get content property, fallback to empty if not found
        query = f"""
        MATCH (c:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: $chunk_id}})
        RETURN c.{PROPERTY_KEYS['CONTENT']} AS content
        """
        
        session = self.driver.session(database=self.database)
        try:
            result = session.run(query, chunk_id=chunk_id)
            record = result.single()
            if record:
                content = record.get("content")
                # Return content if it exists and is not empty
                if content:
                    return content
            return None
        except Exception as e:
            logger.warning(f"Failed to get content for chunk {chunk_id}: {e}")
            return None
        finally:
            session.close()
    
    def enrich_chunks_with_content(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich chunk dictionaries with content.
        
        Retrieves chunk content from Neo4j graph.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Enriched chunks with chunk_text field
        """
        enriched = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            content = self.get_chunk_content(chunk_id)
            
            enriched_chunk = chunk.copy()
            if content:
                enriched_chunk["chunk_text"] = content
            else:
                # Content not found - this happens for old documents ingested before content storage
                logger.warning(f"Content not found for chunk {chunk_id}. Document may need to be re-ingested.")
                enriched_chunk["chunk_text"] = f"[Chunk {chunk.get('chunk_index', 'N/A')} - Content not available. Please re-ingest the document to store content.]"
            enriched.append(enriched_chunk)
        
        return enriched

