"""
Graph builder utilities implementing best practices for Neo4j graph construction.

This module provides helper functions following the Implementation Best Practices:
1. Store References, Not Duplicate Content
2. Batch Operations for Graph Building
3. Relationship Properties for Ranking/Weight
"""
from typing import List, Dict, Any, Optional
from neo4j import Driver
import logging

from graph_schema import (
    NODE_LABELS,
    RELATIONSHIP_TYPES,
    PROPERTY_KEYS,
    RELATIONSHIP_PROPERTIES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Best Practice 1: Store References, Not Duplicate Content
# ============================================================================

def create_document_node_data(
    document_id: str,
    title: str,
    source: str,
    document_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create document node data following best practices.
    
    Stores only reference identifiers and lightweight metadata.
    Does NOT store full document content.
    
    Args:
        document_id: Unique document identifier
        title: Document title
        source: Document source/path
        document_type: Type of document (optional)
        metadata: Lightweight metadata dict (optional)
    
    Returns:
        Dictionary with document node properties
    """
    node_data = {
        PROPERTY_KEYS['DOCUMENT_ID']: document_id,
        PROPERTY_KEYS['TITLE']: title,
        PROPERTY_KEYS['SOURCE']: source,
    }
    
    if document_type:
        node_data[PROPERTY_KEYS['DOCUMENT_TYPE']] = document_type
    
    if metadata:
        node_data[PROPERTY_KEYS['METADATA']] = metadata
    
    return node_data


def create_chunk_node_data(
    chunk_id: str,
    chunk_index: int,
    chunk_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create chunk node data following best practices.
    
    Stores only reference identifier and metadata.
    Does NOT store full chunk content (retrieve from Supabase/Qdrant using chunk_id).
    
    Args:
        chunk_id: Unique chunk identifier (links to content in Supabase/Qdrant)
        chunk_index: Position of chunk in document
        chunk_type: Type of chunk (text, table, image, mixed)
        metadata: Lightweight metadata dict (optional, e.g., {"page": 10})
    
    Returns:
        Dictionary with chunk node properties
    """
    node_data = {
        PROPERTY_KEYS['CHUNK_ID']: chunk_id,
        PROPERTY_KEYS['CHUNK_INDEX']: chunk_index,
        PROPERTY_KEYS['CHUNK_TYPE']: chunk_type,
    }
    
    if metadata:
        node_data[PROPERTY_KEYS['METADATA']] = metadata
    
    return node_data


def create_section_node_data(
    section_id: str,
    section_title: str,
    section_index: int
) -> Dict[str, Any]:
    """
    Create section node data following best practices.
    
    Args:
        section_id: Unique section identifier
        section_title: Section title/heading
        section_index: Position of section in document
    
    Returns:
        Dictionary with section node properties
    """
    return {
        PROPERTY_KEYS['SECTION_ID']: section_id,
        PROPERTY_KEYS['SECTION_TITLE']: section_title,
        PROPERTY_KEYS['SECTION_INDEX']: section_index,
    }


def create_entity_node_data(
    entity_id: str,
    entity_name: str,
    entity_type: str,
    entity_value: Optional[str] = None,
    confidence: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create entity node data following best practices.
    
    Args:
        entity_id: Unique entity identifier
        entity_name: Name of the entity
        entity_type: Type (PERSON, ORGANIZATION, CONCEPT, LOCATION, METRIC, DOMAIN_TERM)
        entity_value: Value for metrics (optional)
        confidence: Entity extraction confidence score (optional)
    
    Returns:
        Dictionary with entity node properties
    """
    node_data = {
        PROPERTY_KEYS['ENTITY_ID']: entity_id,
        PROPERTY_KEYS['ENTITY_NAME']: entity_name,
        PROPERTY_KEYS['ENTITY_TYPE']: entity_type,
    }
    
    if entity_value:
        node_data[PROPERTY_KEYS['ENTITY_VALUE']] = entity_value
    
    if confidence is not None:
        node_data[PROPERTY_KEYS['CONFIDENCE']] = confidence
    
    return node_data


def create_media_node_data(
    media_id: str,
    media_type: str,
    media_url: str,
    caption: Optional[str] = None,
    alt_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create media node data following best practices.
    
    Args:
        media_id: Unique media identifier
        media_type: Type (IMAGE, TABLE, CHART, DIAGRAM)
        media_url: URL/path to media file
        caption: Media caption (optional)
        alt_text: Alt text for accessibility (optional)
    
    Returns:
        Dictionary with media node properties
    """
    node_data = {
        PROPERTY_KEYS['MEDIA_ID']: media_id,
        PROPERTY_KEYS['MEDIA_TYPE']: media_type,
        PROPERTY_KEYS['MEDIA_URL']: media_url,
    }
    
    if caption:
        node_data[PROPERTY_KEYS['CAPTION']] = caption
    
    if alt_text:
        node_data[PROPERTY_KEYS['ALT_TEXT']] = alt_text
    
    return node_data


# ============================================================================
# Best Practice 2: Batch Operations for Graph Building
# ============================================================================

def create_document_graph_batch(
    driver: Driver,
    document_data: Dict[str, Any],
    database: Optional[str] = None
) -> None:
    """
    Create entire document graph structure in a single batch transaction.
    
    This is 10-100x faster than creating nodes one-by-one.
    
    Args:
        driver: Neo4j driver instance
        document_data: Dictionary with structure:
            {
                "document_id": str,
                "title": str,
                "source": str,
                "sections": [
                    {
                        "section_id": str,
                        "title": str,
                        "index": int,
                        "chunks": [
                            {
                                "chunk_id": str,
                                "chunk_index": int,
                                "chunk_type": str,
                                "metadata": dict (optional)
                            }
                        ]
                    }
                ]
            }
        database: Database name (optional, uses default if not provided)
    """
    query = f"""
    UNWIND $data AS doc
    MERGE (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: doc.document_id}})
    SET d.{PROPERTY_KEYS['TITLE']} = doc.title,
        d.{PROPERTY_KEYS['SOURCE']} = doc.source
    WITH d, doc
    UNWIND doc.sections AS section
    MERGE (s:{NODE_LABELS['SECTION']} {{{PROPERTY_KEYS['SECTION_ID']}: section.section_id}})
    SET s.{PROPERTY_KEYS['SECTION_TITLE']} = section.title,
        s.{PROPERTY_KEYS['SECTION_INDEX']} = section.index
    MERGE (d)-[r1:{RELATIONSHIP_TYPES['HAS_SECTION']} {{
        {RELATIONSHIP_PROPERTIES['ORDER']}: section.index
    }}]->(s)
    WITH s, section
    UNWIND section.chunks AS chunk
    MERGE (c:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: chunk.chunk_id}})
    SET c.{PROPERTY_KEYS['CHUNK_INDEX']} = chunk.chunk_index,
        c.{PROPERTY_KEYS['CHUNK_TYPE']} = chunk.chunk_type,
        c.{PROPERTY_KEYS['CONTENT']} = chunk.text
    MERGE (s)-[r2:{RELATIONSHIP_TYPES['HAS_CHUNK']} {{
        {RELATIONSHIP_PROPERTIES['ORDER']}: chunk.chunk_index,
        {RELATIONSHIP_PROPERTIES['CHUNK_INDEX']}: chunk.chunk_index
    }}]->(c)
    """
    
    session = driver.session(database=database) if database else driver.session()
    try:
        session.run(query, data=[document_data])
        logger.info(f"Created document graph for: {document_data.get('document_id')}")
    finally:
        session.close()


def create_chunks_batch(
    driver: Driver,
    chunks: List[Dict[str, Any]],
    database: Optional[str] = None
) -> None:
    """
    Create multiple chunk nodes in a single batch operation.
    
    Args:
        driver: Neo4j driver instance
        chunks: List of chunk node data dictionaries
        database: Database name (optional)
    """
    query = f"""
    UNWIND $chunks AS chunk
    MERGE (c:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: chunk.chunk_id}})
    SET c.{PROPERTY_KEYS['CHUNK_INDEX']} = chunk.chunk_index,
        c.{PROPERTY_KEYS['CHUNK_TYPE']} = chunk.chunk_type
    """
    
    if chunks and PROPERTY_KEYS['METADATA'] in chunks[0]:
        query += f", c.{PROPERTY_KEYS['METADATA']} = chunk.metadata"
    
    session = driver.session(database=database) if database else driver.session()
    try:
        session.run(query, chunks=chunks)
        logger.info(f"Created {len(chunks)} chunk nodes in batch")
    finally:
        session.close()


def create_relationships_batch(
    driver: Driver,
    relationships: List[Dict[str, Any]],
    database: Optional[str] = None
) -> None:
    """
    Create multiple relationships in a single batch operation.
    
    Args:
        driver: Neo4j driver instance
        relationships: List of relationship dictionaries with:
            {
                "from_label": str,
                "from_id_key": str,
                "from_id_value": str,
                "to_label": str,
                "to_id_key": str,
                "to_id_value": str,
                "relationship_type": str,
                "properties": dict (optional)
            }
        database: Database name (optional)
    """
    if not relationships:
        return
    
    # Process relationships one by one (can be optimized later)
    # This is simpler and more reliable than complex UNWIND query
    session = driver.session(database=database) if database else driver.session()
    try:
        for rel in relationships:
            from_label = rel['from_label']
            from_id_key = rel['from_id_key']
            from_id_value = rel['from_id_value']
            to_label = rel['to_label']
            to_id_key = rel['to_id_key']
            to_id_value = rel['to_id_value']
            rel_type = rel['relationship_type']
            properties = rel.get('properties', {})
            
            # Build query with relationship type
            if properties:
                props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                query = f"""
                MATCH (from:{from_label} {{{from_id_key}: $from_id}})
                MATCH (to:{to_label} {{{to_id_key}: $to_id}})
                MERGE (from)-[r:{rel_type}]->(to)
                SET r += {{{props_str}}}
                """
                params = {
                    "from_id": from_id_value,
                    "to_id": to_id_value,
                    **properties
                }
            else:
                query = f"""
                MATCH (from:{from_label} {{{from_id_key}: $from_id}})
                MATCH (to:{to_label} {{{to_id_key}: $to_id}})
                MERGE (from)-[r:{rel_type}]->(to)
                """
                params = {
                    "from_id": from_id_value,
                    "to_id": to_id_value,
                }
            
            session.run(query, params)
        
        logger.info(f"Created {len(relationships)} relationships in batch")
    finally:
        session.close()


# ============================================================================
# Best Practice 3: Relationship Properties for Ranking/Weight
# ============================================================================

def create_relationship_with_properties(
    driver: Driver,
    from_node_label: str,
    from_node_id_key: str,
    from_node_id_value: str,
    to_node_label: str,
    to_node_id_key: str,
    to_node_id_value: str,
    relationship_type: str,
    properties: Dict[str, Any],
    database: Optional[str] = None
) -> None:
    """
    Create relationship with properties for ranking and filtering.
    
    Properties enable better retrieval quality by storing:
    - Order/position (order, chunk_index)
    - Importance/weight (importance, frequency, context)
    - Temporal/version (created_at, version)
    - Metadata (media_type, position, relevance)
    
    Args:
        driver: Neo4j driver instance
        from_node_label: Label of source node
        from_node_id_key: Property key for source node ID
        from_node_id_value: Value of source node ID
        to_node_label: Label of target node
        to_node_id_key: Property key for target node ID
        to_node_id_value: Value of target node ID
        relationship_type: Type of relationship
        properties: Dictionary of relationship properties
        database: Database name (optional)
    """
    query = f"""
    MATCH (from:{from_node_label} {{{from_node_id_key}: $from_id}})
    MATCH (to:{to_node_label} {{{to_node_id_key}: $to_id}})
    MERGE (from)-[r:{relationship_type}]->(to)
    SET r += $properties
    """
    
    session = driver.session(database=database) if database else driver.session()
    try:
        session.run(query, {
            "from_id": from_node_id_value,
            "to_id": to_node_id_value,
            "properties": properties
        })
        logger.debug(f"Created {relationship_type} relationship with properties")
    finally:
        session.close()


def create_mentions_relationship(
    driver: Driver,
    chunk_id: str,
    entity_id: str,
    frequency: int = 1,
    importance: float = 0.5,
    context: Optional[str] = None,
    database: Optional[str] = None
) -> None:
    """
    Create MENTIONS relationship with importance/weight properties.
    
    Args:
        driver: Neo4j driver instance
        chunk_id: Chunk identifier
        entity_id: Entity identifier
        frequency: How often entity appears in chunk
        importance: Relevance score (0.0-1.0)
        context: Where entity appears (title, body, metadata)
        database: Database name (optional)
    """
    properties = {
        RELATIONSHIP_PROPERTIES['FREQUENCY']: frequency,
        RELATIONSHIP_PROPERTIES['IMPORTANCE']: importance,
    }
    
    if context:
        properties[RELATIONSHIP_PROPERTIES['CONTEXT']] = context
    
    create_relationship_with_properties(
        driver=driver,
        from_node_label=NODE_LABELS['CHUNK'],
        from_node_id_key=PROPERTY_KEYS['CHUNK_ID'],
        from_node_id_value=chunk_id,
        to_node_label=NODE_LABELS['ENTITY'],
        to_node_id_key=PROPERTY_KEYS['ENTITY_ID'],
        to_node_id_value=entity_id,
        relationship_type=RELATIONSHIP_TYPES['MENTIONS'],
        properties=properties,
        database=database
    )


def create_about_relationship(
    driver: Driver,
    section_id: str,
    entity_id: str,
    importance: float = 0.5,
    frequency: int = 1,
    database: Optional[str] = None
) -> None:
    """
    Create ABOUT relationship with importance/weight properties.
    
    Args:
        driver: Neo4j driver instance
        section_id: Section identifier
        entity_id: Entity identifier
        importance: Relevance score (0.0-1.0)
        frequency: How often entity appears in section
        database: Database name (optional)
    """
    properties = {
        RELATIONSHIP_PROPERTIES['IMPORTANCE']: importance,
        RELATIONSHIP_PROPERTIES['FREQUENCY']: frequency,
    }
    
    create_relationship_with_properties(
        driver=driver,
        from_node_label=NODE_LABELS['SECTION'],
        from_node_id_key=PROPERTY_KEYS['SECTION_ID'],
        from_node_id_value=section_id,
        to_node_label=NODE_LABELS['ENTITY'],
        to_node_id_key=PROPERTY_KEYS['ENTITY_ID'],
        to_node_id_value=entity_id,
        relationship_type=RELATIONSHIP_TYPES['ABOUT'],
        properties=properties,
        database=database
    )


def create_has_media_relationship(
    driver: Driver,
    chunk_id: str,
    media_id: str,
    position: str = "after",
    relevance: float = 0.5,
    media_type: Optional[str] = None,
    database: Optional[str] = None
) -> None:
    """
    Create HAS_MEDIA relationship with metadata properties.
    
    Args:
        driver: Neo4j driver instance
        chunk_id: Chunk identifier
        media_id: Media identifier
        position: Where media appears (before, after, inline)
        relevance: How relevant media is to chunk (0.0-1.0)
        media_type: Type of media (optional, for filtering)
        database: Database name (optional)
    """
    properties = {
        RELATIONSHIP_PROPERTIES['POSITION']: position,
        RELATIONSHIP_PROPERTIES['RELEVANCE']: relevance,
    }
    
    if media_type:
        properties[RELATIONSHIP_PROPERTIES['MEDIA_TYPE']] = media_type
    
    create_relationship_with_properties(
        driver=driver,
        from_node_label=NODE_LABELS['CHUNK'],
        from_node_id_key=PROPERTY_KEYS['CHUNK_ID'],
        from_node_id_value=chunk_id,
        to_node_label=NODE_LABELS['MEDIA'],
        to_node_id_key=PROPERTY_KEYS['MEDIA_ID'],
        to_node_id_value=media_id,
        relationship_type=RELATIONSHIP_TYPES['HAS_MEDIA'],
        properties=properties,
        database=database
    )

