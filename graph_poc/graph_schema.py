"""
Neo4j graph schema definitions.

This module contains Cypher queries and schema definitions for the knowledge graph.
Following the semantic, hierarchical design from graph.md:
- Document -> Section -> Chunk (vertical structure)
- Chunk -> Entity (horizontal meaning)
- Section -> Entity (top entities)
- Entity -> Entity (semantic relationships)
- Chunk -> Media (multimodal content)
- Media -> Entity (descriptions)
"""

# Node labels
NODE_LABELS = {
    "DOCUMENT": "Document",
    "SECTION": "Section",
    "CHUNK": "Chunk",
    "ENTITY": "Entity",
    "MEDIA": "Media",
}

# Relationship types
RELATIONSHIP_TYPES = {
    "HAS_SECTION": "HAS_SECTION",      # Document -> Section
    "HAS_CHUNK": "HAS_CHUNK",          # Section -> Chunk
    "MENTIONS": "MENTIONS",             # Chunk -> Entity
    "ABOUT": "ABOUT",                   # Section -> Entity (top entities)
    "RELATED_TO": "RELATED_TO",         # Entity -> Entity (semantic relationships)
    "HAS_MEDIA": "HAS_MEDIA",           # Chunk -> Media
    "DESCRIBES": "DESCRIBES",           # Media -> Entity
}

# Property keys (standardized)
PROPERTY_KEYS = {
    # Document properties
    "DOCUMENT_ID": "document_id",
    "TITLE": "title",
    "SOURCE": "source",
    "METADATA": "metadata",
    "CREATED_AT": "created_at",
    "DOCUMENT_TYPE": "document_type",
    
    # Section properties
    "SECTION_ID": "section_id",
    "SECTION_TITLE": "section_title",
    "SECTION_INDEX": "section_index",
    
    # Chunk properties
    "CHUNK_ID": "chunk_id",
    "CONTENT": "content",  # Note: In production, store only chunk_id reference, not full content
    "CHUNK_INDEX": "chunk_index",
    "CHUNK_TYPE": "chunk_type",  # text, table, image, mixed
    
    # Entity properties
    "ENTITY_ID": "entity_id",
    "ENTITY_TYPE": "entity_type",  # PERSON, ORGANIZATION, CONCEPT, LOCATION, METRIC, DOMAIN_TERM
    "ENTITY_NAME": "entity_name",
    "ENTITY_VALUE": "entity_value",  # For metrics (e.g., "5.2%", "$1M")
    "CONFIDENCE": "confidence",  # Entity extraction confidence score
    
    # Media properties
    "MEDIA_ID": "media_id",
    "MEDIA_TYPE": "media_type",  # IMAGE, TABLE, CHART, DIAGRAM
    "MEDIA_URL": "media_url",
    "CAPTION": "caption",
    "ALT_TEXT": "alt_text",
}

# Entity types enum
ENTITY_TYPES = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "CONCEPT": "CONCEPT",
    "LOCATION": "LOCATION",
    "METRIC": "METRIC",
    "DOMAIN_TERM": "DOMAIN_TERM",
}

# Media types enum
MEDIA_TYPES = {
    "IMAGE": "IMAGE",
    "TABLE": "TABLE",
    "CHART": "CHART",
    "DIAGRAM": "DIAGRAM",
}

# Chunk types enum
CHUNK_TYPES = {
    "TEXT": "text",
    "TABLE": "table",
    "IMAGE": "image",
    "MIXED": "mixed",
}

# Relationship property keys (for ranking, ordering, etc.)
RELATIONSHIP_PROPERTIES = {
    # Order/Position
    "ORDER": "order",
    "CHUNK_INDEX": "chunk_index",
    
    # Importance/Weight
    "IMPORTANCE": "importance",  # 0.0-1.0 relevance score
    "FREQUENCY": "frequency",   # How often entity appears
    "CONTEXT": "context",       # Where entity appears (title, body, metadata)
    
    # Temporal/Version
    "CREATED_AT": "created_at",
    "VERSION": "version",
    
    # Media-specific
    "POSITION": "position",     # Where media appears relative to chunk (before, after, inline)
    "RELEVANCE": "relevance",   # How relevant media is to chunk content (0.0-1.0)
    "MEDIA_TYPE": "media_type", # Type of media on relationship
}

# Schema validation helpers
def validate_node_label(label: str) -> bool:
    """Validate that a node label is in the schema."""
    return label in NODE_LABELS.values()


def validate_relationship_type(rel_type: str) -> bool:
    """Validate that a relationship type is in the schema."""
    return rel_type in RELATIONSHIP_TYPES.values()


def validate_entity_type(entity_type: str) -> bool:
    """Validate that an entity type is valid."""
    return entity_type in ENTITY_TYPES.values()


def validate_media_type(media_type: str) -> bool:
    """Validate that a media type is valid."""
    return media_type in MEDIA_TYPES.values()


def validate_chunk_type(chunk_type: str) -> bool:
    """Validate that a chunk type is valid."""
    return chunk_type in CHUNK_TYPES.values()


# Schema documentation strings
SCHEMA_DOCUMENTATION = """
Graph Schema Overview:

NODES:
1. Document: Root node representing a single document
   - Properties: document_id (unique), title, source, metadata, created_at, document_type
   
2. Section: Logical document segments (chapter headings, sections)
   - Properties: section_id (unique), section_title, section_index
   - Connected via: Document -[:HAS_SECTION]-> Section
   
3. Chunk: Text chunks that serve as retrieval granularity
   - Properties: chunk_id (unique), chunk_index, chunk_type, content (reference only in production)
   - Connected via: Section -[:HAS_CHUNK]-> Chunk
   
4. Entity: Meaningful entities (people, organizations, concepts, locations, metrics, domain terms)
   - Properties: entity_id (unique), entity_type, entity_name, entity_value (for metrics), confidence
   - Connected via: Chunk -[:MENTIONS]-> Entity, Section -[:ABOUT]-> Entity
   
5. Media: Images, tables, charts, diagrams
   - Properties: media_id (unique), media_type, media_url, caption, alt_text
   - Connected via: Chunk -[:HAS_MEDIA]-> Media, Media -[:DESCRIBES]-> Entity

RELATIONSHIPS:
- HAS_SECTION: Document -> Section (with order property)
- HAS_CHUNK: Section -> Chunk (with order, chunk_index properties)
- MENTIONS: Chunk -> Entity (with count, context, importance properties)
- ABOUT: Section -> Entity (with importance, frequency properties)
- RELATED_TO: Entity -> Entity (sparse, high-value semantic relationships)
- HAS_MEDIA: Chunk -> Media (with position, relevance, media_type properties)
- DESCRIBES: Media -> Entity (with relevance property)

DESIGN PRINCIPLES:
- Vertical structure: Document -> Sections -> Chunks (hierarchical)
- Horizontal meaning: Chunks -> Entities (semantic connections)
- Sparse entity relationships: Only strong semantic connections between entities
- Compact design: Store references, not duplicate content
- Fast retrieval: Optimized for RAG queries
"""

