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
    "TOPIC": "Topic",
}

# Relationship types
RELATIONSHIP_TYPES = {
    "HAS_SECTION": "HAS_SECTION",      # Document -> Section
    "HAS_CHUNK": "HAS_CHUNK",          # Section -> Chunk
    "NEXT_CHUNK": "NEXT_CHUNK",        # Chunk -> Chunk (sequential within section)
    "MENTIONS": "MENTIONS",             # Chunk -> Entity
    "ABOUT": "ABOUT",                   # Section -> Entity (top entities)
    "RELATED_TO": "RELATED_TO",         # Entity -> Entity (semantic relationships)
    "HAS_MEDIA": "HAS_MEDIA",           # Chunk -> Media
    "DESCRIBES": "DESCRIBES",           # Media -> Entity
    "HAS_TOPIC": "HAS_TOPIC",           # Document/Section/Chunk -> Topic
    "ASSOCIATED_WITH": "ASSOCIATED_WITH", # Entity -> Topic
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
    "ENTITY_TYPE": "entity_type",  # PERSON, ORG, GPE, DATE, MONEY, PERCENT, CONCEPT
    "ENTITY_NAME": "entity_name",
    "ENTITY_NORMALIZED_NAME": "normalized_name",  # Lowercase, trimmed for cross-document matching
    "ENTITY_VALUE": "entity_value",  # For metrics (e.g., "5.2%", "$1M")
    "CONFIDENCE": "confidence",  # Entity extraction confidence score
    
    # Media properties
    "MEDIA_ID": "media_id",
    "MEDIA_TYPE": "media_type",  # IMAGE, TABLE, CHART, DIAGRAM
    "MEDIA_URL": "media_url",
    "CAPTION": "caption",
    "ALT_TEXT": "alt_text",
    
    # Topic properties
    "TOPIC_ID": "topic_id",
    "TOPIC_NAME": "topic_name",
    "TOPIC_NORMALIZED_NAME": "normalized_name",  # Lowercase for matching
    "TOPIC_KEYWORDS": "keywords",  # Array of keywords
    "TOPIC_DESCRIPTION": "description",
}

# Entity types enum (aligned with spaCy NER labels)
ENTITY_TYPES = {
    "PERSON": "PERSON",           # People, including fictional
    "ORG": "ORG",                 # Organizations, companies, agencies
    "GPE": "GPE",                 # Geopolitical entities (countries, cities, states)
    "LOC": "LOC",                 # Non-GPE locations (mountains, water bodies)
    "DATE": "DATE",               # Absolute or relative dates
    "TIME": "TIME",               # Times smaller than a day
    "MONEY": "MONEY",             # Monetary values
    "PERCENT": "PERCENT",         # Percentages
    "PRODUCT": "PRODUCT",         # Products (objects, vehicles, foods, etc.)
    "EVENT": "EVENT",             # Named events (hurricanes, battles, wars)
    "LAW": "LAW",                 # Named documents made into laws
    "WORK_OF_ART": "WORK_OF_ART", # Titles of books, songs, etc.
    "CONCEPT": "CONCEPT",         # Fallback for regex-extracted concepts
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
   - Properties: chunk_id (unique), chunk_index, chunk_type, content
   - Connected via: Section -[:HAS_CHUNK]-> Chunk, Chunk -[:NEXT_CHUNK]-> Chunk
   
4. Entity: Named entities extracted via spaCy NER (PERSON, ORG, GPE, DATE, MONEY, etc.)
   - Properties: entity_id (unique), entity_type, entity_name, normalized_name, confidence
   - normalized_name enables cross-document entity resolution
   - Connected via: Chunk -[:MENTIONS]-> Entity, Section -[:ABOUT]-> Entity
   
5. Media: Images, tables, charts, diagrams
   - Properties: media_id (unique), media_type, media_url, caption, alt_text
   - Connected via: Chunk -[:HAS_MEDIA]-> Media, Media -[:DESCRIBES]-> Entity

6. Topic: Global ontology/thematic nodes for cross-document navigation
   - Properties: topic_id (unique), topic_name, normalized_name, keywords (array), description
   - Topics create semantic bridges across documents
   - Connected via: Document/Section/Chunk -[:HAS_TOPIC]-> Topic, Entity -[:ASSOCIATED_WITH]-> Topic

RELATIONSHIPS:
- HAS_SECTION: Document -> Section (with order property)
- HAS_CHUNK: Section -> Chunk (with order, chunk_index properties)
- NEXT_CHUNK: Chunk -> Chunk (sequential within section, for context expansion)
- MENTIONS: Chunk -> Entity (with frequency, context, importance properties)
- ABOUT: Section -> Entity (with importance, frequency properties)
- RELATED_TO: Entity -> Entity (sparse, high-value semantic relationships)
- HAS_MEDIA: Chunk -> Media (with position, relevance, media_type properties)
- DESCRIBES: Media -> Entity (with relevance property)
- HAS_TOPIC: Document/Section/Chunk -> Topic (with relevance property)
- ASSOCIATED_WITH: Entity -> Topic (with relevance property)

DESIGN PRINCIPLES:
- Vertical structure: Document -> Sections -> Chunks (hierarchical)
- Sequential linking: Chunks linked via NEXT_CHUNK for context expansion
- Cross-document entities: Same entity shares node via normalized_name
- Cross-document topics: Topics create thematic bridges across documents
- Horizontal meaning: Chunks -> Entities (semantic connections)
- Universal design: Works for any domain via spaCy NER
- Fast retrieval: Optimized for RAG queries
- Topic-based navigation: Traverse documents via shared themes
"""

