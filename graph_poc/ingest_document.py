"""
Document ingestion script for Graph POC.

Processes a document from file path and stores it in Neo4j knowledge graph.
"""
import sys
import logging
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from document_processor import DocumentProcessor
from graph_builder import (
    create_document_node_data,
    create_entity_node_data,
    create_document_graph_batch,
    create_mentions_relationship,
    create_about_relationship,
)
from graph_schema import NODE_LABELS, PROPERTY_KEYS, RELATIONSHIP_TYPES
from neo4j_connection import get_neo4j_driver, close_neo4j_driver, get_database_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_entity_nodes(driver, entities: list, database: Optional[str] = None):
    """Create entity nodes in batch."""
    if not entities:
        return
    
    query = f"""
    UNWIND $entities AS entity
    MERGE (e:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_ID']}: entity.entity_id}})
    SET e.{PROPERTY_KEYS['ENTITY_NAME']} = entity.entity_name,
        e.{PROPERTY_KEYS['ENTITY_TYPE']} = entity.entity_type,
        e.{PROPERTY_KEYS['CONFIDENCE']} = entity.confidence
    """
    
    # Add entity_value if present
    if entities and 'entity_value' in entities[0] and entities[0]['entity_value']:
        query += f", e.{PROPERTY_KEYS['ENTITY_VALUE']} = entity.entity_value"
    
    session = driver.session(database=database) if database else driver.session()
    try:
        session.run(query, entities=entities)
        logger.info(f"Created {len(entities)} entity nodes")
    finally:
        session.close()


def link_entities_to_chunks(driver, document_data: dict, database: Optional[str] = None):
    """
    Link entities to chunks where they appear.
    
    For simplicity, we link entities to chunks in the same section.
    In production, use proper entity linking/NER.
    """
    relationships = []
    
    for section in document_data['sections']:
        section_text = ' '.join([chunk.get('text', '') for chunk in section['chunks']])
        section_text_lower = section_text.lower()
        
        for entity in document_data['entities']:
            entity_name_lower = entity['entity_name'].lower()
            
            # Check if entity appears in section
            if entity_name_lower in section_text_lower:
                # Count frequency
                frequency = section_text_lower.count(entity_name_lower)
                
                # Link to all chunks in section (simplified)
                for chunk in section['chunks']:
                    chunk_text_lower = chunk.get('text', '').lower()
                    if entity_name_lower in chunk_text_lower:
                        relationships.append({
                            "from_label": NODE_LABELS['CHUNK'],
                            "from_id_key": PROPERTY_KEYS['CHUNK_ID'],
                            "from_id_value": chunk['chunk_id'],
                            "to_label": NODE_LABELS['ENTITY'],
                            "to_id_key": PROPERTY_KEYS['ENTITY_ID'],
                            "to_id_value": entity['entity_id'],
                            "relationship_type": RELATIONSHIP_TYPES['MENTIONS'],
                            "properties": {
                                "frequency": frequency,
                                "importance": min(0.9, 0.5 + (frequency * 0.1)),  # Simple importance calculation
                            }
                        })
                
                # Link section to entity (top entities)
                if frequency >= 2:  # Only link if entity appears multiple times
                    relationships.append({
                        "from_label": NODE_LABELS['SECTION'],
                        "from_id_key": PROPERTY_KEYS['SECTION_ID'],
                        "from_id_value": section['section_id'],
                        "to_label": NODE_LABELS['ENTITY'],
                        "to_id_key": PROPERTY_KEYS['ENTITY_ID'],
                        "to_id_value": entity['entity_id'],
                        "relationship_type": RELATIONSHIP_TYPES['ABOUT'],
                        "properties": {
                            "importance": min(0.9, 0.5 + (frequency * 0.1)),
                            "frequency": frequency,
                        }
                    })
    
    # Create relationships in batch
    if relationships:
        from graph_builder import create_relationships_batch
        create_relationships_batch(driver, relationships, database)
        logger.info(f"Created {len(relationships)} entity relationships")


def ingest_document(file_path: str, database: Optional[str] = None) -> str:
    """
    Ingest a document and store it in Neo4j knowledge graph.
    
    Args:
        file_path: Path to document file
        database: Neo4j database name (optional)
    
    Returns:
        Document ID
    """
    logger.info("=" * 60)
    logger.info("Document Ingestion - Graph POC")
    logger.info("=" * 60)
    
    # Validate file exists
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Process document
    logger.info(f"\n[Step 1/4] Processing document: {file_path}")
    processor = DocumentProcessor()
    document_data = processor.process_document(file_path)
    
    document_id = document_data['document_id']
    logger.info(f"Document ID: {document_id}")
    logger.info(f"  - Sections: {len(document_data['sections'])}")
    logger.info(f"  - Chunks: {sum(len(s['chunks']) for s in document_data['sections'])}")
    logger.info(f"  - Entities: {len(document_data['entities'])}")
    
    # Connect to Neo4j
    logger.info(f"\n[Step 2/4] Connecting to Neo4j...")
    driver = get_neo4j_driver()
    db_name = database or get_database_name()
    
    try:
        # Create document graph structure (Document -> Sections -> Chunks)
        logger.info(f"\n[Step 3/4] Creating document graph structure...")
        from graph_builder import create_document_graph_batch
        create_document_graph_batch(driver, document_data, database=db_name)
        logger.info("[OK] Document structure created")
        
        # Create entity nodes
        if document_data['entities']:
            logger.info(f"\n[Step 4/4] Creating entity nodes and relationships...")
            create_entity_nodes(driver, document_data['entities'], database=db_name)
            
            # Link entities to chunks and sections
            link_entities_to_chunks(driver, document_data, database=db_name)
            logger.info("[OK] Entities linked to chunks and sections")
        else:
            logger.info("[INFO] No entities found to link")
        
        logger.info("\n" + "=" * 60)
        logger.info("[SUCCESS] Document ingestion complete!")
        logger.info("=" * 60)
        logger.info(f"\nDocument ID: {document_id}")
        logger.info(f"You can now query the graph using this document_id.")
        
        return document_id
        
    except Exception as e:
        logger.error(f"\n[ERROR] Failed to ingest document: {e}")
        raise
    finally:
        close_neo4j_driver()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python ingest_document.py <file_path>")
        print("\nExample:")
        print("  python ingest_document.py document.pdf")
        print("  python ingest_document.py document.txt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        document_id = ingest_document(file_path)
        print(f"\n[SUCCESS] Document ingested with ID: {document_id}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to ingest document: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

