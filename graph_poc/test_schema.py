"""
Test script for graph schema POC.

This script demonstrates how to use the graph schema definitions
and validates the schema structure.
"""
from graph_schema import (
    NODE_LABELS,
    RELATIONSHIP_TYPES,
    PROPERTY_KEYS,
    ENTITY_TYPES,
    MEDIA_TYPES,
    CHUNK_TYPES,
    RELATIONSHIP_PROPERTIES,
    validate_node_label,
    validate_relationship_type,
    validate_entity_type,
    validate_media_type,
    validate_chunk_type,
    SCHEMA_DOCUMENTATION,
)


def test_schema_constants():
    """Test that all schema constants are properly defined."""
    print("=" * 60)
    print("Testing Graph Schema Constants")
    print("=" * 60)
    
    # Test node labels
    print("\n[Node Labels]")
    for key, value in NODE_LABELS.items():
        print(f"  {key}: {value}")
        assert validate_node_label(value), f"Invalid node label: {value}"
    
    # Test relationship types
    print("\n[Relationship Types]")
    for key, value in RELATIONSHIP_TYPES.items():
        print(f"  {key}: {value}")
        assert validate_relationship_type(value), f"Invalid relationship type: {value}"
    
    # Test entity types
    print("\n[Entity Types]")
    for key, value in ENTITY_TYPES.items():
        print(f"  {key}: {value}")
        assert validate_entity_type(value), f"Invalid entity type: {value}"
    
    # Test media types
    print("\n[Media Types]")
    for key, value in MEDIA_TYPES.items():
        print(f"  {key}: {value}")
        assert validate_media_type(value), f"Invalid media type: {value}"
    
    # Test chunk types
    print("\n[Chunk Types]")
    for key, value in CHUNK_TYPES.items():
        print(f"  {key}: {value}")
        assert validate_chunk_type(value), f"Invalid chunk type: {value}"
    
    print("\n[SUCCESS] All schema constants are valid!")


def test_cypher_examples():
    """Generate example Cypher queries using the schema."""
    print("\n" + "=" * 60)
    print("Example Cypher Queries Using Schema")
    print("=" * 60)
    
    # Example 1: Create a Document node
    print("\n1. Create Document Node:")
    doc_query = f"""
    CREATE (d:{NODE_LABELS['DOCUMENT']} {{
        {PROPERTY_KEYS['DOCUMENT_ID']}: $document_id,
        {PROPERTY_KEYS['TITLE']}: $title,
        {PROPERTY_KEYS['SOURCE']}: $source
    }})
    RETURN d
    """
    print(doc_query)
    
    # Example 2: Create Document -> Section relationship
    print("\n2. Create Document -> Section Relationship:")
    section_query = f"""
    MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $doc_id}})
    MATCH (s:{NODE_LABELS['SECTION']} {{{PROPERTY_KEYS['SECTION_ID']}: $section_id}})
    MERGE (d)-[r:{RELATIONSHIP_TYPES['HAS_SECTION']} {{
        {RELATIONSHIP_PROPERTIES['ORDER']}: $order
    }}]->(s)
    RETURN r
    """
    print(section_query)
    
    # Example 3: Create Chunk -> Entity relationship
    print("\n3. Create Chunk -> Entity Relationship:")
    mentions_query = f"""
    MATCH (c:{NODE_LABELS['CHUNK']} {{{PROPERTY_KEYS['CHUNK_ID']}: $chunk_id}})
    MATCH (e:{NODE_LABELS['ENTITY']} {{{PROPERTY_KEYS['ENTITY_ID']}: $entity_id}})
    MERGE (c)-[r:{RELATIONSHIP_TYPES['MENTIONS']} {{
        {RELATIONSHIP_PROPERTIES['FREQUENCY']}: $frequency,
        {RELATIONSHIP_PROPERTIES['IMPORTANCE']}: $importance
    }}]->(e)
    RETURN r
    """
    print(mentions_query)
    
    # Example 4: Query entities by type
    print("\n4. Query Entities by Type:")
    entity_query = f"""
    MATCH (e:{NODE_LABELS['ENTITY']})
    WHERE e.{PROPERTY_KEYS['ENTITY_TYPE']} = $entity_type
    RETURN e.{PROPERTY_KEYS['ENTITY_NAME']} AS name, 
           e.{PROPERTY_KEYS['ENTITY_TYPE']} AS type
    LIMIT 10
    """
    print(entity_query)
    
    # Example 5: Traverse document structure
    print("\n5. Traverse Document Structure:")
    traverse_query = f"""
    MATCH (d:{NODE_LABELS['DOCUMENT']} {{{PROPERTY_KEYS['DOCUMENT_ID']}: $doc_id}})
           -[:{RELATIONSHIP_TYPES['HAS_SECTION']}]->(s:{NODE_LABELS['SECTION']})
           -[:{RELATIONSHIP_TYPES['HAS_CHUNK']}]->(c:{NODE_LABELS['CHUNK']})
           -[:{RELATIONSHIP_TYPES['MENTIONS']}]->(e:{NODE_LABELS['ENTITY']})
    RETURN d.{PROPERTY_KEYS['TITLE']} AS doc_title,
           s.{PROPERTY_KEYS['SECTION_TITLE']} AS section_title,
           c.{PROPERTY_KEYS['CHUNK_ID']} AS chunk_id,
           e.{PROPERTY_KEYS['ENTITY_NAME']} AS entity_name
    ORDER BY s.{PROPERTY_KEYS['SECTION_INDEX']}, c.{PROPERTY_KEYS['CHUNK_INDEX']}
    """
    print(traverse_query)


def test_validation_functions():
    """Test validation helper functions."""
    print("\n" + "=" * 60)
    print("Testing Validation Functions")
    print("=" * 60)
    
    # Test valid inputs
    assert validate_node_label("Document") == True
    assert validate_node_label("Entity") == True
    assert validate_relationship_type("HAS_SECTION") == True
    assert validate_entity_type("PERSON") == True
    assert validate_media_type("IMAGE") == True
    assert validate_chunk_type("text") == True
    
    # Test invalid inputs
    assert validate_node_label("InvalidNode") == False
    assert validate_relationship_type("INVALID_REL") == False
    assert validate_entity_type("INVALID_TYPE") == False
    
    print("\n[SUCCESS] All validation functions work correctly!")


def main():
    """Run all tests."""
    try:
        test_schema_constants()
        test_validation_functions()
        test_cypher_examples()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed! Schema is ready for use.")
        print("=" * 60)
        print("\n" + SCHEMA_DOCUMENTATION)
        
    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

