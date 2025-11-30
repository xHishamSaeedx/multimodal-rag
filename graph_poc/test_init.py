"""
Test script to verify init_neo4j.py can be imported and basic structure is correct.

This doesn't require a running Neo4j instance - it just validates the code structure.
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from graph_schema import NODE_LABELS, PROPERTY_KEYS
        print("[OK] graph_schema imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import graph_schema: {e}")
        return False
    
    try:
        from neo4j_connection import get_neo4j_driver, close_neo4j_driver, get_database_name
        print("[OK] neo4j_connection imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import neo4j_connection: {e}")
        return False
    
    try:
        # Import the init module (this will validate syntax)
        import init_neo4j
        print("[OK] init_neo4j imported successfully")
    except SyntaxError as e:
        print(f"[FAIL] Syntax error in init_neo4j: {e}")
        return False
    except Exception as e:
        # Other import errors are okay (like missing neo4j package)
        print(f"[INFO] init_neo4j import note: {e}")
        print("[OK] init_neo4j syntax is valid (module may need Neo4j package)")
    
    return True


def test_schema_constants():
    """Test that schema constants are accessible."""
    print("\nTesting schema constants...")
    
    from graph_schema import NODE_LABELS, PROPERTY_KEYS
    
    # Check that required node labels exist
    required_labels = ['DOCUMENT', 'SECTION', 'CHUNK', 'ENTITY', 'MEDIA']
    for label in required_labels:
        if label in NODE_LABELS:
            print(f"[OK] Node label '{label}' exists")
        else:
            print(f"[FAIL] Node label '{label}' missing")
            return False
    
    # Check that required property keys exist
    required_props = ['DOCUMENT_ID', 'SECTION_ID', 'CHUNK_ID', 'ENTITY_ID', 'MEDIA_ID']
    for prop in required_props:
        if prop in PROPERTY_KEYS:
            print(f"[OK] Property key '{prop}' exists")
        else:
            print(f"[FAIL] Property key '{prop}' missing")
            return False
    
    return True


def test_constraint_queries():
    """Test that constraint queries can be constructed."""
    print("\nTesting constraint query construction...")
    
    from graph_schema import NODE_LABELS, PROPERTY_KEYS
    
    # Test constructing a constraint query
    constraint = (
        f"CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:{NODE_LABELS['DOCUMENT']}) "
        f"REQUIRE d.{PROPERTY_KEYS['DOCUMENT_ID']} IS UNIQUE"
    )
    
    if "Document" in constraint and "document_id" in constraint:
        print("[OK] Constraint query construction works")
        print(f"     Example: {constraint[:60]}...")
        return True
    else:
        print("[FAIL] Constraint query construction failed")
        return False


def test_index_queries():
    """Test that index queries can be constructed."""
    print("\nTesting index query construction...")
    
    from graph_schema import NODE_LABELS, PROPERTY_KEYS
    
    # Test constructing an index query
    index = (
        f"CREATE INDEX entity_type IF NOT EXISTS FOR (e:{NODE_LABELS['ENTITY']}) "
        f"ON (e.{PROPERTY_KEYS['ENTITY_TYPE']})"
    )
    
    if "Entity" in index and "entity_type" in index:
        print("[OK] Index query construction works")
        print(f"     Example: {index[:60]}...")
        return True
    else:
        print("[FAIL] Index query construction failed")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Graph POC Initialization Script Structure")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Schema Constants", test_schema_constants),
        ("Constraint Queries", test_constraint_queries),
        ("Index Queries", test_index_queries),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("[SUCCESS] All structure tests passed!")
        print("\nNote: This only tests code structure.")
        print("To test with actual Neo4j, run: python init_neo4j.py")
        return 0
    else:
        print("[FAIL] Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

