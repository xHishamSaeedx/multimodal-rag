"""
Test script to verify document ingestion components work correctly.

This tests the structure without requiring a running Neo4j instance.
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent))


def test_document_processor():
    """Test document processor functionality."""
    print("Testing document processor...")
    
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    
    # Create a test text file
    test_content = """
# Introduction

This is the introduction section. It contains some text about the document.

## Section 1: Overview

This section discusses the overview. It mentions important concepts like Machine Learning and Artificial Intelligence.

The document also contains metrics like 5.2% growth and $1M revenue.

## Section 2: Details

This section has more details about the topics discussed.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        # Test text extraction
        text, metadata = processor.extract_text_from_file(temp_path)
        assert len(text) > 0
        print("  [OK] Text extraction works")
        
        # Test section identification
        sections = processor.identify_sections(text)
        assert len(sections) > 0
        print(f"  [OK] Section identification works ({len(sections)} sections found)")
        
        # Test chunking
        chunks = processor.chunk_text(text)
        assert len(chunks) > 0
        print(f"  [OK] Text chunking works ({len(chunks)} chunks created)")
        
        # Test entity extraction
        entities = processor.extract_entities_simple(text)
        print(f"  [OK] Entity extraction works ({len(entities)} entities found)")
        
        # Test full document processing
        document_data = processor.process_document(temp_path)
        assert 'document_id' in document_data
        assert 'sections' in document_data
        assert 'entities' in document_data
        print("  [OK] Full document processing works")
        print(f"      - Document ID: {document_data['document_id']}")
        print(f"      - Sections: {len(document_data['sections'])}")
        print(f"      - Total chunks: {sum(len(s['chunks']) for s in document_data['sections'])}")
        print(f"      - Entities: {len(document_data['entities'])}")
        
        return True
        
    finally:
        Path(temp_path).unlink()


def test_ingestion_structure():
    """Test that ingestion script structure is correct."""
    print("\nTesting ingestion script structure...")
    
    import ingest_document
    
    # Check that main function exists
    assert hasattr(ingest_document, 'ingest_document')
    assert callable(ingest_document.ingest_document)
    print("  [OK] ingest_document function exists")
    
    # Check function signature
    import inspect
    sig = inspect.signature(ingest_document.ingest_document)
    assert 'file_path' in sig.parameters
    print("  [OK] Function signature is correct")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Document Ingestion Components")
    print("=" * 60)
    
    try:
        test_document_processor()
        test_ingestion_structure()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All ingestion components are ready!")
        print("=" * 60)
        print("\nTo ingest a document:")
        print("  1. Make sure Neo4j is running")
        print("  2. Run: python ingest_document.py <file_path>")
        print("\nExample:")
        print("  python ingest_document.py document.pdf")
        
        return 0
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

