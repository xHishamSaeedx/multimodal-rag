"""
Update Elasticsearch index mapping for Phase 2 multimodal support.

This script extends the existing chunks index to support multimodal content:
- Adds chunk_type, embedding_type fields
- Adds table_markdown and image_caption fields for searchable content
- Extends metadata object with image/table-specific fields

The update is backward compatible - existing documents will continue to work.
"""

import os
import sys
from typing import Dict, Any

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import RequestError
except ImportError:
    print("Error: elasticsearch is not installed.")
    print("Install it with: pip install 'elasticsearch>=8.0.0,<9.0.0'")
    sys.exit(1)


def get_multimodal_mapping_update() -> Dict[str, Any]:
    """
    Define the mapping update for Phase 2 multimodal fields.
    
    Returns:
        Dictionary containing the new fields to add to the mapping
    """
    return {
        "properties": {
            # Phase 2: Multimodal type fields
            "chunk_type": {
                "type": "keyword",  # text, table, image, mixed
                "index": True
            },
            "embedding_type": {
                "type": "keyword",  # text, table, image
                "index": True
            },
            # Phase 2: Searchable multimodal content
            "table_markdown": {
                "type": "text",
                "analyzer": "standard",  # Searchable table content
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "image_caption": {
                "type": "text",
                "analyzer": "standard",  # Searchable image captions
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            # Phase 2: Extended metadata with image/table-specific fields
            "metadata": {
                "properties": {
                    # Existing Phase 1 fields (preserved for backward compatibility)
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "tags": {
                        "type": "keyword"
                    },
                    "author": {
                        "type": "keyword"
                    },
                    "version": {
                        "type": "keyword"
                    },
                    "page_number": {
                        "type": "integer"
                    },
                    "section": {
                        "type": "text"
                    },
                    "chunk_index": {
                        "type": "integer"
                    },
                    # Phase 2: Image-specific metadata
                    "image_type": {
                        "type": "keyword"  # diagram, chart, photo, screenshot
                    },
                    # Phase 2: Table-specific metadata
                    "table_headers": {
                        "type": "text",  # Searchable table headers
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "row_count": {
                        "type": "integer"
                    },
                    "col_count": {
                        "type": "integer"
                    }
                }
            }
        }
    }


def update_elasticsearch_mapping(
    elasticsearch_url: str = "http://localhost:9200",
    index_name: str = "chunks",
    dry_run: bool = False,
) -> bool:
    """
    Update the Elasticsearch index mapping to support Phase 2 multimodal content.

    Args:
        elasticsearch_url: Elasticsearch server URL (default: http://localhost:9200)
        index_name: Name of the index to update (default: chunks)
        dry_run: If True, only show what would be updated without making changes

    Returns:
        True if successful, False otherwise
    """
    try:
        # Connect to Elasticsearch
        print(f"Connecting to Elasticsearch at {elasticsearch_url}...")
        
        # Parse URL to ensure proper format
        from urllib.parse import urlparse
        parsed = urlparse(elasticsearch_url)
        
        if not parsed.scheme or parsed.scheme not in ['http', 'https']:
            elasticsearch_url = f"http://{elasticsearch_url}"
            parsed = urlparse(elasticsearch_url)
        
        host = parsed.hostname or "localhost"
        port = parsed.port or 9200
        scheme = "http"
        
        # Create Elasticsearch client
        es = None
        connection_methods = [
            lambda: Elasticsearch(
                f"http://{host}:{port}",
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            ),
            lambda: Elasticsearch(
                elasticsearch_url if elasticsearch_url.startswith('http') else f"http://{elasticsearch_url}",
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            ),
            lambda: Elasticsearch(
                hosts=[{"host": host, "port": port, "scheme": scheme}],
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            ),
        ]
        
        for i, method in enumerate(connection_methods, 1):
            try:
                es = method()
                if es.ping():
                    print(f"[OK] Connected to Elasticsearch!")
                    break
            except Exception:
                if i == len(connection_methods):
                    print("Error: Failed to connect to Elasticsearch")
                    return False
                continue
        
        if es is None or not es.ping():
            print("Error: Could not establish connection to Elasticsearch")
            return False
        
        # Check if index exists
        if not es.indices.exists(index=index_name):
            print(f"Error: Index '{index_name}' does not exist.")
            print(f"Please create it first using: python scripts/init_elasticsearch.py")
            return False
        
        print(f"[OK] Index '{index_name}' exists.")
        
        # Get current mapping
        try:
            current_mapping = es.indices.get_mapping(index=index_name)
            current_properties = current_mapping[index_name]["mappings"].get("properties", {})
            print(f"[OK] Retrieved current mapping.")
        except Exception as e:
            print(f"Error: Failed to retrieve current mapping: {e}")
            return False
        
        # Prepare mapping update
        mapping_update = get_multimodal_mapping_update()
        
        # Check which fields already exist
        new_fields = []
        existing_fields = []
        for field_name in mapping_update["properties"].keys():
            if field_name in current_properties:
                existing_fields.append(field_name)
            else:
                new_fields.append(field_name)
        
        # Check metadata subfields
        if "metadata" in current_properties:
            current_metadata_props = current_properties["metadata"].get("properties", {})
            metadata_new_fields = []
            metadata_existing_fields = []
            
            for meta_field in ["image_type", "table_headers", "row_count", "col_count"]:
                if meta_field in current_metadata_props:
                    metadata_existing_fields.append(meta_field)
                else:
                    metadata_new_fields.append(meta_field)
            
            if metadata_new_fields:
                print(f"\nNew metadata fields to add: {', '.join(metadata_new_fields)}")
            if metadata_existing_fields:
                print(f"Existing metadata fields (will be preserved): {', '.join(metadata_existing_fields)}")
        
        # Summary
        print("\n" + "="*60)
        print("Mapping Update Summary")
        print("="*60)
        if new_fields:
            print(f"\nNew top-level fields to add:")
            for field in new_fields:
                print(f"  ✓ {field}")
        else:
            print("\nNo new top-level fields to add.")
        
        if existing_fields:
            print(f"\nExisting fields (will be preserved):")
            for field in existing_fields:
                print(f"  - {field}")
        
        # Show document count
        try:
            stats = es.indices.stats(index=index_name)
            doc_count = stats['indices'][index_name]['total']['docs']['count']
            print(f"\nCurrent document count: {doc_count}")
            if doc_count > 0:
                print("⚠ Note: Existing documents will continue to work (backward compatible)")
        except Exception as e:
            print(f"Could not retrieve document count: {e}")
        
        if dry_run:
            print("\n" + "="*60)
            print("[DRY RUN] No changes made. Use without --dry-run to apply updates.")
            print("="*60)
            return True
        
        # Apply mapping update
        print("\n" + "="*60)
        print("Applying mapping update...")
        print("="*60)
        
        try:
            # Use put_mapping to add new fields (backward compatible)
            es.indices.put_mapping(
                index=index_name,
                body=mapping_update
            )
            print(f"[OK] Mapping update applied successfully!")
        except RequestError as e:
            error_info = e.info if hasattr(e, 'info') else {}
            error_type = error_info.get('error', {}).get('type', 'unknown')
            
            if error_type == 'illegal_argument_exception':
                print(f"[ERROR] Mapping conflict detected: {e}")
                print("\nThis usually means:")
                print("  1. A field already exists with a different type")
                print("  2. The existing mapping is incompatible with the update")
                print("\nTo resolve:")
                print("  1. Check current mapping: curl http://localhost:9200/chunks/_mapping")
                print("  2. If needed, recreate index: python scripts/init_elasticsearch.py --recreate")
                print("     (WARNING: This will delete all indexed documents)")
                return False
            else:
                print(f"[ERROR] Failed to update mapping: {e}")
                if error_info:
                    print(f"Details: {error_info}")
                return False
        
        # Verify update
        print("\nVerifying updated mapping...")
        try:
            updated_mapping = es.indices.get_mapping(index=index_name)
            updated_properties = updated_mapping[index_name]["mappings"].get("properties", {})
            
            # Check if new fields are present
            all_added = True
            for field_name in new_fields:
                if field_name not in updated_properties:
                    print(f"  ⚠ Warning: Field '{field_name}' not found in updated mapping")
                    all_added = False
                else:
                    print(f"  ✓ Field '{field_name}' added successfully")
            
            if all_added:
                print("\n[OK] All new fields verified in mapping!")
            else:
                print("\n[WARN] Some fields may not have been added correctly.")
            
            # Check metadata subfields
            if "metadata" in updated_properties:
                metadata_props = updated_properties["metadata"].get("properties", {})
                for meta_field in ["image_type", "table_headers", "row_count", "col_count"]:
                    if meta_field in metadata_props:
                        print(f"  ✓ Metadata field 'metadata.{meta_field}' added successfully")
            
        except Exception as e:
            print(f"[WARN] Could not verify mapping update: {e}")
        
        print("\n" + "="*60)
        print("[OK] Mapping update complete!")
        print("="*60)
        print("\nThe index now supports:")
        print("  - chunk_type (keyword): text, table, image, mixed")
        print("  - embedding_type (keyword): text, table, image")
        print("  - table_markdown (text): Searchable table content")
        print("  - image_caption (text): Searchable image captions")
        print("  - Extended metadata: image_type, table_headers, row_count, col_count")
        print("\nExisting documents remain fully functional (backward compatible).")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to update Elasticsearch mapping: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Make sure Elasticsearch is running: docker-compose up -d elasticsearch")
        print(f"  2. Check if Elasticsearch is accessible at {elasticsearch_url}")
        print(f"  3. Verify Elasticsearch health: curl http://localhost:9200/_cluster/health")
        print(f"  4. Check current mapping: curl http://localhost:9200/{index_name}/_mapping")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update Elasticsearch index mapping for Phase 2 multimodal support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update mapping (default: chunks index)
  python scripts/update_elasticsearch_mapping.py

  # Dry run (see what would be updated without making changes)
  python scripts/update_elasticsearch_mapping.py --dry-run

  # Custom Elasticsearch URL
  python scripts/update_elasticsearch_mapping.py --url http://localhost:9200

  # Custom index name
  python scripts/update_elasticsearch_mapping.py --index chunks
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9200",
        help="Elasticsearch server URL (default: http://localhost:9200)",
    )
    parser.add_argument(
        "--index",
        default="chunks",
        help="Index name to update (default: chunks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    
    args = parser.parse_args()
    
    # Allow override from environment variable
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", args.url)
    index_name = os.getenv("ELASTICSEARCH_INDEX", args.index)
    
    print("\n" + "="*60)
    print("Elasticsearch Mapping Update - Phase 2 Multimodal Support")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Elasticsearch URL: {elasticsearch_url}")
    print(f"  - Index name: {index_name}")
    if args.dry_run:
        print(f"  - Mode: DRY RUN (no changes will be made)")
    print()
    
    success = update_elasticsearch_mapping(
        elasticsearch_url=elasticsearch_url,
        index_name=index_name,
        dry_run=args.dry_run,
    )
    
    sys.exit(0 if success else 1)

