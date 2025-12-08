"""
Clean Elasticsearch index - Remove documents/chunks from the chunks index.

This script provides several options to clean out Elasticsearch:
- Delete all documents (nuclear option)
- Delete by document_id
- Delete by filename
- Delete by source_path
- Show index statistics before/after

Useful for cleaning up partially processed documents during testing.
"""

import os
import sys
from typing import Dict, Any, Optional
from uuid import UUID

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import RequestError, NotFoundError
except ImportError:
    print("Error: elasticsearch is not installed.")
    print("Install it with: pip install 'elasticsearch>=8.0.0,<9.0.0'")
    sys.exit(1)


def get_elasticsearch_client(elasticsearch_url: str = "http://localhost:9200") -> Elasticsearch:
    """
    Create and return an Elasticsearch client.
    
    Args:
        elasticsearch_url: Elasticsearch server URL
        
    Returns:
        Elasticsearch client instance
    """
    from urllib.parse import urlparse
    
    parsed = urlparse(elasticsearch_url)
    if not parsed.scheme or parsed.scheme not in ['http', 'https']:
        elasticsearch_url = f"http://{elasticsearch_url}"
        parsed = urlparse(elasticsearch_url)
    
    host = parsed.hostname or "localhost"
    port = parsed.port or 9200
    scheme = "http"
    
    # Try different connection methods
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
            # Test connection
            if es.ping() or (es.info() and 'cluster_name' in es.info()):
                return es
        except Exception:
            if i == len(connection_methods):
                raise
    
    raise Exception("Failed to connect to Elasticsearch")


def get_index_stats(es: Elasticsearch, index_name: str) -> Dict[str, Any]:
    """
    Get statistics about the index.
    
    Args:
        es: Elasticsearch client
        index_name: Index name
        
    Returns:
        Dictionary with index statistics
    """
    try:
        if not es.indices.exists(index=index_name):
            return {"exists": False}
        
        stats = es.indices.stats(index=index_name)
        index_stats = stats['indices'][index_name]['total']
        
        # Get document count
        doc_count = index_stats['docs']['count']
        
        # Get index info
        index_info = es.indices.get(index=index_name)
        settings = index_info[index_name]['settings']['index']
        
        return {
            "exists": True,
            "document_count": doc_count,
            "size_bytes": index_stats['store']['size_in_bytes'],
            "shards": settings.get('number_of_shards', 'unknown'),
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def list_documents(
    es: Elasticsearch,
    index_name: str,
    limit: int = 20,
    filter_by: Optional[Dict[str, str]] = None,
) -> None:
    """
    List documents in the index.
    
    Args:
        es: Elasticsearch client
        index_name: Index name
        limit: Maximum number of documents to show
        filter_by: Optional filter (e.g., {"document_id": "...", "filename": "..."})
    """
    try:
        query_body = {
            "size": limit,
            "_source": ["chunk_id", "document_id", "filename", "document_type", "source_path", "created_at"],
            "sort": [{"created_at": {"order": "desc"}}],
        }
        
        if filter_by:
            filter_clauses = []
            for field, value in filter_by.items():
                if field == "document_id":
                    filter_clauses.append({"term": {"document_id": str(value)}})
                elif field == "filename":
                    filter_clauses.append({"term": {"filename.keyword": value}})
                elif field == "source_path":
                    filter_clauses.append({"term": {"source_path": value}})
                elif field == "document_type":
                    filter_clauses.append({"term": {"document_type": value}})
            
            if filter_clauses:
                query_body["query"] = {
                    "bool": {
                        "filter": filter_clauses
                    }
                }
        else:
            query_body["query"] = {"match_all": {}}
        
        response = es.search(index=index_name, body=query_body, request_timeout=30)
        
        hits = response.get("hits", {}).get("hits", [])
        total = response.get("hits", {}).get("total", {})
        total_count = total.get("value", 0) if isinstance(total, dict) else total
        
        print(f"\nFound {total_count} document(s) in index '{index_name}'")
        if total_count > limit:
            print(f"Showing first {limit} documents:\n")
        else:
            print(f"Showing all documents:\n")
        
        if not hits:
            print("  (No documents found)")
            return
        
        print(f"{'Chunk ID':<40} {'Document ID':<40} {'Filename':<30} {'Type':<10} {'Created At'}")
        print("-" * 140)
        
        for hit in hits:
            source = hit["_source"]
            chunk_id = source.get("chunk_id", "N/A")[:38]
            doc_id = source.get("document_id", "N/A")[:38]
            filename = source.get("filename", "N/A")[:28]
            doc_type = source.get("document_type", "N/A")[:8]
            created_at = source.get("created_at", "N/A")[:19] if source.get("created_at") else "N/A"
            
            print(f"{chunk_id:<40} {doc_id:<40} {filename:<30} {doc_type:<10} {created_at}")
        
    except Exception as e:
        print(f"Error listing documents: {e}")


def delete_all_documents(es: Elasticsearch, index_name: str, confirm: bool = False) -> bool:
    """
    Delete all documents from the index.
    
    Args:
        es: Elasticsearch client
        index_name: Index name
        confirm: If False, will prompt for confirmation
        
    Returns:
        True if successful
    """
    if not confirm:
        print(f"\n⚠️  WARNING: This will delete ALL documents from index '{index_name}'")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return False
    
    try:
        # Delete all documents using delete_by_query
        query_body = {
            "query": {
                "match_all": {}
            }
        }
        
        print(f"Deleting all documents from index '{index_name}'...")
        response = es.delete_by_query(
            index=index_name,
            body=query_body,
            request_timeout=60,
            refresh=True,  # Refresh index after deletion
        )
        
        deleted_count = response.get("deleted", 0)
        print(f"✅ Successfully deleted {deleted_count} document(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting all documents: {e}")
        return False


def delete_by_document_id(es: Elasticsearch, index_name: str, document_id: str) -> bool:
    """
    Delete all chunks for a specific document.
    
    Args:
        es: Elasticsearch client
        index_name: Index name
        document_id: Document UUID (string)
        
    Returns:
        True if successful
    """
    try:
        # Validate UUID format
        try:
            UUID(document_id)
        except ValueError:
            print(f"❌ Error: '{document_id}' is not a valid UUID")
            return False
        
        query_body = {
            "query": {
                "term": {
                    "document_id": str(document_id)
                }
            }
        }
        
        print(f"Deleting chunks for document_id: {document_id}...")
        response = es.delete_by_query(
            index=index_name,
            body=query_body,
            request_timeout=60,
            refresh=True,
        )
        
        deleted_count = response.get("deleted", 0)
        print(f"✅ Successfully deleted {deleted_count} chunk(s) for document {document_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting by document_id: {e}")
        return False


def delete_by_filename(es: Elasticsearch, index_name: str, filename: str) -> bool:
    """
    Delete all chunks for documents with a specific filename.
    
    Args:
        es: Elasticsearch client
        index_name: Index name
        filename: Filename to match
        
    Returns:
        True if successful
    """
    try:
        query_body = {
            "query": {
                "term": {
                    "filename.keyword": filename
                }
            }
        }
        
        print(f"Deleting chunks for filename: {filename}...")
        response = es.delete_by_query(
            index=index_name,
            body=query_body,
            request_timeout=60,
            refresh=True,
        )
        
        deleted_count = response.get("deleted", 0)
        print(f"✅ Successfully deleted {deleted_count} chunk(s) for filename '{filename}'")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting by filename: {e}")
        return False


def delete_by_source_path(es: Elasticsearch, index_name: str, source_path: str) -> bool:
    """
    Delete all chunks for documents with a specific source_path.
    
    Args:
        es: Elasticsearch client
        index_name: Index name
        source_path: Source path to match
        
    Returns:
        True if successful
    """
    try:
        query_body = {
            "query": {
                "term": {
                    "source_path": source_path
                }
            }
        }
        
        print(f"Deleting chunks for source_path: {source_path}...")
        response = es.delete_by_query(
            index=index_name,
            body=query_body,
            request_timeout=60,
            refresh=True,
        )
        
        deleted_count = response.get("deleted", 0)
        print(f"✅ Successfully deleted {deleted_count} chunk(s) for source_path '{source_path}'")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting by source_path: {e}")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean Elasticsearch index - Remove documents/chunks from the chunks index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show index statistics
  python scripts/clean_elasticsearch.py --stats
  
  # List all documents (first 20)
  python scripts/clean_elasticsearch.py --list
  
  # Delete all documents (with confirmation)
  python scripts/clean_elasticsearch.py --delete-all
  
  # Delete by document_id
  python scripts/clean_elasticsearch.py --delete-by-doc-id <uuid>
  
  # Delete by filename
  python scripts/clean_elasticsearch.py --delete-by-filename "document.pdf"
  
  # Delete by source_path
  python scripts/clean_elasticsearch.py --delete-by-path "path/to/document.pdf"
        """
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:9200",
        help="Elasticsearch server URL (default: http://localhost:9200)",
    )
    parser.add_argument(
        "--index",
        default="chunks",
        help="Index name (default: chunks)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List documents in the index (first 20)",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=20,
        help="Limit for --list option (default: 20)",
    )
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete all documents from the index (requires confirmation)",
    )
    parser.add_argument(
        "--delete-all-yes",
        action="store_true",
        help="Delete all documents without confirmation (use with caution!)",
    )
    parser.add_argument(
        "--delete-by-doc-id",
        type=str,
        help="Delete all chunks for a specific document_id (UUID)",
    )
    parser.add_argument(
        "--delete-by-filename",
        type=str,
        help="Delete all chunks for documents with a specific filename",
    )
    parser.add_argument(
        "--delete-by-path",
        type=str,
        help="Delete all chunks for documents with a specific source_path",
    )
    
    args = parser.parse_args()
    
    # Allow override from environment variables
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", args.url)
    index_name = os.getenv("ELASTICSEARCH_INDEX", args.index)
    
    try:
        # Connect to Elasticsearch
        print(f"Connecting to Elasticsearch at {elasticsearch_url}...")
        es = get_elasticsearch_client(elasticsearch_url)
        print("✅ Connected to Elasticsearch")
        
        # Check if index exists
        if not es.indices.exists(index=index_name):
            print(f"❌ Error: Index '{index_name}' does not exist.")
            print(f"   Run: python scripts/init_elasticsearch.py")
            sys.exit(1)
        
        # Show stats before operations
        print(f"\n📊 Index Statistics (before):")
        stats_before = get_index_stats(es, index_name)
        if stats_before.get("exists"):
            print(f"  - Document count: {stats_before.get('document_count', 0):,}")
            print(f"  - Size: {stats_before.get('size_bytes', 0):,} bytes")
            print(f"  - Shards: {stats_before.get('shards', 'unknown')}")
        else:
            print(f"  - Index does not exist or error: {stats_before.get('error', 'unknown')}")
        
        # Perform requested operations
        if args.stats:
            # Stats already shown above
            pass
        
        if args.list:
            list_documents(es, index_name, limit=args.list_limit)
        
        if args.delete_all or args.delete_all_yes:
            success = delete_all_documents(es, index_name, confirm=args.delete_all_yes)
            if not success:
                sys.exit(1)
        
        if args.delete_by_doc_id:
            success = delete_by_document_id(es, index_name, args.delete_by_doc_id)
            if not success:
                sys.exit(1)
        
        if args.delete_by_filename:
            success = delete_by_filename(es, index_name, args.delete_by_filename)
            if not success:
                sys.exit(1)
        
        if args.delete_by_path:
            success = delete_by_source_path(es, index_name, args.delete_by_path)
            if not success:
                sys.exit(1)
        
        # Show stats after operations (if any delete operations were performed)
        if args.delete_all or args.delete_all_yes or args.delete_by_doc_id or args.delete_by_filename or args.delete_by_path:
            print(f"\n📊 Index Statistics (after):")
            stats_after = get_index_stats(es, index_name)
            if stats_after.get("exists"):
                print(f"  - Document count: {stats_after.get('document_count', 0):,}")
                print(f"  - Size: {stats_after.get('size_bytes', 0):,} bytes")
                deleted = stats_before.get('document_count', 0) - stats_after.get('document_count', 0)
                if deleted > 0:
                    print(f"  - Deleted: {deleted:,} document(s)")
        
        print("\n✅ Done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

