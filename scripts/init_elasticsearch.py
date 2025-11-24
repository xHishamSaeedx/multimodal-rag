"""
Initialize Elasticsearch index for BM25 sparse search on chunks.

This script creates the Elasticsearch index as specified in the Phase 1 documentation:
- Index name: chunks
- BM25 similarity scoring (default)
- Indexed fields: chunk_text, document metadata, filename, timestamps
"""

import os
import sys
from typing import Dict, Any

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import RequestError
except ImportError:
    print("Error: elasticsearch is not installed.")
    print("Install it with: pip install elasticsearch")
    sys.exit(1)


def get_index_mapping() -> Dict[str, Any]:
    """
    Define the Elasticsearch index mapping for text chunks.
    
    Returns:
        Dictionary containing index settings and mappings
    """
    return {
        "settings": {
            "number_of_shards": 1,  # Single shard for single-node setup
            "number_of_replicas": 0,  # No replicas for single-node setup
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard"  # Standard analyzer for BM25
                    }
                }
            },
            # BM25 similarity is default in Elasticsearch 8.x, but we can be explicit
            "similarity": {
                "default": {
                    "type": "BM25",
                    "k1": 1.2,
                    "b": 0.75
                }
            }
        },
        "mappings": {
            "properties": {
                # Primary identifier fields
                "chunk_id": {
                    "type": "keyword",  # Exact match, not analyzed
                    "index": True
                },
                "document_id": {
                    "type": "keyword",
                    "index": True
                },
                # Full-text search field - this is what BM25 will search
                "chunk_text": {
                    "type": "text",
                    "analyzer": "standard",  # Standard analyzer for tokenization
                    "fields": {
                        "keyword": {
                            "type": "keyword",  # For exact matches
                            "ignore_above": 256
                        }
                    }
                },
                # Document metadata fields
                "filename": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",  # For exact filename matches
                            "index": True
                        }
                    }
                },
                "document_type": {
                    "type": "keyword",
                    "index": True
                },
                "source_path": {
                    "type": "keyword",
                    "index": True
                },
                # Metadata object - flexible JSONB-like structure
                "metadata": {
                    "type": "object",
                    "enabled": True,  # Store full object for filtering
                    "properties": {
                        "title": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "tags": {
                            "type": "keyword"  # Array of tags
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
                        }
                    }
                },
                # Timestamps
                "created_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis"
                },
                "updated_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis"
                }
            }
        }
    }


def init_elasticsearch_index(
    elasticsearch_url: str = "http://localhost:9200",
    index_name: str = "chunks",
    recreate: bool = False,
) -> bool:
    """
    Initialize or recreate the Elasticsearch index for chunks.

    Args:
        elasticsearch_url: Elasticsearch server URL (default: http://localhost:9200)
        index_name: Name of the index (default: chunks)
        recreate: If True, delete existing index and create new one

    Returns:
        True if successful, False otherwise
    """
    try:
        # Connect to Elasticsearch
        print(f"Connecting to Elasticsearch at {elasticsearch_url}...")
        
        # For Elasticsearch 8.x without security, ensure we use HTTP and proper connection format
        # Parse URL to ensure proper format
        from urllib.parse import urlparse
        parsed = urlparse(elasticsearch_url)
        
        # Ensure we're using HTTP (not HTTPS) for Elasticsearch without security
        if not parsed.scheme or parsed.scheme not in ['http', 'https']:
            elasticsearch_url = f"http://{elasticsearch_url}"
            parsed = urlparse(elasticsearch_url)
        
        host = parsed.hostname or "localhost"
        port = parsed.port or 9200
        scheme = "http"  # Force HTTP since security is disabled
        
        # Create Elasticsearch client - try simplest format first
        es = None
        connection_methods = [
            # Method 1: Simple URL string (most compatible)
            lambda: Elasticsearch(
                f"http://{host}:{port}",
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            ),
            # Method 2: URL with full path
            lambda: Elasticsearch(
                elasticsearch_url if elasticsearch_url.startswith('http') else f"http://{elasticsearch_url}",
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            ),
            # Method 3: Explicit hosts list with scheme (required for client 9.x)
            lambda: Elasticsearch(
                hosts=[{"host": host, "port": port, "scheme": scheme}],
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            ),
        ]
        
        last_error = None
        connection_errors = []
        for i, method in enumerate(connection_methods, 1):
            try:
                print(f"Trying connection method {i}...", end=" ")
                es = method()
                print("client created, testing connection...")
                
                # Test connection - try ping() first, if it fails try info() as fallback
                try:
                    ping_result = es.ping()
                    if ping_result:
                        print(f"[OK] Connected using method {i}!")
                        break
                    else:
                        print(f"ping() returned False, trying info()...")
                        # ping() returned False, try info() as alternative test
                        try:
                            cluster_info = es.info()
                            if cluster_info and 'cluster_name' in cluster_info:
                                print(f"[OK] Connected using method {i} (via info() check)!")
                                break
                        except Exception as info_err:
                            print(f"info() also failed: {info_err}")
                            connection_errors.append(f"Method {i}: ping()=False, info() failed: {info_err}")
                            last_error = Exception(f"ping() returned False and info() failed: {info_err}")
                            continue
                        last_error = Exception(f"ping() returned False with method {i}")
                        connection_errors.append(f"Method {i}: ping() returned False")
                        continue
                except Exception as ping_err:
                    # ping() raised exception, try info() as alternative
                    print(f"ping() failed: {ping_err}, trying info()...")
                    try:
                        cluster_info = es.info()
                        if cluster_info and 'cluster_name' in cluster_info:
                            print(f"[OK] Connected using method {i} (via info() check)!")
                            break
                    except Exception as info_err:
                        print(f"info() also failed: {info_err}")
                        connection_errors.append(f"Method {i}: ping() error: {ping_err}, info() error: {info_err}")
                        last_error = info_err
                        continue
                        
            except Exception as e:
                print(f"Failed: {e}")
                connection_errors.append(f"Method {i}: {type(e).__name__}: {e}")
                last_error = e
                if i < len(connection_methods):
                    continue
                else:
                    # Last method failed
                    print(f"\nError: All connection methods failed.")
                    print(f"\nDetailed errors:")
                    for err in connection_errors:
                        print(f"  - {err}")
                    if last_error:
                        print(f"\nLast error details:")
                        print(f"  - Type: {type(last_error).__name__}")
                        print(f"  - Message: {str(last_error)}")
                    print(f"\nDebug info:")
                    print(f"  - URL: {elasticsearch_url}")
                    print(f"  - Parsed: {scheme}://{host}:{port}")
                    print(f"\nVerify Elasticsearch is accessible:")
                    print(f"  curl http://{host}:{port}/_cluster/health")
                    print(f"  Or visit: http://{host}:{port}")
                    print(f"\nIf curl works but Python doesn't, you may need to:")
                    print(f"  pip install 'elasticsearch>=8.0.0,<9.0.0'  # Use version 8.x to match server")
                    return False
        
        if es is None:
            print("Error: Failed to establish connection to Elasticsearch after all methods.")
            return False
        
        # Final verification
        try:
            if not es.ping():
                print("Warning: Connection established but ping() returns False")
            else:
                print("[OK] Connection verified with ping()")
        except Exception as final_err:
            print(f"Warning: Could not verify with ping(): {final_err}")
        
        print("[OK] Connected to Elasticsearch successfully!")
        
        # Check if index exists
        index_exists = es.indices.exists(index=index_name)
        
        if index_exists:
            if recreate:
                print(f"Deleting existing index '{index_name}'...")
                es.indices.delete(index=index_name)
                index_exists = False
            else:
                print(f"Index '{index_name}' already exists.")
                # Get index information
                index_info = es.indices.get(index=index_name)
                index_stats = es.indices.stats(index=index_name)
                
                print(f"  - Index status: {es.cluster.health(index=index_name)['status']}")
                print(f"  - Document count: {index_stats['indices'][index_name]['total']['docs']['count']}")
                print(f"  - Shards: {index_info[index_name]['settings']['index']['number_of_shards']}")
                return True
        
        # Create index
        if not index_exists:
            print(f"Creating index '{index_name}' with BM25 similarity...")
            index_config = get_index_mapping()
            
            es.indices.create(
                index=index_name,
                body=index_config
            )
            print(f"[OK] Index '{index_name}' created successfully!")
        
        # Verify index was created and is ready
        if es.indices.exists(index=index_name):
            health = es.cluster.health(index=index_name, wait_for_status="yellow", timeout="30s")
            if health["status"] in ["green", "yellow"]:
                print(f"[OK] Index '{index_name}' is ready.")
                print(f"  - Status: {health['status']}")
                print(f"  - Shards: {health['active_shards']}")
                
                # Show index settings
                index_info = es.indices.get(index=index_name)
                settings = index_info[index_name]['settings']['index']
                print(f"  - Similarity: {settings.get('similarity', {}).get('default', {}).get('type', 'BM25 (default)')}")
                print(f"  - Analyzer: standard (for BM25)")
                return True
            else:
                print(f"⚠ Warning: Index status is {health['status']}")
                return False
        else:
            print("Error: Index was not created.")
            return False
    
    except RequestError as e:
        print(f"Error: Elasticsearch request failed: {e}")
        if hasattr(e, 'info'):
            print(f"Details: {e.info}")
        return False
    except Exception as e:
        print(f"Error: Failed to initialize Elasticsearch index: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Make sure Elasticsearch is running: docker-compose up -d elasticsearch")
        print(f"  2. Check if Elasticsearch is accessible at {elasticsearch_url}")
        print(f"  3. Verify Elasticsearch health: curl http://localhost:9200/_cluster/health")
        print(f"  4. Wait for Elasticsearch to fully start (may take 30-60 seconds on first run)")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Initialize Elasticsearch index for text chunks (BM25 sparse search)"
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
        "--recreate",
        action="store_true",
        help="Delete existing index and create new one",
    )
    
    args = parser.parse_args()
    
    # Allow override from environment variable
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", args.url)
    index_name = os.getenv("ELASTICSEARCH_INDEX", args.index)
    
    success = init_elasticsearch_index(
        elasticsearch_url=elasticsearch_url,
        index_name=index_name,
        recreate=args.recreate,
    )
    
    sys.exit(0 if success else 1)

