"""
Initialize Qdrant vector database with the text_chunks collection.

This script creates the Qdrant collection as specified in the Phase 1 documentation:
- Collection name: text_chunks
- Vector dimensions: 384-768 (configurable based on embedding model)
- Payload fields: chunk_id, document_id, text, metadata
"""

import os
import sys
from typing import Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        CollectionStatus,
    )
except ImportError:
    print("Error: qdrant-client is not installed.")
    print("Install it with: pip install qdrant-client")
    sys.exit(1)


def init_qdrant_collection(
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "text_chunks",
    vector_size: int = 768,  # Default for e5-base-v2/all-mpnet-base-v2 (best quality), can be 384 for all-MiniLM-L6-v2
    recreate: bool = False,
) -> bool:
    """
    Initialize or recreate the Qdrant collection for text chunks.

    Args:
        qdrant_url: Qdrant server URL (default: http://localhost:6333)
        collection_name: Name of the collection (default: text_chunks)
        vector_size: Dimension of the vectors (384 or 768)
        recreate: If True, delete existing collection and create new one

    Returns:
        True if successful, False otherwise
    """
    try:
        # Connect to Qdrant
        print(f"Connecting to Qdrant at {qdrant_url}...")
        client = QdrantClient(url=qdrant_url)

        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)

        if collection_exists:
            if recreate:
                print(f"Deleting existing collection '{collection_name}'...")
                client.delete_collection(collection_name)
                collection_exists = False
            else:
                print(f"Collection '{collection_name}' already exists.")
                # Verify collection configuration
                collection_info = client.get_collection(collection_name)
                print(f"  - Vector size: {collection_info.config.params.vectors.size}")
                print(f"  - Distance metric: {collection_info.config.params.vectors.distance}")
                print(f"  - Status: {collection_info.status}")
                return True

        # Create collection
        if not collection_exists:
            print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,  # Cosine similarity for text embeddings
                ),
            )
            print(f"✓ Collection '{collection_name}' created successfully!")

        # Verify collection was created
        collection_info = client.get_collection(collection_name)
        if collection_info.status == CollectionStatus.GREEN:
            print(f"✓ Collection '{collection_name}' is ready.")
            print(f"  - Vector size: {collection_info.config.params.vectors.size}")
            print(f"  - Distance metric: {collection_info.config.params.vectors.distance}")
            print(f"  - Points count: {collection_info.points_count}")
            return True
        else:
            print(f"⚠ Warning: Collection status is {collection_info.status}")
            return False

    except Exception as e:
        print(f"Error: Failed to initialize Qdrant collection: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Make sure Qdrant is running: docker-compose up -d qdrant")
        print(f"  2. Check if Qdrant is accessible at {qdrant_url}")
        print(f"  3. Verify Qdrant health: curl http://localhost:6333/health")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collection for text chunks"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        default="text_chunks",
        help="Collection name (default: text_chunks)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        choices=[384, 768],
        default=768,
        help="Vector dimension: 768 for e5-base-v2/all-mpnet-base-v2 (best quality, recommended), 384 for all-MiniLM-L6-v2 (faster) (default: 768)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection and create new one",
    )

    args = parser.parse_args()

    # Allow override from environment variable
    qdrant_url = os.getenv("QDRANT_URL", args.url)
    vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", args.vector_size))

    success = init_qdrant_collection(
        qdrant_url=qdrant_url,
        collection_name=args.collection,
        vector_size=vector_size,
        recreate=args.recreate,
    )

    sys.exit(0 if success else 1)

