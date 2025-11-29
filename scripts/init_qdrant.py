"""
Initialize Qdrant vector database collections.

Phase 1 (text_chunks):
- Collection name: text_chunks
- Vector dimensions: 384-768 (configurable based on embedding model)
- Payload fields: chunk_id, document_id, text, metadata

Phase 2 (multimodal):
- Collection name: table_chunks
- Vector dimensions: 768 (matching text embeddings)
- Payload fields: chunk_id, document_id, table_data, table_markdown, metadata

- Collection name: image_chunks
- Vector dimensions: 512 (CLIP base), 768 (SigLIP base), or 1024 (SigLIP large)
- Payload fields: chunk_id, document_id, image_path, caption, image_type, metadata
"""

import os
import sys
from pathlib import Path
from typing import Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        CollectionStatus,
        HnswConfigDiff,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
    )
except ImportError:
    print("Error: qdrant-client is not installed.")
    print("Install it with: pip install qdrant-client")
    sys.exit(1)

# Try to import backend config for defaults
try:
    # Add backend directory to path
    backend_dir = Path(__file__).parent.parent / "backend"
    sys.path.insert(0, str(backend_dir))
    from app.core.config import settings
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    settings = None


def init_qdrant_collection(
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "text_chunks",
    vector_size: int = 768,  # Default for e5-base-v2/all-mpnet-base-v2 (best quality), can be 384 for all-MiniLM-L6-v2, 1024 for SigLIP large
    recreate: bool = False,
) -> bool:
    """
    Initialize or recreate a Qdrant collection.

    Args:
        qdrant_url: Qdrant server URL (default: http://localhost:6333)
        collection_name: Name of the collection (default: text_chunks)
        vector_size: Dimension of the vectors (384, 512, 768, or 1024)
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
            
            # Optimize HNSW parameters for speed (especially for image_chunks)
            # Lower m and ef_construct = faster search and indexing
            # For image_chunks, use aggressively optimized speed settings + scalar quantization
            hnsw_config = None
            quantization_config = None
            
            if collection_name == "image_chunks":
                # Aggressively optimized for speed: very low m and ef_construct for fastest image search
                hnsw_config = HnswConfigDiff(
                    m=4,              # Very low m (default 16) = minimal connections = fastest traversal
                    ef_construct=64,  # Very low ef_construct (default 200) = fastest index building
                )
                
                # Enable scalar quantization for image_chunks (20-40ms speed improvement)
                # Converts float32 vectors to int8, reducing memory by 75% and speeding up distance calculations
                quantization_config = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,  # 8-bit integers (vs 32-bit floats)
                        quantile=0.99,         # Exclude extreme 1% values for better accuracy
                        always_ram=True,       # Keep quantized vectors in RAM for fastest access
                    )
                )
                print(f"  - HNSW config: m=4, ef_construct=64 (ultra-fast, target <50ms)")
                print(f"  - Scalar quantization: INT8 (75% memory reduction, 20-40ms faster)")
            else:
                # Balanced settings for text/table chunks (still faster than defaults)
                hnsw_config = HnswConfigDiff(
                    m=12,             # Moderate m for balance between speed and accuracy
                    ef_construct=128, # Moderate ef_construct
                )
                print(f"  - HNSW config: m=12, ef_construct=128 (balanced)")
            
            # Build collection creation parameters
            collection_params = {
                "collection_name": collection_name,
                "vectors_config": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,  # Cosine similarity for embeddings
                    hnsw_config=hnsw_config,
                ),
            }
            
            # Add quantization config for image_chunks
            if quantization_config is not None:
                collection_params["quantization_config"] = quantization_config
            
            client.create_collection(**collection_params)
            print(f"[OK] Collection '{collection_name}' created successfully!")

        # Verify collection was created
        collection_info = client.get_collection(collection_name)
        if collection_info.status == CollectionStatus.GREEN:
            print(f"[OK] Collection '{collection_name}' is ready.")
            print(f"  - Vector size: {collection_info.config.params.vectors.size}")
            print(f"  - Distance metric: {collection_info.config.params.vectors.distance}")
            print(f"  - Points count: {collection_info.points_count}")
            return True
        else:
            print(f"[WARN] Warning: Collection status is {collection_info.status}")
            return False

    except Exception as e:
        print(f"Error: Failed to initialize Qdrant collection: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Make sure Qdrant is running: docker-compose up -d qdrant")
        print(f"  2. Check if Qdrant is accessible at {qdrant_url}")
        print(f"  3. Verify Qdrant health: curl http://localhost:6333/health")
        return False


def init_multimodal_collections(
    qdrant_url: str = "http://localhost:6333",
    image_vector_size: int = 1024,  # 512 for CLIP base, 768 for SigLIP base, 1024 for SigLIP large (default)
    recreate: bool = False,
) -> bool:
    """
    Initialize Phase 2 multimodal collections: table_chunks and image_chunks.

    Args:
        qdrant_url: Qdrant server URL (default: http://localhost:6333)
        image_vector_size: Dimension for image embeddings (512 for CLIP base, 768 for SigLIP base, 1024 for SigLIP large)
        recreate: If True, delete existing collections and create new ones

    Returns:
        True if all collections were created successfully, False otherwise
    """
    print("\n" + "="*60)
    print("Initializing Phase 2 Multimodal Collections")
    print("="*60)

    # Initialize table_chunks collection
    print("\n[1/2] Setting up table_chunks collection...")
    table_success = init_qdrant_collection(
        qdrant_url=qdrant_url,
        collection_name="table_chunks",
        vector_size=768,  # Matching text embeddings
        recreate=recreate,
    )

    if not table_success:
        print("[ERROR] Failed to create table_chunks collection")
        return False

    # Initialize image_chunks collection
    print("\n[2/2] Setting up image_chunks collection...")
    image_success = init_qdrant_collection(
        qdrant_url=qdrant_url,
        collection_name="image_chunks",
        vector_size=image_vector_size,
        recreate=recreate,
    )

    if not image_success:
        print("[ERROR] Failed to create image_chunks collection")
        return False

    # Summary
    print("\n" + "="*60)
    print("[OK] All multimodal collections initialized successfully!")
    print("="*60)
    print("\nCollections created:")
    print("  - table_chunks (768 dimensions)")
    print("    Payload: chunk_id, document_id, table_data, table_markdown, metadata")
    print(f"  - image_chunks ({image_vector_size} dimensions)")
    print("    Payload: chunk_id, document_id, image_path, caption, image_type, metadata")
    print("\n" + "="*60)

    return True


def init_all_collections(
    qdrant_url: str = "http://localhost:6333",
    text_vector_size: int = 768,
    image_vector_size: int = 1024,
    recreate: bool = False,
) -> bool:
    """
    Initialize all collections (Phase 1 + Phase 2).

    Args:
        qdrant_url: Qdrant server URL (default: http://localhost:6333)
        text_vector_size: Dimension for text embeddings (384 or 768)
        image_vector_size: Dimension for image embeddings (512 for CLIP base, 768 for SigLIP base, 1024 for SigLIP large)
        recreate: If True, delete existing collections and create new ones

    Returns:
        True if all collections were created successfully, False otherwise
    """
    print("\n" + "="*60)
    print("Initializing All Collections (Phase 1 + Phase 2)")
    print("="*60)

    # Initialize text_chunks (Phase 1)
    print("\n[Phase 1] Setting up text_chunks collection...")
    text_success = init_qdrant_collection(
        qdrant_url=qdrant_url,
        collection_name="text_chunks",
        vector_size=text_vector_size,
        recreate=recreate,
    )

    if not text_success:
        print("[ERROR] Failed to create text_chunks collection")
        return False

    # Initialize multimodal collections (Phase 2)
    multimodal_success = init_multimodal_collections(
        qdrant_url=qdrant_url,
        image_vector_size=image_vector_size,
        recreate=recreate,
    )

    if not multimodal_success:
        return False

    print("\n[OK] All collections (Phase 1 + Phase 2) initialized successfully!")
    return True


def verify_collections(
    qdrant_url: str = "http://localhost:6333",
) -> bool:
    """
    Verify that all collections exist and have correct configurations.

    Args:
        qdrant_url: Qdrant server URL (default: http://localhost:6333)

    Returns:
        True if all collections are properly configured, False otherwise
    """
    try:
        print("\n" + "="*60)
        print("Verifying Qdrant Collections")
        print("="*60)

        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        # Expected collections
        expected_collections = {
            "text_chunks": {"vector_size": 768, "description": "Phase 1: Text chunks"},
            "table_chunks": {"vector_size": 768, "description": "Phase 2: Table chunks"},
            "image_chunks": {"vector_size": [512, 768, 1024], "description": "Phase 2: Image chunks"},
        }

        all_valid = True

        for collection_name, expected in expected_collections.items():
            print(f"\n[{collection_name}]")
            if collection_name not in collection_names:
                print(f"  ❌ Collection does not exist")
                all_valid = False
                continue

            try:
                collection_info = client.get_collection(collection_name)
                actual_size = collection_info.config.params.vectors.size
                expected_size = expected["vector_size"]

                # For image_chunks, accept 512, 768, or 1024
                if collection_name == "image_chunks":
                    if actual_size not in expected_size:
                        print(f"  ❌ Vector size mismatch: expected {expected_size}, got {actual_size}")
                        all_valid = False
                    else:
                        print(f"  ✓ Collection exists")
                        print(f"    - Vector size: {actual_size} (valid)")
                else:
                    if actual_size != expected_size:
                        print(f"  ❌ Vector size mismatch: expected {expected_size}, got {actual_size}")
                        all_valid = False
                    else:
                        print(f"  ✓ Collection exists")
                        print(f"    - Vector size: {actual_size} (correct)")

                print(f"    - Distance metric: {collection_info.config.params.vectors.distance}")
                print(f"    - Status: {collection_info.status}")
                print(f"    - Points count: {collection_info.points_count}")
                print(f"    - Description: {expected['description']}")

            except Exception as e:
                print(f"  ❌ Error checking collection: {e}")
                all_valid = False

        print("\n" + "="*60)
        if all_valid:
            print("[OK] All collections are properly configured!")
        else:
            print("[ERROR] Some collections have issues. Please review above.")
        print("="*60)

        return all_valid

    except Exception as e:
        print(f"\n[ERROR] Failed to verify collections: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Make sure Qdrant is running: docker-compose up -d qdrant")
        print(f"  2. Check if Qdrant is accessible at {qdrant_url}")
        print(f"  3. Verify Qdrant health: curl http://localhost:6333/health")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collections for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create text_chunks collection (Phase 1)
  python scripts/init_qdrant.py --collection text_chunks

  # Create table_chunks collection (Phase 2)
  python scripts/init_qdrant.py --collection table_chunks --vector-size 768

  # Create image_chunks collection (Phase 2, CLIP base)
  python scripts/init_qdrant.py --collection image_chunks --vector-size 512

  # Create all Phase 2 multimodal collections
  python scripts/init_qdrant.py --multimodal

  # Create all collections (Phase 1 + Phase 2)
  python scripts/init_qdrant.py --all

  # Verify all collections are properly configured
  python scripts/init_qdrant.py --verify

  # Recreate existing collection
  python scripts/init_qdrant.py --collection text_chunks --recreate
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        choices=["text_chunks", "table_chunks", "image_chunks"],
        help="Collection name to create (default: text_chunks). Use --multimodal or --all for Phase 2 collections.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        choices=[384, 512, 768, 1024],
        help="Vector dimension: 384 for all-MiniLM-L6-v2, 768 for e5-base-v2/all-mpnet-base-v2, 512 for CLIP base, 1024 for SigLIP large (default: 768 for text/table, 1024 for image)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection(s) and create new one(s)",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Create all Phase 2 multimodal collections (table_chunks, image_chunks)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create all collections (Phase 1 + Phase 2: text_chunks, table_chunks, image_chunks)",
    )
    parser.add_argument(
        "--image-vector-size",
        type=int,
        choices=[512, 768, 1024],
        default=1024,
        help="Vector dimension for image_chunks: 512 for CLIP base, 768 for SigLIP base, 1024 for SigLIP large (default: 1024)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing collections configuration (no changes made)",
    )

    args = parser.parse_args()

    # Try to get defaults from backend config if available
    if HAS_CONFIG and settings:
        config_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        config_collection = settings.qdrant_collection_name
        config_vector_size = settings.qdrant_vector_size
        print(f"[OK] Loaded defaults from backend settings")
        print(f"  - Qdrant URL: {config_url}")
        print(f"  - Collection: {config_collection}")
        print(f"  - Vector size: {config_vector_size}")
    else:
        config_url = None
        config_collection = None
        config_vector_size = None

    # Allow override from environment variable or command line args
    # Priority: env var > cmd arg > config > default
    qdrant_url = os.getenv("QDRANT_URL") or args.url or config_url or "http://localhost:6333"

    # Handle verification mode
    if args.verify:
        success = verify_collections(qdrant_url=qdrant_url)
        sys.exit(0 if success else 1)

    # Handle different modes
    if args.all:
        # Create all collections (Phase 1 + Phase 2)
        text_vector_size = args.vector_size or config_vector_size or 768
        image_vector_size = args.image_vector_size or 1024
        success = init_all_collections(
            qdrant_url=qdrant_url,
            text_vector_size=text_vector_size,
            image_vector_size=image_vector_size,
            recreate=args.recreate,
        )
    elif args.multimodal:
        # Create only Phase 2 multimodal collections
        image_vector_size = args.image_vector_size or 1024
        success = init_multimodal_collections(
            qdrant_url=qdrant_url,
            image_vector_size=image_vector_size,
            recreate=args.recreate,
        )
    else:
        # Create single collection (backward compatible)
        collection_name = args.collection or config_collection or "text_chunks"
        
        # Determine vector size based on collection type
        if collection_name == "image_chunks":
            default_vector_size = args.vector_size or args.image_vector_size or 1024
        elif collection_name == "table_chunks":
            default_vector_size = args.vector_size or 768
        else:  # text_chunks
            default_vector_size = args.vector_size or config_vector_size or 768
        
        vector_size_env = os.getenv("QDRANT_VECTOR_SIZE")
        vector_size = int(vector_size_env) if vector_size_env else default_vector_size

        print(f"\nUsing Configuration:")
        print(f"  - Qdrant URL: {qdrant_url}")
        print(f"  - Collection: {collection_name}")
        print(f"  - Vector size: {vector_size}")
        if args.recreate:
            print(f"  - Mode: RECREATE (will delete existing collection)")

        success = init_qdrant_collection(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            vector_size=vector_size,
            recreate=args.recreate,
        )

    sys.exit(0 if success else 1)

