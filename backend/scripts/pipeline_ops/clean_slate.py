"""
Clean Slate Reconstruction Script - Complete system reset.

This script performs a complete clean slate reconstruction by clearing all data stores
in the correct order for a fresh start of the multimodal RAG system.

Order of operations:
1. Supabase (PostgreSQL) - Metadata & document database
2. Elasticsearch - BM25 sparse index
3. Neo4j - Knowledge graph
4. Qdrant - Vector database collections
5. MinIO - Raw data lake

Usage:
    python backend/scripts/pipeline_ops/clean_slate.py
    python backend/scripts/pipeline_ops/clean_slate.py --confirm  # Skips all confirmation prompts
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

# Import required modules
try:
    from app.core.database import get_supabase_client
    from app.core.config import settings
    from app.utils.logging import get_logger

    # External dependencies
    from qdrant_client import QdrantClient
    from elasticsearch import Elasticsearch
    import boto3
    from botocore.exceptions import ClientError
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install required packages:")
    print("  pip install supabase qdrant-client elasticsearch boto3")
    sys.exit(1)

logger = get_logger(__name__)


def get_confirmation(message: str, confirm: bool = False) -> bool:
    """Get user confirmation unless --confirm flag is used."""
    if confirm:
        print(f"✅ {message} (auto-confirmed)")
        return True

    response = input(f"⚠️  {message} Continue? (yes/no): ")
    if response.lower() != "yes":
        print("⏭️  Skipped.")
        return False
    return True


def cleanup_supabase(confirm: bool = False) -> bool:
    """
    Clear all data from Supabase tables.

    Based on: backend/scripts/supabase/cleanup_supabase.py pattern
    """
    print("\n" + "="*60)
    print("🗂️  PHASE 1: Clearing Supabase (PostgreSQL)")
    print("="*60)

    if not get_confirmation("This will delete ALL documents, chunks, images, and tables from Supabase", confirm):
        return True  # Skip is not failure

    try:
        print("🔌 Connecting to Supabase...")
        client = get_supabase_client()
        print("✅ Connected to Supabase")

        # Get pre-deletion counts
        print("\n📊 Gathering statistics...")
        try:
            docs_count = len(client.table('documents').select('id').execute().data)
            chunks_count = len(client.table('chunks').select('id').execute().data)
            images_count = len(client.table('images').select('id').execute().data)
            tables_count = len(client.table('tables').select('id').execute().data)

            print(f"  📄 Documents: {docs_count:,}")
            print(f"  📝 Chunks: {chunks_count:,}")
            print(f"  🖼️  Images: {images_count:,}")
            print(f"  📊 Tables: {tables_count:,}")
            print(f"  📈 Total records: {docs_count + chunks_count + images_count + tables_count:,}")
        except Exception as e:
            logger.warning(f"Could not get counts: {e}")

        # Delete in proper order (respecting foreign key constraints)
        print("\n🗑️  Deleting data...")

        # 1. Delete tables (no dependencies)
        print("  - Deleting tables...")
        result = client.table('tables').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"    ✅ Deleted {len(result.data)} table records")

        # 2. Delete images (no dependencies)
        print("  - Deleting images...")
        result = client.table('images').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"    ✅ Deleted {len(result.data)} image records")

        # 3. Delete chunks (depends on documents)
        print("  - Deleting chunks...")
        result = client.table('chunks').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"    ✅ Deleted {len(result.data)} chunk records")

        # 4. Delete documents (depends on nothing)
        print("  - Deleting documents...")
        result = client.table('documents').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"    ✅ Deleted {len(result.data)} document records")

        # Verify cleanup
        print("\n🔍 Verifying cleanup...")
        docs_remaining = len(client.table('documents').select('id').execute().data)
        chunks_remaining = len(client.table('chunks').select('id').execute().data)
        images_remaining = len(client.table('images').select('id').execute().data)
        tables_remaining = len(client.table('tables').select('id').execute().data)

        if docs_remaining == 0 and chunks_remaining == 0 and images_remaining == 0 and tables_remaining == 0:
            print("✅ Supabase cleanup completed successfully!")
            return True
        else:
            print(f"⚠️  Warning: Some records remain - Documents: {docs_remaining}, Chunks: {chunks_remaining}, Images: {images_remaining}, Tables: {tables_remaining}")
            return False

    except Exception as e:
        print(f"❌ Supabase cleanup failed: {e}")
        logger.error(f"Supabase cleanup failed: {e}", exc_info=True)
        return False


def cleanup_elasticsearch(confirm: bool = False) -> bool:
    """
    Clear all documents from Elasticsearch index.

    Based on: scripts/elasticsearch/clean_elasticsearch.py pattern
    """
    print("\n" + "="*60)
    print("🔍 PHASE 2: Clearing Elasticsearch")
    print("="*60)

    if not get_confirmation("This will delete ALL documents from Elasticsearch index", confirm):
        return True

    try:
        print("🔌 Connecting to Elasticsearch...")
        es_url = getattr(settings, 'elasticsearch_url', 'http://localhost:9200')
        es = Elasticsearch(es_url, request_timeout=30)
        print("✅ Connected to Elasticsearch")

        index_name = getattr(settings, 'elasticsearch_index_name', 'rag_chunks')

        # Check if index exists
        if not es.indices.exists(index=index_name):
            print(f"ℹ️  Index '{index_name}' does not exist - nothing to clean")
            return True

        # Get pre-deletion stats
        stats = es.indices.stats(index=index_name)
        doc_count = stats['indices'][index_name]['total']['docs']['count']
        print(f"📊 Index contains {doc_count:,} documents")

        # Delete all documents
        print("🗑️  Deleting all documents...")
        query = {"query": {"match_all": {}}}
        response = es.delete_by_query(
            index=index_name,
            body=query,
            request_timeout=60,
            refresh=True
        )

        deleted_count = response.get("deleted", 0)
        print(f"✅ Deleted {deleted_count:,} documents from index '{index_name}'")

        # Verify cleanup
        stats_after = es.indices.stats(index=index_name)
        remaining = stats_after['indices'][index_name]['total']['docs']['count']
        if remaining == 0:
            print("✅ Elasticsearch cleanup completed successfully!")
            return True
        else:
            print(f"⚠️  Warning: {remaining} documents still remain")
            return False

    except Exception as e:
        print(f"❌ Elasticsearch cleanup failed: {e}")
        logger.error(f"Elasticsearch cleanup failed: {e}", exc_info=True)
        return False


def cleanup_neo4j(confirm: bool = False) -> bool:
    """
    Clear all nodes and relationships from Neo4j.

    Based on: backend/scripts/neo4j/cleanup_neo4j.py pattern
    """
    print("\n" + "="*60)
    print("🕸️  PHASE 3: Clearing Neo4j Knowledge Graph")
    print("="*60)

    if not get_confirmation("This will delete ALL nodes and relationships from Neo4j", confirm):
        return True

    try:
        print("🔌 Connecting to Neo4j...")
        from app.core.neo4j_database import get_neo4j_driver

        driver = get_neo4j_driver()
        session = driver.session(database=settings.neo4j_database)
        print("✅ Connected to Neo4j")

        # Execute nuclear cleanup
        print("🗑️  Deleting all nodes and relationships...")
        query = "MATCH (n) DETACH DELETE n"
        result = session.run(query)

        # Get summary
        summary = result.consume()
        nodes_deleted = summary.counters.nodes_deleted
        relationships_deleted = summary.counters.relationships_deleted

        print(f"✅ Deleted {nodes_deleted:,} nodes")
        print(f"✅ Deleted {relationships_deleted:,} relationships")

        session.close()
        driver.close()

        print("✅ Neo4j cleanup completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Neo4j cleanup failed: {e}")
        logger.error(f"Neo4j cleanup failed: {e}", exc_info=True)
        return False


def cleanup_qdrant(confirm: bool = False) -> bool:
    """
    Delete all Qdrant collections.

    Based on: backend/scripts/qdrant/init_qdrant.py pattern
    """
    print("\n" + "="*60)
    print("🔗 PHASE 4: Clearing Qdrant Collections")
    print("="*60)

    if not get_confirmation("This will delete ALL Qdrant collections", confirm):
        return True

    try:
        print("🔌 Connecting to Qdrant...")
        qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        client = QdrantClient(url=qdrant_url)
        print("✅ Connected to Qdrant")

        # Get existing collections
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if not collection_names:
            print("ℹ️  No collections found - nothing to clean")
            return True

        print(f"📊 Found {len(collection_names)} collections: {', '.join(collection_names)}")

        # Delete each collection
        print("🗑️  Deleting collections...")
        deleted_count = 0

        for collection_name in collection_names:
            try:
                # Get collection info before deletion
                info = client.get_collection(collection_name)
                points_count = info.points_count

                # Delete collection
                client.delete_collection(collection_name)
                deleted_count += 1

                print(f"  ✅ Deleted '{collection_name}' ({points_count:,} vectors)")

            except Exception as e:
                print(f"  ❌ Failed to delete '{collection_name}': {e}")

        print(f"✅ Successfully deleted {deleted_count} collections")
        return True

    except Exception as e:
        print(f"❌ Qdrant cleanup failed: {e}")
        logger.error(f"Qdrant cleanup failed: {e}", exc_info=True)
        return False


def cleanup_supabase_storage(confirm: bool = False) -> bool:
    """
    Clear all Supabase Storage buckets.

    Uses Supabase Storage API to delete all files from buckets.
    """
    print("\n" + "="*60)
    print("📦 PHASE 5: Clearing Supabase Storage")
    print("="*60)

    try:
        # Get Supabase client
        client = get_supabase_client()

        # Buckets to clean
        buckets_to_clean = ['document-images']  # Add more buckets as needed

        print(f"🎯 Target buckets: {', '.join(buckets_to_clean)}")

        if not get_confirmation(f"This will delete ALL files from Supabase Storage buckets: {', '.join(buckets_to_clean)}", confirm):
            return True  # Skip is not failure

        print("🔌 Using Supabase Storage API...")

        total_files_deleted = 0

        for bucket_name in buckets_to_clean:
            print(f"\n🎯 Processing bucket: {bucket_name}")

            try:
                # First, list all document folders (top-level prefixes)
                document_folders = client.storage.from_(bucket_name).list()
                print(f"  📁 Found {len(document_folders)} document folders in '{bucket_name}'")

                if not document_folders:
                    print(f"  ℹ️  Bucket '{bucket_name}' is already empty")
                    continue

                # For each document folder, list and delete all files
                batch_size = 100  # Supabase allows batch deletion
                all_file_paths = []

                for folder_info in document_folders:
                    document_id = folder_info['name']
                    print(f"  📂 Processing document folder: {document_id}")

                    try:
                        # List all files in this document's folder
                        files_in_folder = client.storage.from_(bucket_name).list(document_id)

                        if files_in_folder:
                            # Build full paths for files in this folder
                            for file_info in files_in_folder:
                                file_path = f"{document_id}/{file_info['name']}"
                                all_file_paths.append(file_path)

                            print(f"    📋 Found {len(files_in_folder)} files")
                        else:
                            print("    📋 No files found")

                    except Exception as e:
                        print(f"    ⚠️  Error listing files in folder {document_id}: {e}")
                        continue

                if not all_file_paths:
                    print("  ℹ️  No files found in any document folders")
                    continue

                print(f"  📊 Total files to delete: {len(all_file_paths)}")

                # Delete files in batches
                files_deleted = 0
                for i in range(0, len(all_file_paths), batch_size):
                    batch = all_file_paths[i:i + batch_size]

                    try:
                        # Delete batch of files
                        client.storage.from_(bucket_name).remove(batch)
                        files_deleted += len(batch)
                        print(f"  ✅ Deleted batch of {len(batch)} files ({files_deleted:,}/{len(all_file_paths):,} total)")
                    except Exception as e:
                        print(f"  ⚠️  Failed to delete batch: {e}")
                        # Continue with next batch

                print(f"  ✅ Deleted {files_deleted} files from '{bucket_name}'")
                total_files_deleted += files_deleted

            except Exception as e:
                print(f"  ❌ Error with bucket '{bucket_name}': {e}")
                return False

        print("✅ Supabase Storage cleanup completed successfully!")
        print(f"   Deleted {total_files_deleted} files from {len(buckets_to_clean)} bucket(s)")
        return True

    except Exception as e:
        print(f"❌ Supabase Storage cleanup failed: {e}")
        logger.error(f"Supabase Storage cleanup failed: {e}", exc_info=True)
        return False


def cleanup_minio(confirm: bool = False) -> bool:
    """
    Empty the MinIO bucket by deleting all objects.

    Uses boto3 to connect to MinIO and programmatically delete all objects.
    """
    print("\n" + "="*60)
    print("📦 PHASE 5: Clearing MinIO Data Lake")
    print("="*60)

    try:
        # Get MinIO configuration
        minio_endpoint = getattr(settings, 'minio_endpoint', 'localhost:9000')
        bucket_name = getattr(settings, 'minio_bucket_name', 'raw-documents')
        access_key = getattr(settings, 'minio_access_key', 'admin')
        secret_key = getattr(settings, 'minio_secret_key', 'admin12345')
        use_ssl = getattr(settings, 'minio_use_ssl', False)

        print(f"🎯 Target bucket: '{bucket_name}' at {minio_endpoint}")

        if not get_confirmation(f"This will delete ALL files from MinIO bucket '{bucket_name}'", confirm):
            return True  # Skip is not failure

        print("🔌 Connecting to MinIO...")

        # Create S3 client for MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url=f"http{'s' if use_ssl else ''}://{minio_endpoint}",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=None,  # MinIO doesn't require a region
            use_ssl=use_ssl
        )

        print("✅ Connected to MinIO")

        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"📋 Bucket '{bucket_name}' exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"ℹ️  Bucket '{bucket_name}' does not exist - nothing to clean")
                return True
            else:
                print(f"❌ Error checking bucket: {e}")
                return False

        # List all objects in the bucket
        print("📋 Listing objects in bucket...")
        objects_to_delete = []
        continuation_token = None

        while True:
            list_kwargs = {
                'Bucket': bucket_name,
                'MaxKeys': 1000
            }
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token

            try:
                response = s3_client.list_objects_v2(**list_kwargs)
                contents = response.get('Contents', [])

                for obj in contents:
                    objects_to_delete.append({'Key': obj['Key']})

                if not response.get('IsTruncated'):
                    break
                continuation_token = response.get('NextContinuationToken')

            except ClientError as e:
                print(f"❌ Error listing objects: {e}")
                return False

        total_objects = len(objects_to_delete)
        if total_objects == 0:
            print("ℹ️  Bucket is already empty")
            return True

        print(f"📊 Found {total_objects:,} objects to delete")

        # Delete objects in batches (S3 allows max 1000 objects per delete request)
        batch_size = 1000
        deleted_count = 0

        print("🗑️  Deleting objects...")
        for i in range(0, total_objects, batch_size):
            batch = objects_to_delete[i:i + batch_size]
            delete_request = {
                'Objects': batch,
                'Quiet': True  # Don't return deleted object info to reduce response size
            }

            try:
                response = s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete=delete_request
                )

                batch_deleted = len(batch)
                deleted_count += batch_deleted
                print(f"  ✅ Deleted batch of {batch_deleted} objects ({deleted_count:,}/{total_objects:,} total)")

            except ClientError as e:
                print(f"  ❌ Error deleting batch: {e}")
                return False

        # Verify cleanup
        print("🔍 Verifying cleanup...")
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            remaining = len(response.get('Contents', []))
            if remaining == 0:
                print("✅ MinIO cleanup completed successfully!")
                print(f"   Deleted {deleted_count:,} objects from bucket '{bucket_name}'")
                return True
            else:
                print(f"⚠️  Warning: {remaining} objects still remain")
                return False

        except ClientError as e:
            print(f"❌ Error verifying cleanup: {e}")
            return False

    except Exception as e:
        print(f"❌ MinIO cleanup failed: {e}")
        logger.error(f"MinIO cleanup failed: {e}", exc_info=True)
        return False


def run_clean_slate(confirm: bool = False) -> bool:
    """
    Execute complete clean slate procedure.

    Args:
        confirm: Skip confirmation prompts

    Returns:
        True if all phases completed successfully
    """
    print("🧹 CLEAN SLATE RECONSTRUCTION")
    print("="*60)
    print("This will clear ALL data from the multimodal RAG system:")
    print("1. 🗂️  Supabase (PostgreSQL) - Documents, chunks, images, tables")
    print("2. 🔍 Elasticsearch - BM25 index chunks")
    print("3. 🕸️  Neo4j - Knowledge graph nodes and relationships")
    print("4. 🔗 Qdrant - Vector collections")
    print("5. 📦 Supabase Storage - Document images and files")
    print("6. 📦 MinIO - Raw document files")
    print("="*60)

    if not confirm and not get_confirmation("Begin complete clean slate reconstruction?", confirm):
        print("❌ Clean slate cancelled by user")
        return False

    results = []
    start_time = __import__('time').time()

    # Execute cleanup phases in order
    phases = [
        ("Supabase", cleanup_supabase),
        ("Elasticsearch", cleanup_elasticsearch),
        ("Neo4j", cleanup_neo4j),
        ("Qdrant", cleanup_qdrant),
        ("Supabase Storage", cleanup_supabase_storage),
        ("MinIO", cleanup_minio),
    ]

    for phase_name, cleanup_func in phases:
        try:
            success = cleanup_func(confirm)
            results.append((phase_name, success))

            if not success and phase_name not in ["Supabase Storage", "MinIO"]:  # Storage failures are not critical
                print(f"❌ Critical failure in {phase_name} phase - aborting")
                break

        except Exception as e:
            print(f"❌ Unexpected error in {phase_name}: {e}")
            results.append((phase_name, False))
            break

    # Summary
    print("\n" + "="*60)
    print("📊 CLEAN SLATE SUMMARY")
    print("="*60)

    successful = 0
    for phase_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{phase_name:15} : {status}")
        if success:
            successful += 1

    elapsed = __import__('time').time() - start_time

    print("-"*60)
    print(f"Completed: {successful}/{len(phases)} phases")
    print(f"Time taken: {elapsed:.1f} seconds")
    if successful == len(phases):
        print("🎉 CLEAN SLATE RECONSTRUCTION COMPLETED SUCCESSFULLY!")
        print("   You can now run initialization scripts to rebuild the system.")
        return True
    else:
        print("⚠️  CLEAN SLATE PARTIALLY COMPLETED")
        print("   Some phases failed - check logs and retry or fix manually.")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete clean slate reconstruction of multimodal RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (with confirmations)
  python backend/scripts/pipeline_ops/clean_slate.py

  # Automated mode (skip all confirmations)
  python backend/scripts/pipeline_ops/clean_slate.py --confirm

  # Show what would be cleaned without doing it
  python backend/scripts/pipeline_ops/clean_slate.py --dry-run
        """
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip all confirmation prompts (use with caution!)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually doing it"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE")
        print("This would clean:")
        print("1. 🗂️  Supabase: All documents, chunks, images, tables")
        print("2. 🔍 Elasticsearch: All chunks from BM25 index")
        print("3. 🕸️  Neo4j: All nodes and relationships")
        print("4. 🔗 Qdrant: All vector collections")
        print("5. 📦 Supabase Storage: All files in document-images bucket")
        print("6. 📦 MinIO: All files in raw-documents bucket")
        print("\nRun without --dry-run to actually perform the cleanup.")
        return

    try:
        success = run_clean_slate(confirm=args.confirm)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
