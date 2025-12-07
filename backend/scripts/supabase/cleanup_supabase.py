#!/usr/bin/env python3
"""
Supabase Cleanup Script - Clear tables and buckets.

This script provides targeted cleanup of Supabase tables and Supabase Storage buckets.
You can specify which tables and buckets to clear, or clear everything.

Usage:
    cd backend
    python scripts/supabase/cleanup_supabase.py --all
    python scripts/supabase/cleanup_supabase.py --tables documents chunks
    python scripts/supabase/cleanup_supabase.py --buckets
    python scripts/supabase/cleanup_supabase.py --confirm --tables all --buckets
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

try:
    from app.core.database import get_supabase_client
    from app.core.config import settings
    from app.utils.logging import get_logger

except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install required packages:")
    print("  pip install supabase")
    sys.exit(1)

logger = get_logger(__name__)


@dataclass
class CleanupConfig:
    """Configuration for cleanup operations."""
    tables: List[str] = None
    buckets: List[str] = None
    confirm: bool = False
    dry_run: bool = False

    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.buckets is None:
            self.buckets = []


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


def get_available_tables() -> List[str]:
    """Get list of available tables in the system."""
    return ['documents', 'chunks', 'images', 'tables']


def get_available_buckets() -> List[str]:
    """Get list of available Supabase Storage buckets in the system."""
    # Based on db/db.sql, there's one bucket: document-images
    buckets = ['document-images']
    return buckets


def cleanup_supabase_tables(client, tables: List[str], dry_run: bool = False) -> bool:
    """
    Clear specified tables from Supabase.

    Args:
        client: Supabase client
        tables: List of table names to clear
        dry_run: If True, only show what would be deleted

    Returns:
        True if successful
    """
    print(f"\n🗂️  Clearing Supabase Tables: {', '.join(tables)}")

    if dry_run:
        print("🔍 DRY RUN - Would delete from tables:")
        for table in tables:
            print(f"  - {table}")
        return True

    # Get pre-deletion counts
    print("\n📊 Gathering statistics...")
    total_records = 0
    table_counts = {}

    for table in tables:
        try:
            count = len(client.table(table).select('id').execute().data)
            table_counts[table] = count
            total_records += count
            print(f"  {table}: {count:,} records")
        except Exception as e:
            print(f"  ⚠️  Could not count {table}: {e}")
            table_counts[table] = 0

    print(f"  📈 Total records to delete: {total_records:,}")

    if total_records == 0:
        print("ℹ️  No records to delete.")
        return True

    # Delete in proper order (respecting foreign key constraints)
    # Order: tables -> images -> chunks -> documents
    delete_order = []
    if 'tables' in tables:
        delete_order.append('tables')
    if 'images' in tables:
        delete_order.append('images')
    if 'chunks' in tables:
        delete_order.append('chunks')
    if 'documents' in tables:
        delete_order.append('documents')

    print("\n🗑️  Deleting data...")
    deleted_total = 0

    for table in delete_order:
        try:
            result = client.table(table).delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            deleted_count = len(result.data)
            deleted_total += deleted_count
            print(f"  ✅ {table}: deleted {deleted_count:,} records")
        except Exception as e:
            print(f"  ❌ Failed to delete from {table}: {e}")
            return False

    # Verify cleanup
    print("\n🔍 Verifying cleanup...")
    remaining_total = 0

    for table in tables:
        try:
            remaining = len(client.table(table).select('id').execute().data)
            remaining_total += remaining
            if remaining > 0:
                print(f"  ⚠️  {table}: {remaining} records still remain")
        except Exception as e:
            print(f"  ⚠️  Could not verify {table}: {e}")

    if remaining_total == 0:
        print("✅ Table cleanup completed successfully!")
        return True
    else:
        print(f"⚠️  Warning: {remaining_total} records still remain")
        return False


def cleanup_supabase_buckets(buckets: List[str], supabase_client, dry_run: bool = False) -> bool:
    """
    Clear specified Supabase Storage buckets.

    Args:
        buckets: List of bucket names to clear
        supabase_client: Supabase client instance
        dry_run: If True, only show what would be deleted

    Returns:
        True if successful
    """
    print(f"\n🪣 Clearing Supabase Storage Buckets: {', '.join(buckets)}")

    if dry_run:
        print("🔍 DRY RUN - Would empty buckets:")
        for bucket in buckets:
            print(f"  - {bucket}")
        return True

    try:
        print("🔌 Using Supabase Storage API...")

        for bucket_name in buckets:
            print(f"\n🎯 Processing bucket: {bucket_name}")

            try:
                # List all files in the bucket
                files = supabase_client.storage.from_(bucket_name).list()
                print(f"  📋 Found {len(files)} files in '{bucket_name}'")

                if not files:
                    print(f"  ℹ️  Bucket '{bucket_name}' is already empty")
                    continue

                # Delete files in batches
                batch_size = 100  # Supabase allows batch deletion
                files_deleted = 0

                for i in range(0, len(files), batch_size):
                    batch = files[i:i + batch_size]
                    file_paths = [file['name'] for file in batch]

                    try:
                        # Delete batch of files
                        supabase_client.storage.from_(bucket_name).remove(file_paths)
                        files_deleted += len(file_paths)
                        print(f"  ✅ Deleted batch of {len(file_paths)} files")
                    except Exception as e:
                        print(f"  ⚠️  Failed to delete batch: {e}")
                        # Continue with next batch

                print(f"  ✅ Deleted {files_deleted} files from '{bucket_name}'")

            except Exception as e:
                print(f"  ❌ Error with bucket '{bucket_name}': {e}")
                return False

        print("✅ Supabase Storage cleanup completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Supabase Storage cleanup failed: {e}")
        logger.error(f"Supabase Storage cleanup failed: {e}", exc_info=True)
        return False


def run_cleanup(config: CleanupConfig) -> bool:
    """
    Execute the cleanup operations.

    Args:
        config: Cleanup configuration

    Returns:
        True if all operations completed successfully
    """
    print("🧹 SUPABASE CLEANUP")
    print("="*60)

    # Show what will be cleaned
    operations = []
    if config.tables:
        operations.append(f"Tables: {', '.join(config.tables)}")
    if config.buckets:
        operations.append(f"Buckets: {', '.join(config.buckets)}")

    if not operations:
        print("❌ No cleanup operations specified.")
        return False

    print("Operations to perform:")
    for op in operations:
        print(f"  - {op}")

    if config.dry_run:
        print("\n🔍 DRY RUN MODE - No actual changes will be made")

    print("="*60)

    # Get confirmation for destructive operations
    if not config.dry_run and (config.tables or config.buckets):
        if not get_confirmation("This will permanently delete data", config.confirm):
            print("❌ Cleanup cancelled by user")
            return False

    results = []

    # Connect to Supabase if needed
    client = None
    if config.tables:
        try:
            print("\n🔌 Connecting to Supabase...")
            client = get_supabase_client()
            print("✅ Connected to Supabase")
        except Exception as e:
            print(f"❌ Failed to connect to Supabase: {e}")
            return False

    # Cleanup tables
    if config.tables:
        try:
            success = cleanup_supabase_tables(client, config.tables, config.dry_run)
            results.append(("Tables", success))
        except Exception as e:
            print(f"❌ Table cleanup failed: {e}")
            results.append(("Tables", False))

    # Cleanup buckets
    if config.buckets:
        try:
            success = cleanup_supabase_buckets(config.buckets, client, config.dry_run)
            results.append(("Buckets", success))
        except Exception as e:
            print(f"❌ Bucket cleanup failed: {e}")
            results.append(("Buckets", False))

    # Summary
    print("\n" + "="*60)
    print("📊 CLEANUP SUMMARY")
    print("="*60)

    successful = 0
    for operation, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{operation:15} : {status}")
        if success:
            successful += 1

    if successful == len(results):
        if config.dry_run:
            print("🎭 DRY RUN COMPLETED - No changes were made")
        else:
            print("🎉 CLEANUP COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("⚠️  CLEANUP PARTIALLY COMPLETED")
        return False


def main():
    """Main entry point."""
    import argparse

    available_tables = get_available_tables()
    available_buckets = get_available_buckets()

    parser = argparse.ArgumentParser(
        description="Clean up Supabase tables and Supabase Storage buckets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Clear all tables and buckets
  cd backend
  python scripts/supabase/cleanup_supabase.py --all --confirm

  # Clear specific tables
  cd backend
  python scripts/supabase/cleanup_supabase.py --tables documents chunks --confirm

  # Clear all tables
  cd backend
  python scripts/supabase/cleanup_supabase.py --tables all --confirm

  # Clear buckets only
  cd backend
  python scripts/supabase/cleanup_supabase.py --buckets --confirm

  # Dry run to see what would be deleted
  cd backend
  python scripts/supabase/cleanup_supabase.py --all --dry-run

Available tables: {', '.join(available_tables)}
Available Supabase Storage buckets: {', '.join(available_buckets)}
        """
    )

    parser.add_argument(
        "--tables",
        nargs="*",
        choices=available_tables + ["all"],
        help=f"Tables to clear. Choices: {', '.join(available_tables)}, or 'all'"
    )

    parser.add_argument(
        "--buckets",
        nargs="*",
        choices=available_buckets + ["all"],
        help=f"Supabase Storage buckets to clear. Choices: {', '.join(available_buckets)}, or 'all'"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear all tables and buckets"
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompts (use with caution!)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.tables and not args.buckets:
        parser.error("Must specify --all, --tables, or --buckets")

    if args.all and (args.tables or args.buckets):
        parser.error("Cannot use --all with --tables or --buckets")

    # Build configuration
    config = CleanupConfig(
        confirm=args.confirm,
        dry_run=args.dry_run
    )

    if args.all:
        config.tables = available_tables.copy()
        config.buckets = available_buckets.copy()
    else:
        if args.tables:
            if "all" in args.tables:
                config.tables = available_tables.copy()
            else:
                config.tables = args.tables

        if args.buckets:
            if "all" in args.buckets:
                config.buckets = available_buckets.copy()
            else:
                config.buckets = args.buckets

    try:
        success = run_cleanup(config)
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
