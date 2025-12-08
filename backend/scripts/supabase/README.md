# Supabase Scripts

Utility scripts for managing Supabase database tables and MinIO buckets.

## cleanup_supabase.py

A comprehensive script to clean up Supabase tables and Supabase Storage buckets with granular control over what gets deleted.

### Features

- **Selective Table Cleanup**: Clear specific tables or all tables
- **Bucket Management**: Empty Supabase Storage buckets automatically
- **Dry Run Mode**: Preview what would be deleted without making changes
- **Safety Confirmations**: Built-in confirmation prompts (can be skipped with `--confirm`)
- **Detailed Reporting**: Shows statistics before and after cleanup

### Available Tables

- `documents` - Document metadata
- `chunks` - Text chunks from documents
- `images` - Image metadata and references
- `tables` - Table data extracted from documents

### Available Buckets

- `raw-documents` - Raw uploaded files
- `processed-files` - Processed document files
- `temp-uploads` - Temporary upload files

## Usage

### Quick Start

```bash
# Clear everything (requires confirmation)
cd backend
python scripts/supabase/cleanup_supabase.py --all

# Clear everything without confirmation
cd backend
python scripts/supabase/cleanup_supabase.py --all --confirm
```

### Selective Cleanup

```bash
# Clear specific tables only
cd backend
python scripts/supabase/cleanup_supabase.py --tables documents chunks --confirm

# Clear all tables
cd backend
python scripts/supabase/cleanup_supabase.py --tables all --confirm

# Clear buckets only
cd backend
python scripts/supabase/cleanup_supabase.py --buckets all --confirm

# Clear tables and buckets
cd backend
python scripts/supabase/cleanup_supabase.py --tables all --buckets --confirm
```

### Preview Mode

```bash
# See what would be deleted without actually deleting
cd backend
python scripts/supabase/cleanup_supabase.py --all --dry-run
```

## How It Works

### Table Cleanup Order

The script deletes data in the correct order to respect foreign key constraints:

1. **tables** - No dependencies
2. **images** - No dependencies
3. **chunks** - Depends on documents
4. **documents** - No dependencies (parent table)

### Bucket Cleanup

- Uses Supabase Storage API through the Supabase client
- Lists all files in the specified buckets
- Deletes files in batches for efficiency
- Reports total files deleted

## Configuration

The script uses configuration from `backend/config.yaml` and `app/core/config.py`:

- **Supabase**: Uses existing Supabase client configuration for both database and storage operations

## Safety Features

- **Confirmation Prompts**: All destructive operations require user confirmation
- **Dry Run Mode**: Preview operations without making changes
- **Error Handling**: Graceful handling of connection issues and missing resources
- **Verification**: Checks cleanup completion and reports remaining data

## Examples

### Development Cleanup

```bash
# Clear all data during development
cd backend
python scripts/supabase/cleanup_supabase.py --all --confirm
```

### Selective Testing

```bash
# Clear only image-related data for testing
cd backend
python scripts/supabase/cleanup_supabase.py --tables images --confirm
```

### Bucket Management

```bash
# Clear uploaded files without affecting database
cd backend
python scripts/supabase/cleanup_supabase.py --buckets all --confirm
```

## Integration

This script is designed to work alongside the pipeline operations:

- **clean_slate.py**: For complete system reset (includes this script's functionality)
- **reconstruct.py**: For rebuilding schemas after cleanup

The cleanup script provides more granular control compared to the full clean slate operation.

## Notes

- Scripts should be run from the `backend/` directory
- Make sure your environment is configured with correct Supabase and MinIO credentials
- The cleanup script uses the same connection settings as the main application
