"""
Storage services.

Provides storage abstractions for:
- MinIO/S3 (raw data lake)
- Future: Other storage backends
"""

from app.services.storage.minio_storage import MinIOStorage, StorageError

__all__ = ["MinIOStorage", "StorageError"]
