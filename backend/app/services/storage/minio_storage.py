"""
MinIO storage service.

Handles raw document storage in MinIO (S3-compatible object storage).
Implements the data lake pattern with structured paths.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional
from io import BytesIO

from minio import Minio
from minio.error import S3Error
from minio.commonconfig import REPLACE
from minio.deleteobjects import DeleteObject

from app.core.config import settings
from app.utils.exceptions import StorageError

logger = logging.getLogger(__name__)


class MinIOStorage:
    """
    MinIO storage service for raw document storage.
    
    Implements structured storage following the pattern:
    raw_documents/{source}/{document_type}/{filename}
    
    Supports version tracking via timestamped filenames.
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        use_ssl: Optional[bool] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO server endpoint (default: from config)
            access_key: Access key (default: from config)
            secret_key: Secret key (default: from config)
            bucket_name: Bucket name (default: from config)
            use_ssl: Whether to use SSL (default: from config)
            region: AWS region (default: from config)
        """
        self.endpoint = endpoint or settings.minio_endpoint
        self.access_key = access_key or settings.minio_access_key
        self.secret_key = secret_key or settings.minio_secret_key
        self.bucket_name = bucket_name or settings.minio_bucket_name
        self.use_ssl = use_ssl if use_ssl is not None else settings.minio_use_ssl
        self.region = region or settings.minio_region
        
        # Initialize MinIO client
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.use_ssl,
                region=self.region,
            )
            logger.info(f"Initialized MinIO client for endpoint: {self.endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {str(e)}")
            raise StorageError(
                f"Failed to initialize MinIO client: {str(e)}",
                {"endpoint": self.endpoint, "error": str(e)},
            ) from e
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Ensure the bucket exists, create if it doesn't."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                logger.info(f"Creating bucket: {self.bucket_name}")
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created successfully")
            else:
                logger.debug(f"Bucket '{self.bucket_name}' already exists")
        except S3Error as e:
            logger.error(f"Failed to check/create bucket: {str(e)}")
            raise StorageError(
                f"Failed to ensure bucket exists: {str(e)}",
                {"bucket_name": self.bucket_name, "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error ensuring bucket exists: {str(e)}")
            raise StorageError(
                f"Unexpected error ensuring bucket exists: {str(e)}",
                {"bucket_name": self.bucket_name, "error": str(e)},
            ) from e
    
    def upload_raw_document(
        self,
        file_bytes: bytes,
        filename: str,
        source: str = "upload",
        document_type: Optional[str] = None,
        add_timestamp: bool = True,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a raw document to MinIO with structured path.
        
        Path structure: raw_documents/{source}/{document_type}/{filename}
        If add_timestamp is True, appends timestamp to filename for versioning.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            source: Source identifier (e.g., "upload", "api", "scraper")
            document_type: Document type (e.g., "pdf", "docx"). If None, inferred from filename
            add_timestamp: Whether to add timestamp to filename for versioning
            content_type: MIME type of the file. If None, inferred from filename
        
        Returns:
            Object key (path) in MinIO where the file was stored
        
        Raises:
            StorageError: If upload fails
        """
        try:
            # Infer document type from filename if not provided
            if document_type is None:
                file_ext = Path(filename).suffix.lower()
                document_type = file_ext[1:] if file_ext else "unknown"
            
            # Generate filename with optional timestamp
            if add_timestamp:
                file_path = Path(filename)
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                name_without_ext = file_path.stem
                ext = file_path.suffix
                timestamped_filename = f"{name_without_ext}_{timestamp}{ext}"
            else:
                timestamped_filename = filename
            
            # Construct object key following the pattern: raw_documents/{source}/{document_type}/{filename}
            object_key = f"raw_documents/{source}/{document_type}/{timestamped_filename}"
            
            # Infer content type if not provided
            if content_type is None:
                content_type = self._infer_content_type(filename)
            
            # Upload file
            file_stream = BytesIO(file_bytes)
            file_size = len(file_bytes)
            
            logger.info(
                f"Uploading document to MinIO: {object_key} "
                f"({file_size} bytes, content-type: {content_type})"
            )
            
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
                data=file_stream,
                length=file_size,
                content_type=content_type,
            )
            
            logger.info(f"Successfully uploaded document to MinIO: {object_key}")
            
            return object_key
        
        except S3Error as e:
            logger.error(f"MinIO S3Error during upload: {str(e)}")
            raise StorageError(
                f"Failed to upload document to MinIO: {str(e)}",
                {
                    "filename": filename,
                    "source": source,
                    "document_type": document_type,
                    "error": str(e),
                },
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during upload: {str(e)}")
            raise StorageError(
                f"Unexpected error during upload: {str(e)}",
                {
                    "filename": filename,
                    "source": source,
                    "document_type": document_type,
                    "error": str(e)},
            ) from e
    
    def download_document(self, object_key: str) -> bytes:
        """
        Download a document from MinIO by object key.
        
        Args:
            object_key: Object key (path) in MinIO
        
        Returns:
            File content as bytes
        
        Raises:
            StorageError: If download fails
        """
        try:
            logger.info(f"Downloading document from MinIO: {object_key}")
            
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
            )
            
            file_bytes = response.read()
            response.close()
            response.release_conn()
            
            logger.info(f"Successfully downloaded document from MinIO: {object_key} ({len(file_bytes)} bytes)")
            
            return file_bytes
        
        except S3Error as e:
            logger.error(f"MinIO S3Error during download: {str(e)}")
            raise StorageError(
                f"Failed to download document from MinIO: {str(e)}",
                {"object_key": object_key, "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during download: {str(e)}")
            raise StorageError(
                f"Unexpected error during download: {str(e)}",
                {"object_key": object_key, "error": str(e)},
            ) from e
    
    def delete_document(self, object_key: str) -> None:
        """
        Delete a document from MinIO by object key.
        
        Args:
            object_key: Object key (path) in MinIO
        
        Raises:
            StorageError: If deletion fails
        """
        try:
            logger.info(f"Deleting document from MinIO: {object_key}")
            
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
            )
            
            logger.info(f"Successfully deleted document from MinIO: {object_key}")
        
        except S3Error as e:
            logger.error(f"MinIO S3Error during deletion: {str(e)}")
            raise StorageError(
                f"Failed to delete document from MinIO: {str(e)}",
                {"object_key": object_key, "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during deletion: {str(e)}")
            raise StorageError(
                f"Unexpected error during deletion: {str(e)}",
                {"object_key": object_key, "error": str(e)},
            ) from e
    
    def get_document_url(self, object_key: str, expires_in_seconds: int = 3600) -> str:
        """
        Get a presigned URL for a document.
        
        Args:
            object_key: Object key (path) in MinIO
            expires_in_seconds: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            Presigned URL string
        
        Raises:
            StorageError: If URL generation fails
        """
        try:
            logger.debug(f"Generating presigned URL for: {object_key} (expires in {expires_in_seconds}s)")
            
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
                expires=expires_in_seconds,
            )
            
            return url
        
        except S3Error as e:
            logger.error(f"MinIO S3Error during URL generation: {str(e)}")
            raise StorageError(
                f"Failed to generate presigned URL: {str(e)}",
                {"object_key": object_key, "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during URL generation: {str(e)}")
            raise StorageError(
                f"Unexpected error during URL generation: {str(e)}",
                {"object_key": object_key, "error": str(e)},
            ) from e
    
    def list_documents(
        self,
        prefix: str = "raw_documents/",
        recursive: bool = True,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """
        List documents in MinIO bucket.
        
        Args:
            prefix: Prefix to filter objects (default: "raw_documents/")
            recursive: Whether to recursively list (default: True)
            limit: Maximum number of objects to return (default: None for all)
        
        Returns:
            List of dictionaries containing document metadata:
            - object_key: Full object key (path)
            - filename: Filename extracted from path
            - size: File size in bytes
            - last_modified: Last modification timestamp
            - content_type: MIME type
            - source: Source identifier from path
            - document_type: Document type from path
        """
        try:
            logger.info(f"Listing documents in MinIO with prefix: {prefix}")
            
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=recursive,
            )
            
            documents = []
            for obj in objects:
                if limit and len(documents) >= limit:
                    break
                
                # Parse path to extract metadata
                # Path format: raw_documents/{source}/{document_type}/{filename}
                parts = obj.object_name.split("/")
                source = parts[1] if len(parts) > 1 else "unknown"
                document_type = parts[2] if len(parts) > 2 else "unknown"
                filename = parts[-1] if parts else obj.object_name
                
                # Get object metadata
                try:
                    stat = self.client.stat_object(
                        bucket_name=self.bucket_name,
                        object_name=obj.object_name,
                    )
                    content_type = stat.content_type or self._infer_content_type(filename)
                except S3Error:
                    content_type = self._infer_content_type(filename)
                
                documents.append({
                    "object_key": obj.object_name,
                    "filename": filename,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "content_type": content_type,
                    "source": source,
                    "document_type": document_type,
                })
            
            logger.info(f"Found {len(documents)} documents")
            return documents
        
        except S3Error as e:
            logger.error(f"MinIO S3Error during listing: {str(e)}")
            raise StorageError(
                f"Failed to list documents in MinIO: {str(e)}",
                {"prefix": prefix, "error": str(e)},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during listing: {str(e)}")
            raise StorageError(
                f"Unexpected error during listing: {str(e)}",
                {"prefix": prefix, "error": str(e)},
            ) from e
    
    def document_exists(self, object_key: str) -> bool:
        """
        Check if a document exists in MinIO.
        
        Args:
            object_key: Object key (path) in MinIO
        
        Returns:
            True if document exists, False otherwise
        """
        try:
            self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
            )
            return True
        except S3Error:
            return False
        except Exception as e:
            logger.warning(f"Error checking if document exists: {str(e)}")
            return False
    
    @staticmethod
    def _infer_content_type(filename: str) -> str:
        """
        Infer content type from filename extension.
        
        Args:
            filename: Filename with extension
        
        Returns:
            MIME type string
        """
        extension_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".json": "application/json",
            ".xml": "application/xml",
            ".csv": "text/csv",
            ".html": "text/html",
            ".htm": "text/html",
        }
        
        ext = Path(filename).suffix.lower()
        return extension_map.get(ext, "application/octet-stream")
