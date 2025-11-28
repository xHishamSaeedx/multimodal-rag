"""
Supabase storage service for images.

Handles image file storage in Supabase Storage (document-images bucket).
"""

import logging
import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from app.core.database import get_supabase_client, DatabaseError
from app.utils.exceptions import StorageError

logger = logging.getLogger(__name__)


class SupabaseImageStorage:
    """
    Supabase storage service for document images.
    
    Implements structured storage following the pattern:
    {document_id}/image_{timestamp}-{random}.{ext}
    """
    
    BUCKET_NAME = "document-images"
    
    def __init__(self):
        """Initialize Supabase storage client."""
        try:
            self.client = get_supabase_client()
            logger.info("Initialized Supabase image storage client")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase storage client: {str(e)}")
            raise StorageError(
                f"Failed to initialize Supabase storage client: {str(e)}",
                {"error": str(e)},
            ) from e
    
    def upload_image(
        self,
        image_bytes: bytes,
        document_id: UUID,
        image_index: int,
        image_ext: str,
        add_timestamp: bool = True,
    ) -> str:
        """
        Upload image to Supabase storage.
        
        Args:
            image_bytes: Image file content as bytes
            document_id: Document UUID
            image_index: Image index within document (for filename)
            image_ext: Image file extension (jpg, png, etc.)
            add_timestamp: Whether to add timestamp to filename (default: True)
        
        Returns:
            Storage path (relative to bucket): {document_id}/image_{timestamp}-{hash}.{ext}
        
        Raises:
            StorageError: If upload fails
        """
        try:
            # Generate unique filename
            image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
            
            if add_timestamp:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{image_index}_{timestamp}_{image_hash}.{image_ext}"
            else:
                filename = f"image_{image_index}_{image_hash}.{image_ext}"
            
            # Storage path: {document_id}/filename
            storage_path = f"{document_id}/{filename}"
            
            logger.debug(f"Uploading image to Supabase: {storage_path}")
            
            # Upload to Supabase storage
            # Supabase storage client expects a file path (it calls open() internally)
            # So we need to write bytes to a temporary file first
            tmp_file_path = None
            try:
                # Create temporary file with proper extension
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_ext}") as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_file_path = tmp_file.name
                
                # Upload using temporary file path
                result = self.client.storage.from_(self.BUCKET_NAME).upload(
                    path=storage_path,
                    file=tmp_file_path,
                    file_options={
                        "content-type": self._get_content_type(image_ext),
                        "upsert": "false",  # Don't overwrite existing files
                    },
                )
            finally:
                # Always clean up temporary file
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file {tmp_file_path}: {cleanup_error}")
            
            # Check for errors
            if hasattr(result, 'error') and result.error:
                raise StorageError(
                    f"Failed to upload image to Supabase: {result.error}",
                    {"storage_path": storage_path, "error": result.error},
                )
            
            logger.info(f"✓ Uploaded image to Supabase: {storage_path}")
            return storage_path
            
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            logger.error(f"Error uploading image to Supabase: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to upload image to Supabase: {str(e)}",
                {"document_id": str(document_id), "image_index": image_index, "error": str(e)},
            ) from e
    
    def download_image(self, storage_path: str) -> bytes:
        """
        Download image from Supabase storage.
        
        Args:
            storage_path: Storage path of the image (e.g., "doc_id/image_1.png")
        
        Returns:
            Image bytes
        
        Raises:
            StorageError: If download fails
        """
        try:
            response = self.client.storage.from_(self.BUCKET_NAME).download(storage_path)
            
            if hasattr(response, 'error') and response.error:
                raise StorageError(
                    f"Failed to download image from Supabase: {response.error}",
                    {"storage_path": storage_path, "error": response.error},
                )
            
            # Supabase download returns bytes
            if isinstance(response, bytes):
                return response
            else:
                # If it's a response object, read the content
                return response.content if hasattr(response, 'content') else bytes(response)
        
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            logger.error(f"Error downloading image from Supabase: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to download image from Supabase: {str(e)}",
                {"storage_path": storage_path, "error": str(e)},
            ) from e
    
    def get_image_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """
        Get signed URL for image (temporary access).
        
        Args:
            storage_path: Storage path (relative to bucket)
            expires_in: URL expiration time in seconds (default: 3600 = 1 hour)
        
        Returns:
            Signed URL for image access
        """
        try:
            result = self.client.storage.from_(self.BUCKET_NAME).create_signed_url(
                path=storage_path,
                expires_in=expires_in,
            )
            
            if hasattr(result, 'error') and result.error:
                raise StorageError(
                    f"Failed to create signed URL: {result.error}",
                    {"storage_path": storage_path, "error": result.error},
                )
            
            # Supabase returns a dict with 'signedURL' key
            if isinstance(result, dict):
                return result.get('signedURL', '')
            elif hasattr(result, 'signedURL'):
                return result.signedURL
            else:
                # Fallback: try to get URL directly
                return str(result)
                
        except Exception as e:
            logger.error(f"Error creating signed URL: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to create signed URL: {str(e)}",
                {"storage_path": storage_path, "error": str(e)},
            ) from e
    
    def delete_image(self, storage_path: str) -> bool:
        """
        Delete image from Supabase storage.
        
        Args:
            storage_path: Storage path (relative to bucket)
        
        Returns:
            True if successful
        
        Raises:
            StorageError: If deletion fails
        """
        try:
            result = self.client.storage.from_(self.BUCKET_NAME).remove([storage_path])
            
            if hasattr(result, 'error') and result.error:
                raise StorageError(
                    f"Failed to delete image from Supabase: {result.error}",
                    {"storage_path": storage_path, "error": result.error},
                )
            
            logger.info(f"✓ Deleted image from Supabase: {storage_path}")
            return True
            
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            logger.error(f"Error deleting image from Supabase: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to delete image from Supabase: {str(e)}",
                {"storage_path": storage_path, "error": str(e)},
            ) from e
    
    def delete_images_by_document(self, document_id: UUID) -> bool:
        """
        Delete all images for a document from Supabase storage.
        
        This deletes all images in the {document_id}/ directory.
        
        Args:
            document_id: Document UUID
        
        Returns:
            True if successful
        
        Raises:
            StorageError: If deletion fails
        """
        try:
            # List all files in the document's directory
            folder_path = str(document_id)
            files_response = self.client.storage.from_(self.BUCKET_NAME).list(folder_path)
            
            # Handle different response formats from Supabase
            if hasattr(files_response, 'data'):
                files = files_response.data
            elif isinstance(files_response, list):
                files = files_response
            else:
                files = []
            
            if not files:
                logger.debug(f"No images found for document {document_id}")
                return True
            
            # Extract file paths - handle both dict and object formats
            image_paths = []
            for file in files:
                if isinstance(file, dict):
                    file_name = file.get('name', '')
                elif hasattr(file, 'name'):
                    file_name = file.name
                else:
                    continue
                
                if file_name:
                    image_paths.append(f"{folder_path}/{file_name}")
            
            if not image_paths:
                logger.debug(f"No image paths found for document {document_id}")
                return True
            
            # Delete all images
            result = self.client.storage.from_(self.BUCKET_NAME).remove(image_paths)
            
            if hasattr(result, 'error') and result.error:
                raise StorageError(
                    f"Failed to delete images from Supabase: {result.error}",
                    {"document_id": str(document_id), "error": result.error},
                )
            
            logger.info(f"✓ Deleted {len(image_paths)} image(s) from Supabase for document: {document_id}")
            return True
            
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            logger.error(f"Error deleting images from Supabase: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to delete images from Supabase: {str(e)}",
                {"document_id": str(document_id), "error": str(e)},
            ) from e
    
    def _get_content_type(self, image_ext: str) -> str:
        """Get MIME type for image extension."""
        content_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "svg": "image/svg+xml",
        }
        return content_types.get(image_ext.lower(), "image/png")

