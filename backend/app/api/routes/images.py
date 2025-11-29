"""
Image endpoints.

GET /api/v1/images/{image_path}/url - Get signed URL for an image
"""
from fastapi import APIRouter, HTTPException, status, Query, Path as FastAPIPath
from typing import Optional
from urllib.parse import unquote

from app.api.schemas import ErrorResponse
from app.services.storage.supabase_storage import SupabaseImageStorage, StorageError
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


@router.get(
    "/{image_path:path}/url",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_image_url(
    image_path: str = FastAPIPath(
        ...,
        description="Storage path of the image (relative to bucket)",
    ),
    expires_in: int = Query(
        default=3600,
        ge=60,
        le=86400,
        description="URL expiration time in seconds (default: 3600 = 1 hour, max: 86400 = 24 hours)",
    ),
) -> dict:
    """
    Get a signed URL for an image stored in Supabase storage.
    
    This endpoint generates a temporary signed URL that can be used to access
    the image. The URL expires after the specified time.
    
    Args:
        image_path: Storage path of the image (relative to bucket)
        expires_in: URL expiration time in seconds (60-86400)
    
    Returns:
        Dictionary containing:
            - image_url: Signed URL for the image
            - expires_in: Expiration time in seconds
            - image_path: Storage path of the image
    
    Raises:
        HTTPException: If image path is invalid or URL generation fails
    """
    try:
        # URL decode the image path
        image_path = unquote(image_path)
        
        logger.debug(
            "image_url_request",
            image_path=image_path,
            expires_in=expires_in,
        )
        
        # Initialize storage service
        image_storage = SupabaseImageStorage()
        
        # Generate signed URL
        try:
            image_url = image_storage.get_image_url(image_path, expires_in=expires_in)
            
            logger.info(
                "image_url_generated",
                image_path=image_path,
                expires_in=expires_in,
            )
            
            return {
                "success": True,
                "image_url": image_url,
                "expires_in": expires_in,
                "image_path": image_path,
            }
            
        except StorageError as e:
            logger.warning(
                "image_url_generation_failed",
                image_path=image_path,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "success": False,
                    "error": f"Failed to generate image URL: {str(e)}",
                    "details": {"image_path": image_path},
                },
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "image_url_unexpected_error",
            error_type=type(e).__name__,
            error_message=str(e),
            image_path=image_path if 'image_path' in locals() else None,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": f"Unexpected error during image URL generation: {str(e)}",
                "details": {
                    "image_path": image_path if 'image_path' in locals() else None,
                    "error_type": type(e).__name__,
                },
            },
        )
