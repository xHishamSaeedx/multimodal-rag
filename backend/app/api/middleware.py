"""
FastAPI middleware for request correlation IDs.

This middleware generates and sets a correlation ID for each request,
which is then included in all log entries for that request.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.logging import (
    get_logger,
    set_correlation_id,
    generate_correlation_id,
    get_correlation_id,
)

logger = get_logger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add correlation IDs to requests.
    
    Generates a unique correlation ID for each request and:
    1. Sets it in the context for logging
    2. Adds it to response headers (X-Correlation-ID)
    3. Logs request/response information
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
        set_correlation_id(correlation_id)
        
        # Log request start
        start_time = time.time()
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            # Log request completion
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=round(duration, 3),
            )
            
            return response
            
        except Exception as e:
            # Calculate duration even on error
            duration = time.time() - start_time
            
            # Log request error
            logger.error(
                "request_error",
                method=request.method,
                path=request.url.path,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_seconds=round(duration, 3),
                exc_info=True,
            )
            
            # Re-raise the exception
            raise

