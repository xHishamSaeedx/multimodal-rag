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
)
from app.utils.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_request_size_bytes,
    http_response_size_bytes,
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
        request_body_size = 0
        
        # Get request body size if available
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    request_body_size = int(content_length)
                except ValueError:
                    request_body_size = 0
        
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
            
            # Get endpoint path (normalize)
            endpoint = request.url.path
            method = request.method
            status_code = response.status_code
            
            # Record HTTP metrics
            http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_body_size)
            
            # Get response body size
            response_body_size = int(response.headers.get("content-length", 0))
            http_response_size_bytes.labels(method=method, endpoint=endpoint).observe(response_body_size)
            
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
            
            # Record error metrics
            endpoint = request.url.path
            method = request.method
            status_code = 500  # Assume 500 for unhandled exceptions
            http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_body_size)
            
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

