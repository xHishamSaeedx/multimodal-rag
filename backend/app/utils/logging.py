"""
Structured logging configuration with structlog.

This module configures structlog for JSON output with correlation IDs
and context enrichment for request tracing.
"""
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict

import structlog
from structlog.types import EventDict, Processor

# Context variable for correlation ID (request ID)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Processor to add correlation ID to log events.
    
    Args:
        logger: The logger instance
        method_name: The logging method name (info, error, etc.)
        event_dict: The event dictionary
        
    Returns:
        Event dictionary with correlation_id added
    """
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    include_timestamp: bool = True,
) -> None:
    """
    Configure structlog for structured JSON logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Whether to output JSON format (True) or human-readable (False)
        include_timestamp: Whether to include timestamps in logs
    """
    # Shared processors that run for all log entries
    shared_processors: list[Processor] = [
        # Add log level
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add correlation ID from context
        add_correlation_id,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso") if include_timestamp else structlog.processors.TimeStamper(),
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exceptions
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        # JSON output for production/container environments
        processors = shared_processors + [
            # Remove color codes
            structlog.processors.UnicodeDecoder(),
            # Output as JSON
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable output for development
        processors = shared_processors + [
            # Remove color codes
            structlog.processors.UnicodeDecoder(),
            # Pretty print with colors
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    import logging
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured structlog logger instance.
    
    Args:
        name: Logger name (typically __name__). If None, uses the calling module.
        
    Returns:
        Configured structlog logger
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "app")
    
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID for the current context.
    
    This is typically called at the start of a request to set a unique
    request ID that will be included in all log entries for that request.
    
    Args:
        correlation_id: Unique identifier for the request/operation
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    """
    Get the current correlation ID from context.
    
    Returns:
        Current correlation ID, or empty string if not set
    """
    return correlation_id_var.get()


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID (UUID).
    
    Returns:
        A new UUID string
    """
    return str(uuid.uuid4())


# Initialize logging on module import
# Default to JSON output, can be overridden by calling configure_logging()
configure_logging(log_level="INFO", json_output=True)
