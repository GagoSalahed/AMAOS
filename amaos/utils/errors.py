"""Error handling utilities for AMAOS.

This module provides standardized error classes and handling utilities for AMAOS.
"""

import logging
import traceback
from enum import Enum
from typing import Any, Dict, Optional, Type, Union, Callable

from pydantic import BaseModel, Field


class ErrorSeverity(str, Enum):
    """Severity levels for AMAOS errors."""
    
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCode(str, Enum):
    """Standard error codes for AMAOS system."""
    
    # General errors
    UNKNOWN = "unknown_error"
    VALIDATION = "validation_error"
    TIMEOUT = "timeout_error"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    
    # Node-related errors
    NODE_INITIALIZATION = "node_initialization_error"
    NODE_UNAVAILABLE = "node_unavailable"
    NODE_COMMUNICATION = "node_communication_error"
    
    # Memory-related errors
    MEMORY_STORAGE = "memory_storage_error"
    MEMORY_RETRIEVAL = "memory_retrieval_error"
    
    # LLM-related errors
    LLM_API_ERROR = "llm_api_error"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_CONTEXT_OVERFLOW = "llm_context_overflow"
    LLM_CONTENT_FILTER = "llm_content_filter"
    
    # Tool-related errors
    TOOL_EXECUTION = "tool_execution_error"
    TOOL_NOT_FOUND = "tool_not_found"
    
    # Config-related errors
    CONFIG_INVALID = "config_invalid"
    CONFIG_MISSING = "config_missing"
    
    # Authentication/API errors
    AUTH_FAILURE = "authentication_failure"
    API_ERROR = "api_error"


class ErrorContext(BaseModel):
    """Context information for an error."""
    
    source: str = Field(description="Component that originated the error")
    timestamp: str = Field(description="ISO-formatted timestamp when error occurred")
    correlation_id: Optional[str] = Field(default=None, description="For tracing related errors")
    request_id: Optional[str] = Field(default=None, description="ID of the request that caused the error")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary."""
        return self.model_dump()


class AMAOSError(Exception):
    """Base exception class for AMAOS-specific errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Union[ErrorCode, str] = ErrorCode.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        """Initialize the error with context.
        
        Args:
            message: Human-readable error message
            error_code: Error code identifying the type of error
            severity: Severity level of the error
            context: Additional context for the error
            original_exception: Original exception if this is a wrapped exception
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code if isinstance(error_code, ErrorCode) else error_code
        self.severity = severity
        self.context = context
        self.original_exception = original_exception
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation.
        
        Returns:
            Dictionary containing error details
        """
        result: Dict[str, Any] = {
            "error_code": self.error_code.name if isinstance(self.error_code, ErrorCode) else str(self.error_code),
            "message": self.message,
        }
        
        if self.context is not None:
            # Convert context to dict for proper type compatibility
            # Handle the context serialization in a type-safe way
            if isinstance(self.context, ErrorContext):
                result["context"] = self.context.to_dict()  # Use the dedicated method
            elif isinstance(self.context, dict):  # type: ignore[unreachable]
                result["context"] = self.context
            elif hasattr(self.context, 'model_dump'):
                result["context"] = self.context.model_dump()
            else:
                result["context"] = {"value": str(self.context)}  # This is likely unreachable with modern pydantic
            
        if self.original_exception:
            result["original_error"] = str(self.original_exception)
            # Traceback returns a list of strings, but we need a single string
            result["traceback"] = "".join(traceback.format_exception(
                type(self.original_exception), 
                self.original_exception,
                self.original_exception.__traceback__
            ))
            
        return result
    
    def log(self, logger: Optional[logging.Logger] = None) -> None:
        """Log the error with appropriate severity.
        
        Args:
            logger: Logger to use; if None, creates a new logger
        """
        if logger is None:
            logger = logging.getLogger("amaos.error")
            
        log_method = getattr(logger, self.severity.value, logger.error)
        
        # Log the basic error info
        log_method(f"{self.error_code}: {self.message}")
        
        # Log additional context if available
        if self.context:
            log_method(f"Error context: {self.context.model_dump()}")
            
        # Log original exception details if available
        if self.original_exception:
            log_method(f"Original exception: {self.original_exception}")
            log_method("".join(traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            )))
            

class ValidationError(AMAOSError):
    """Error raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION,
            severity=ErrorSeverity.WARNING,
            context=context,
            original_exception=original_exception,
        )


class TimeoutError(AMAOSError):
    """Error raised when an operation times out."""
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            context=context,
            original_exception=original_exception,
        )


class NotFoundError(AMAOSError):
    """Error raised when a requested resource is not found."""
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.NOT_FOUND,
            severity=ErrorSeverity.WARNING,
            context=context,
            original_exception=original_exception,
        )


class NodeError(AMAOSError):
    """Base class for node-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Union[ErrorCode, str] = ErrorCode.NODE_UNAVAILABLE,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )


class MemoryError(AMAOSError):
    """Base class for memory-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Union[ErrorCode, str] = ErrorCode.MEMORY_STORAGE,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )


class LLMError(AMAOSError):
    """Base class for LLM-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Union[ErrorCode, str] = ErrorCode.LLM_API_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )


def handle_exceptions(
    func: Callable,
    error_map: Optional[Dict[Type[Exception], Type[AMAOSError]]] = None,
    default_error_class: Type[AMAOSError] = AMAOSError,
    logger: Optional[logging.Logger] = None,
) -> Callable:
    """Decorator for standardized exception handling.
    
    Args:
        func: Function to wrap
        error_map: Mapping from exception types to AMAOS error classes
        default_error_class: Default AMAOS error class to use for unmapped exceptions
        logger: Logger to use for logging errors
        
    Returns:
        Wrapped function that handles exceptions according to the provided mapping
    """
    error_map = error_map or {}
    
    if logger is None:
        logger = logging.getLogger("amaos.error_handler")
    
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except tuple(error_map.keys()) as e:
            error_class = error_map[type(e)]
            amaos_error = error_class(str(e), original_exception=e)
            amaos_error.log(logger)
            raise amaos_error
        except Exception as e:
            if isinstance(e, AMAOSError):
                e.log(logger)
                raise
            else:
                amaos_error = default_error_class(str(e), original_exception=e)
                amaos_error.log(logger)
                raise amaos_error
    
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except tuple(error_map.keys()) as e:
            error_class = error_map[type(e)]
            amaos_error = error_class(str(e), original_exception=e)
            amaos_error.log(logger)
            raise amaos_error
        except Exception as e:
            if isinstance(e, AMAOSError):
                e.log(logger)
                raise
            else:
                amaos_error = default_error_class(str(e), original_exception=e)
                amaos_error.log(logger)
                raise amaos_error
    
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper