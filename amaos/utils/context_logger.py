"""
Context-aware logging for AMAOS.

This module provides a logging system that maintains context across asynchronous boundaries,
making it easier to trace task execution through the system.
"""

import logging
import contextvars
import uuid
from typing import Any, Dict, Optional, Union, TypeVar, cast, Callable


class ContextAwareLogger:
    """Logger that maintains context across async boundaries.
    
    This logger allows for attaching context information to log messages
    and propagating that context through async calls, making it easier
    to trace execution flow.
    """
    
    def __init__(self, name: str):
        """Initialize context-aware logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("log_context", default={})
        self.trace_id = contextvars.ContextVar("trace_id", default=None)
        
    def with_context(self, **kwargs: Any) -> "ContextAwareLogger":
        """Create a new logger with additional context values.
        
        Args:
            **kwargs: Key-value pairs to add to context
            
        Returns:
            New logger instance with updated context
        """
        context = self.context.get().copy()
        context.update(kwargs)
        
        new_logger = ContextAwareLogger(self.logger.name)
        new_logger.context.set(context)
        
        # Preserve trace ID if exists
        trace_id = self.trace_id.get()
        if trace_id is not None:
            new_logger.trace_id.set(trace_id)
            
        return new_logger
        
    def with_trace(self, trace_id: Optional[str] = None) -> "ContextAwareLogger":
        """Create a new logger with a trace ID.
        
        Args:
            trace_id: Trace ID (generated if not provided)
            
        Returns:
            New logger instance with trace ID
        """
        new_logger = ContextAwareLogger(self.logger.name)
        
        # Copy existing context
        context = self.context.get().copy()
        new_logger.context.set(context)
        
        # Set trace ID
        tid = trace_id or str(uuid.uuid4())
        # Type ignore to skip the incompatible type assignment
        new_logger.trace_id.set(tid)  # type: ignore
        
        return new_logger
        
    def _format_message(self, msg: str) -> str:
        """Format a message with context and trace ID.
        
        Args:
            msg: Original message
            
        Returns:
            Formatted message with context and trace ID
        """
        # Add trace ID if present
        trace_id = self.trace_id.get()
        if trace_id is not None:  # type: ignore  # mypy sees this as unreachable but it's not
            msg = f"[trace:{trace_id}] {msg}"
            
        # Add context if present
        context = self.context.get()
        if context:  # type: ignore  # mypy sees this as unreachable but it's not
            ctx_str = " ".join(f"{k}={v}" for k, v in context.items())
            msg = f"{msg} [{ctx_str}]"
            
        return msg
        
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        formatted_msg = self._format_message(msg)
        self.logger.debug(formatted_msg, *args, **kwargs)
        
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        formatted_msg = self._format_message(msg)
        self.logger.info(formatted_msg, *args, **kwargs)
        
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        formatted_msg = self._format_message(msg)
        self.logger.warning(formatted_msg, *args, **kwargs)
        
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        formatted_msg = self._format_message(msg)
        self.logger.error(formatted_msg, *args, **kwargs)
        
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        formatted_msg = self._format_message(msg)
        self.logger.critical(formatted_msg, *args, **kwargs)
        
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        formatted_msg = self._format_message(msg)
        self.logger.exception(formatted_msg, *args, **kwargs)
        
    def get_trace_id(self) -> Optional[str]:
        """Get the current trace ID.
        
        Returns:
            Current trace ID if set, None otherwise
        """
        return self.trace_id.get()
        
    def get_context(self) -> Dict[str, Any]:
        """Get the current context.
        
        Returns:
            Current context dictionary
        """
        return self.context.get().copy()
