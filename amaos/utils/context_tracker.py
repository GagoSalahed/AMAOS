"""
Context management utilities for AMAOS.

This module provides utilities for tracking context across asynchronous boundaries,
making it easier to trace task execution through the system.
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, cast

# Context variable for storing context information
# Type ignore to bypass mypy error about lack of annotation 
_context_var = contextvars.ContextVar("context_tracker", default={})  # type: ignore


class ContextTracker:
    """A utility class for tracking context across async boundaries.
    
    This class provides static methods for managing context information
    that persists across async calls, making it easier to trace
    execution flow in complex asynchronous systems.
    """
    
    @staticmethod
    @contextmanager
    def context(new_context: Dict[str, Any]) -> Iterator[None]:
        """Context manager for temporarily setting context values.
        
        Args:
            new_context: Key-value pairs to add to the current context
            
        Yields:
            None, used for 'with' block
        """
        # Get current context
        current = _context_var.get().copy()
        
        # Update with new context
        current.update(new_context)
        
        # Set the context
        token = _context_var.set(current)
        try:
            yield
        finally:
            # Restore previous context
            _context_var.reset(token)
    
    @staticmethod
    def get_current() -> Dict[str, Any]:
        """Get the current context.
        
        Returns:
            The current context dictionary
        """
        return _context_var.get().copy()
    
    @staticmethod
    def set_value(key: str, value: Any) -> None:
        """Set a single context value.
        
        Args:
            key: Context key
            value: Context value
        """
        current = _context_var.get().copy()
        current[key] = value
        _context_var.set(current)
    
    @staticmethod
    def get_value(key: str, default: Any = None) -> Any:
        """Get a single context value.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            The context value or default
        """
        return _context_var.get().get(key, default)
