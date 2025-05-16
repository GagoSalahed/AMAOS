"""Memory interface for AMAOS.

This module defines the interfaces for memory management in the AMAOS system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union, runtime_checkable

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Enum representing the types of memory."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryEntry(BaseModel):
    """Model representing a memory entry."""

    key: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_type: MemoryType = MemoryType.SHORT_TERM
    ttl: Optional[int] = None  # Time to live in seconds, None for no expiration


class MemoryQueryResult(BaseModel):
    """Model representing a memory query result."""

    entries: List[MemoryEntry]
    total: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryStats(BaseModel):
    """Model representing memory statistics."""

    total_entries: int
    entries_by_type: Dict[MemoryType, int]
    size_bytes: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class MemoryInterface(Protocol):
    """Protocol defining the interface for memory management."""

    async def set(self, key: str, value: Any, **kwargs: Any) -> None:
        """Set a value in memory.
        
        Args:
            key: Key to store the value under.
            value: Value to store.
            **kwargs: Additional arguments.
        """
        ...

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from memory.
        
        Args:
            key: Key to retrieve.
            default: Default value to return if key not found.
            
        Returns:
            The value associated with the key, or the default value if not found.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a value from memory.
        
        Args:
            key: Key to delete.
            
        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory.
        
        Args:
            key: Key to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        ...

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern.
        
        Args:
            pattern: Pattern to match keys against.
            
        Returns:
            List of matching keys.
        """
        ...

    async def query(self, query: str, **kwargs: Any) -> MemoryQueryResult:
        """Query memory.
        
        Args:
            query: Query string.
            **kwargs: Additional arguments.
            
        Returns:
            Query result.
        """
        ...

    async def stats(self) -> MemoryStats:
        """Get memory statistics.
        
        Returns:
            Memory statistics.
        """
        ...


class BaseMemoryInterface(ABC):
    """Abstract base class for memory interfaces."""

    @abstractmethod
    async def set(self, key: str, value: Any, **kwargs: Any) -> None:
        """Set a value in memory.
        
        Args:
            key: Key to store the value under.
            value: Value to store.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from memory.
        
        Args:
            key: Key to retrieve.
            default: Default value to return if key not found.
            
        Returns:
            The value associated with the key, or the default value if not found.
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from memory.
        
        Args:
            key: Key to delete.
            
        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory.
        
        Args:
            key: Key to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern.
        
        Args:
            pattern: Pattern to match keys against.
            
        Returns:
            List of matching keys.
        """
        pass

    @abstractmethod
    async def query(self, query: str, **kwargs: Any) -> MemoryQueryResult:
        """Query memory.
        
        Args:
            query: Query string.
            **kwargs: Additional arguments.
            
        Returns:
            Query result.
        """
        pass

    @abstractmethod
    async def stats(self) -> MemoryStats:
        """Get memory statistics.
        
        Returns:
            Memory statistics.
        """
        pass
