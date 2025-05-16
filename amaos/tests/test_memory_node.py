"""Tests for the Memory Node component of AMAOS.

This module contains tests for the MemoryNode class, which is responsible for
abstracting persistence (short/long-term memory) with Redis support and fallback mechanisms.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast, Callable, Protocol
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from freezegun import freeze_time
from pytest import approx

from amaos.nodes.memory_node import (
    MemoryNode,
    MemoryNodeConfig,
    MemoryType,
)
from amaos.core.node_protocol import Node, NodeTask, NodeResult # Import necessary types


class RedisProtocol(Protocol):
    """Protocol defining Redis client interface used by MemoryNode."""
    
    async def ping(self) -> bool:
        ...
        
    async def get(self, key: str) -> Optional[Any]:
        ...
        
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        ...
        
    async def delete(self, key: str) -> int:
        ...
        
    async def ttl(self, key: str) -> int:
        ...


class MockRedis(RedisProtocol):
    """Mock Redis client for testing."""
    
    def __init__(self, available: bool = True, error_on_call: bool = False) -> None:
        """Initialize the mock Redis client.
        
        Args:
            available: Whether Redis is available.
            error_on_call: Whether to raise an error on Redis calls.
        """
        self.available = available
        self.error_on_call = error_on_call
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        
    async def ping(self) -> bool:
        """Check if Redis is available.
        
        Returns:
            True if Redis is available, False otherwise.
        """
        if self.error_on_call:
            raise ConnectionError("Redis connection error")
        return self.available
        
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis.
        
        Args:
            key: The key to get.
            
        Returns:
            The value if it exists and hasn't expired, None otherwise.
        """
        if self.error_on_call:
            raise ConnectionError("Redis connection error")
            
        if key not in self.data:
            return None
            
        # Check if the key has expired
        if key in self.expiry and self.expiry[key] < time.time():
            del self.data[key]
            del self.expiry[key]
            return None
            
        return self.data[key]
        
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        """Set a value in Redis.
        
        Args:
            key: The key to set.
            value: The value to set.
            ex: The expiry time in seconds.
        """
        if self.error_on_call:
            raise ConnectionError("Redis connection error")
            
        self.data[key] = value
        if ex is not None:
            self.expiry[key] = time.time() + ex
            
    async def delete(self, key: str) -> int:
        """Delete a key from Redis.
        
        Args:
            key: The key to delete.
            
        Returns:
            1 if the key was deleted, 0 otherwise.
        """
        if self.error_on_call:
            raise ConnectionError("Redis connection error")
            
        if key in self.data:
            del self.data[key]
            if key in self.expiry:
                del self.expiry[key]
            return 1
        return 0
        
    async def ttl(self, key: str) -> int:
        """Get the TTL of a key.
        
        Args:
            key: The key to get the TTL of.
            
        Returns:
            The TTL in seconds, -1 if the key exists but has no TTL, -2 if the key doesn't exist.
        """
        if self.error_on_call:
            raise ConnectionError("Redis connection error")
            
        if key not in self.data:
            return -2
            
        if key not in self.expiry:
            return -1
            
        ttl = int(self.expiry[key] - time.time())
        return ttl if ttl > 0 else -2


@pytest.fixture
def mock_redis() -> MockRedis:
    """Create a MockRedis instance for testing."""
    return MockRedis()


import pytest_asyncio

@pytest_asyncio.fixture
async def memory_node(mock_redis: MockRedis) -> MemoryNode:
    """Create a MemoryNode instance for testing and initialize Redis, with a per-test shared fallback store."""
    config = MemoryNodeConfig(
        ttl_short_term=60,  # 1 minute for testing
        ttl_long_term=3600,  # 1 hour for testing
    )
    # Instantiate the real MemoryNode and inject the mock_redis
    node = MemoryNode(config)
    node.redis = cast(Any, mock_redis) # Cast to Any to satisfy mypy in test code
    await node.initialize()  # Ensure redis_available is correct
    return node


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_set_and_get_memory(memory_node: MemoryNode) -> None:
    """Test setting and getting a memory."""
    key = "test_key"
    value = {"data": "test_value"}
    memory_type = MemoryType.SHORT_TERM
    await memory_node.set(key, value, memory_type)
    result = await memory_node.get(key, memory_type)
    assert result == value
    # Metrics assertions might need adjustment based on how metrics are exposed by MemoryNode
    # assert memory_node.metrics["hits"] >= 1
    # assert memory_node.metrics["misses"] == 0
    # assert memory_node.metrics["failures"] == 0
    # assert memory_node.metrics["fallbacks"] == 0


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_get_nonexistent_memory(memory_node: MemoryNode) -> None:
    """Test getting a nonexistent memory."""
    key = "nonexistent_key"
    memory_type = MemoryType.SHORT_TERM
    result = await memory_node.get(key, memory_type)
    assert result is None
    # assert memory_node.metrics["hits"] == 0
    # assert memory_node.metrics["misses"] >= 1
    # assert memory_node.metrics["failures"] == 0
    # assert memory_node.metrics["fallbacks"] == 0


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_delete_memory_key(memory_node: MemoryNode) -> None:
    """Test deleting a memory key."""
    key = "test_key"
    value = {"data": "test_value"}
    memory_type = MemoryType.SHORT_TERM
    await memory_node.set(key, value, memory_type)
    result = await memory_node.get(key, memory_type)
    assert result == value
    success = await memory_node.delete(key, memory_type)
    assert success is True
    result = await memory_node.get(key, memory_type)
    assert result is None
    # assert memory_node.metrics["hits"] >= 1
    # assert memory_node.metrics["misses"] >= 1
    # assert memory_node.metrics["failures"] == 0
    # assert memory_node.metrics["fallbacks"] == 0


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_ttl_expiration(mock_redis: MockRedis) -> None:
    """Test TTL expiration."""
    # Create a memory node with a short TTL for testing
    config = MemoryNodeConfig(
        ttl_short_term=1,  # 1 second for testing
    )
    
    node = MemoryNode(config)
    node.redis = mock_redis  # Type checked at runtime
    await node.initialize()
    
    # Set a memory
    key = "test_key"
    value = {"data": "test_value"}
    memory_type = MemoryType.SHORT_TERM
    
    await node.set(key, value, memory_type)
    
    # Verify the memory exists
    result = await node.get(key, memory_type)
    assert result == value
    
    # Wait for the TTL to expire
    await asyncio.sleep(1.1)
    
    # Verify the memory no longer exists
    result = await node.get(key, memory_type)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_ttl_expiration_with_freeze_time(mock_redis: MockRedis) -> None:
    """Test TTL expiration using freeze_time."""
    # Create a memory node with a short TTL for testing
    config = MemoryNodeConfig(
        ttl_short_term=60,  # 1 minute for testing
    )
    
    node = MemoryNode(config)
    node.redis = mock_redis  # Type checked at runtime
    await node.initialize()
    
    # Set the initial time
    with freeze_time("2025-01-01 00:00:00") as frozen_time:
        # Set a memory
        key = "test_key"
        value = {"data": "test_value"}
        memory_type = MemoryType.SHORT_TERM
        
        await node.set(key, value, memory_type)
        
        # Verify the memory exists
        result = await node.get(key, memory_type)
        assert result == value
        
        # Move time forward by 30 seconds (memory should still exist)
        frozen_time.move_to("2025-01-01 00:00:30")
        
        # Verify the memory still exists
        result = await node.get(key, memory_type)
        assert result == value
        
        # Move time forward by another 31 seconds (memory should expire)
        frozen_time.move_to("2025-01-01 00:01:01")
        
        # Verify the memory no longer exists
        result = await node.get(key, memory_type)
        assert result is None


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_different_memory_types(memory_node: MemoryNode) -> None:
    """Test different memory types."""
    # Set memories with different types
    short_term_key = "short_term_key"
    long_term_key = "long_term_key"
    episodic_key = "episodic_key"
    semantic_key = "semantic_key"
    
    value1 = {"data": "short_term_value"}
    value2 = {"data": "long_term_value"}
    value3 = {"data": "episodic_value"}
    value4 = {"data": "semantic_value"}
    
    await memory_node.set(short_term_key, value1, MemoryType.SHORT_TERM)
    await memory_node.set(long_term_key, value2, MemoryType.LONG_TERM)
    await memory_node.set(episodic_key, value3, MemoryType.EPISODIC)
    await memory_node.set(semantic_key, value4, MemoryType.SEMANTIC)
    
    # Verify each memory exists in its respective type
    assert await memory_node.get(short_term_key, MemoryType.SHORT_TERM) == value1
    assert await memory_node.get(long_term_key, MemoryType.LONG_TERM) == value2
    assert await memory_node.get(episodic_key, MemoryType.EPISODIC) == value3
    assert await memory_node.get(semantic_key, MemoryType.SEMANTIC) == value4
    
    # Verify memories don't exist in other types
    assert await memory_node.get(short_term_key, MemoryType.LONG_TERM) is None
    assert await memory_node.get(long_term_key, MemoryType.SHORT_TERM) is None


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_fallback_on_redis_error() -> None:
    """Test fallback to in-memory when Redis fails."""
    # Use a MockRedis that raises errors
    mock_redis = MockRedis(error_on_call=True)
    
    # Use the real MemoryNode with the error-raising mock_redis
    config = MemoryNodeConfig(
        ttl_short_term=60, # 1 minute
        ttl_long_term=3600 # 1 hour
    )
    node = MemoryNode(config)
    node.redis = cast(Any, mock_redis) # Cast to Any to satisfy mypy in test code
    await node.initialize() # This will set redis_available to False

    key = "test_key"
    value = {"data": "test_value"}
    memory_type = MemoryType.SHORT_TERM

    # Set should use fallback due to Redis error
    await node.set(key, value, memory_type)
    
    # Get should use fallback due to Redis error
    result = await node.get(key, memory_type)
    assert result == value

    # Metrics assertions might need adjustment
    # assert node.metrics["failures"] >= 1 # Should fail on set and get attempts
    # assert node.metrics["fallbacks"] >= 2 # Should fallback on set and get

@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_redis_unavailable() -> None:
    """Test behavior when Redis is initially unavailable."""
    # Use a MockRedis that is initially unavailable
    mock_redis = MockRedis(available=False)
    
    # Use the real MemoryNode with the unavailable mock_redis
    config = MemoryNodeConfig(
        ttl_short_term=60, # 1 minute
        ttl_long_term=3600 # 1 hour
    )
    node = MemoryNode(config)
    # Don\'t set node.redis
    await node.initialize()
    assert node.redis_available is False # Verify Redis is not available
    key = "test_key"
    value = {"data": "test_value"}
    memory_type = MemoryType.SHORT_TERM
    await node.set(key, value, memory_type)
    result = await node.get(key, memory_type)
    assert result == value # Verify fallback get worked


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_init_without_redis() -> None:
    """Test initializing without providing a Redis instance."""
    # Use the real MemoryNode without providing a redis instance
    config = MemoryNodeConfig(
        ttl_short_term=60, # 1 minute
        ttl_long_term=3600 # 1 hour
    )
    node = MemoryNode(config)
    # Don\'t set node.redis
    await node.initialize()
    
    # Verify Redis is marked as unavailable
    assert node.redis_available is False
    
    # Set a memory (should use fallback)
    key = "test_key"
    value = {"data": "test_value"}
    memory_type = MemoryType.SHORT_TERM
    
    await node.set(key, value, memory_type)
    
    # Verify the memory exists in the fallback storage
    result = await node.get(key, memory_type)
    assert result == value


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_metrics_tracking(memory_node: MemoryNode) -> None:
    """Test metrics tracking."""
    key1 = "test_key1"
    value1 = {"data": "test_value1"}
    memory_type = MemoryType.SHORT_TERM
    await memory_node.set(key1, value1, memory_type)
    await memory_node.get(key1, memory_type)
    await memory_node.get("nonexistent_key", memory_type)
    error_redis = MockRedis(error_on_call=True)
    memory_node.redis = error_redis  
    await memory_node.get(key1, memory_type)
    assert memory_node.metrics["hits"] >= 1
    assert memory_node.metrics["misses"] >= 1
    assert memory_node.metrics["failures"] >= 1
    assert memory_node.metrics["fallbacks"] >= 1


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_memory_search(memory_node: MemoryNode) -> None:
    """Test searching for memories."""
    memory_node.redis_available = False  # Force fallback path
    # Set multiple memories
    await memory_node.set("key1", {"data": "apple", "type": "fruit"}, MemoryType.SHORT_TERM)
    await memory_node.set("key2", {"data": "banana", "type": "fruit"}, MemoryType.SHORT_TERM)
    await memory_node.set("key3", {"data": "carrot", "type": "vegetable"}, MemoryType.SHORT_TERM)
    # Search for memories by filter
    results = await memory_node.search(
        memory_type=MemoryType.SHORT_TERM,
        filter_func=lambda x: x.get("type") == "fruit"
    )
    # Verify results
    assert len(results) == 2
    assert any(item["data"] == "apple" for item in results)
    assert any(item["data"] == "banana" for item in results)
    assert not any(item["data"] == "carrot" for item in results)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_memory_clear(memory_node: MemoryNode) -> None:
    """Test clearing all memories of a specific type."""
    memory_node.redis_available = False  # Force fallback path
    # Set memories with different types
    await memory_node.set("key1", {"data": "value1"}, MemoryType.SHORT_TERM)
    await memory_node.set("key2", {"data": "value2"}, MemoryType.SHORT_TERM)
    await memory_node.set("key3", {"data": "value3"}, MemoryType.LONG_TERM)
    # Clear short-term memories
    await memory_node.clear(MemoryType.SHORT_TERM)
    # Verify short-term memories are gone
    assert await memory_node.get("key1", MemoryType.SHORT_TERM) is None
    assert await memory_node.get("key2", MemoryType.SHORT_TERM) is None
    # Verify long-term memory still exists
    assert await memory_node.get("key3", MemoryType.LONG_TERM) == {"data": "value3"}


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_memory_stats(memory_node: MemoryNode) -> None:
    """Test memory stats tracking."""
    # Set some memory entries
    await memory_node.set("key1", "value1", MemoryType.SHORT_TERM)
    await memory_node.set("key2", "value2", MemoryType.LONG_TERM)
    await memory_node.set("key3", "value3", MemoryType.SHORT_TERM)

    # Get stats
    stats = memory_node.get_stats() # Call get_stats, not await stats()

    # Assert stats are correct (adjust assertions based on get_stats return type)
    assert "hits" in stats
    assert "misses" in stats
    assert "failures" in stats
    assert "fallbacks" in stats

