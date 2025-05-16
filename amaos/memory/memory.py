"""Memory implementation for AMAOS.

This module provides a memory implementation using Redis.
"""

# mypy: disable-error-code="unreachable"

import asyncio
import json
import logging
import os
import pickle
import fnmatch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import prometheus_client as prom
from pydantic import BaseModel, Field

from amaos.memory.interface import (
    BaseMemoryInterface,
    MemoryEntry,
    MemoryQueryResult,
    MemoryStats,
    MemoryType,
)


class MemoryConfig(BaseModel):
    """Configuration for memory."""

    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    namespace: str = "amaos:"
    default_ttl: Optional[int] = None  # None means no expiration
    use_mock: bool = False  # Use mock Redis for testing


class Memory(BaseMemoryInterface):
    """Memory implementation using Redis."""

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        """Initialize the memory.
        
        Args:
            config: Configuration for the memory.
        """
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        self._redis = None
        self._mock_data: Dict[str, Tuple[Any, Dict[str, Any], Optional[int]]] = {}
        
        # Set up metrics
        self.metrics = self._setup_metrics()

    def _setup_metrics(self) -> Dict[str, Any]:
        """Set up metrics for memory.
        
        Returns:
            Dictionary of metrics.
        """
        metrics: Dict[str, Any] = {}
        
        # Operation count metrics
        metrics["operations"] = prom.Counter(
            "amaos_memory_operations",
            "Number of memory operations",
            ["operation", "status"]
        )
        
        # Size metrics
        metrics["size"] = prom.Gauge(
            "amaos_memory_size",
            "Size of memory in bytes"
        )
        
        # Entry count metrics
        metrics["entries"] = prom.Gauge(
            "amaos_memory_entries",
            "Number of entries in memory",
            ["memory_type"]
        )
        
        # Latency metrics
        metrics["latency"] = prom.Histogram(
            "amaos_memory_latency",
            "Latency of memory operations",
            ["operation"]
        )
        
        return metrics

    async def initialize(self) -> None:
        """Initialize the memory.
        
        Establishes connection to Redis or falls back to mock implementation.
        """
        # Track whether we need to use mock Redis
        use_mock = self.config.use_mock
        
        # Only try to connect to Redis if explicitly not using mock
        if not use_mock:
            # First check if Redis is available
            try:
                # Import Redis here to avoid dependency if not used
                import redis.asyncio as redis
                
                try:
                    self.logger.info(f"Connecting to Redis at {self.config.redis_url}")
                    
                    # Create Redis connection
                    self._redis = redis.from_url(
                        self.config.redis_url,
                        password=self.config.redis_password,
                        decode_responses=False,  # We'll handle decoding ourselves
                    )
                    
                    # Test connection
                    assert self._redis is not None
                    await self._redis.ping()  # type: ignore
                    self.logger.info("Connected to Redis")
                    # Successfully connected, no need to use mock
                    return
                    
                except Exception as e:
                    # Connection error
                    self.logger.error(f"Error connecting to Redis: {e}")
                    self.logger.warning("Falling back to mock Redis")
                    use_mock = True
                    self._redis = None
            
            except ImportError:
                # Redis package not available
                self.logger.warning("Redis not installed, falling back to mock Redis")
                use_mock = True
                self._redis = None
        
        # If we've reached here, we're using mock Redis
        self.logger.info("Using mock Redis implementation")
        # Initialize the mock data store
        self._mock_data = {}
        await self._setup_mock_redis()
        
        # Update the config with our final decision
        self.config.use_mock = use_mock
        
        if use_mock:
            self.logger.info("Using mock Redis implementation")

    async def _setup_mock_redis(self) -> None:
        """Set up mock Redis implementation."""
        # Add mock Redis setup here
        pass

    def _get_full_key(self, key: str) -> str:
        """Get the full key with namespace.
        
        Args:
            key: Key to get the full key for.
            
        Returns:
            Full key with namespace.
        """
        return f"{self.config.namespace}{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize a value for storage.
        
        Args:
            value: Value to serialize.
            
        Returns:
            Serialized value.
        """
        try:
            # Try to serialize as JSON first
            return json.dumps(value).encode("utf-8")
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)

    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize a value from storage.
        
        Args:
            value: Value to deserialize.
            
        Returns:
            Deserialized value.
        """
        if not value:
            return None
            
        try:
            # Try to deserialize as JSON first
            return json.loads(value.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle for complex objects
            try:
                # Fall back to pickle for complex objects
                return pickle.loads(value)
            except Exception as e:
                self.logger.error(f"Error deserializing value: {e}")
                return None

    async def safe_redis_operation(self, operation_name: str, fallback_value: Any, operation: Callable) -> Any:
        """Safely perform a Redis operation with proper error handling.
        
        Args:
            operation_name: Name of the operation for logging
            fallback_value: Value to return if operation fails
            operation: Async callable that performs the Redis operation
            
        Returns:
            Result of the operation or fallback value if it fails
        """
        if self.config.use_mock or self._redis is None:
            return fallback_value
            
        try:
            return await operation()
        except Exception as e:
            self.logger.error(f"Error in Redis {operation_name}: {e}")
            self.metrics["operations"].labels(
                operation=operation_name,
                status="redis_failure"
            ).inc()
            return fallback_value
            
    async def safe_get_fallback(self, key: str, default: Any = None) -> Optional[Any]:
        """Safely get a value from the fallback store.
        
        Args:
            key: Key to retrieve from fallback store
            default: Default value to return if key is not found or an error occurs
            
        Returns:
            The value from the fallback store or default if not found/error
        """
        try:
            if self._mock_data is not None and key in self._mock_data:
                value, _, ttl = self._mock_data[key]
                return value
            return default
        except Exception as e:
            self.logger.error(f"Error accessing fallback store for key {key}: {e}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Set a value in memory.
        
        Args:
            key: Key to store the value under.
            value: Value to store.
            memory_type: Type of memory.
            ttl: Time to live in seconds, None for no expiration.
            metadata: Additional metadata to store with the value.
            **kwargs: Additional arguments.
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.config.default_ttl
            
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add memory type to metadata
        metadata["memory_type"] = memory_type
        
        # Get full key with namespace
        full_key = self._get_full_key(key)
        
        with self.metrics["latency"].labels(operation="set").time():
            # Try Redis first if configured
            redis_success = False
            
            if not self.config.use_mock and self._redis is not None:
                # Define redis operation as a closure
                async def redis_set_operation() -> bool:
                    serialized_value = self._serialize_value(value)
                    if ttl is not None:
                        await self._redis.set(full_key, serialized_value, ex=ttl)
                    else:
                        await self._redis.set(full_key, serialized_value)
                        
                    # Store metadata
                    serialized_metadata = self._serialize_value(metadata)
                    if ttl is not None:
                        await self._redis.set(f"{full_key}:metadata", serialized_metadata, ex=ttl)
                    else:
                        await self._redis.set(f"{full_key}:metadata", serialized_metadata)
                    return True
                
                # Execute the operation safely
                redis_success = await self.safe_redis_operation("set", False, redis_set_operation)
                
                if redis_success:
                    self.metrics["operations"].labels(
                        operation="set",
                        status="success"
                    ).inc()
            
            # Use mock storage if Redis failed or not configured
            if not redis_success:
                # Store value, metadata, and TTL in mock data
                self._mock_data[full_key] = (value, metadata, ttl)
                
                status = "fallback" if not self.config.use_mock else "success"
                self.metrics["operations"].labels(
                    operation="set",
                    status=status
                ).inc()

    async def get(
        self, key: str, default: Any = None, include_metadata: bool = False
    ) -> Union[Any, Tuple[Any, Dict[str, Any]], None]:
        """Get a value from memory.
        
        Args:
            key: Key to retrieve.
            default: Default value to return if key not found.
            include_metadata: Whether to include metadata in the result.
            
        Returns:
            The value associated with the key, or the default value if not found.
            If include_metadata is True, returns a tuple of (value, metadata).
        """
        full_key = self._get_full_key(key)
        found_value = False
        value: Any = default
        metadata: Dict[str, Any] = {}
        status = "unknown"
        
        with self.metrics["latency"].labels(operation="get").time():
            # Try to get from redis first if available
            if not self.config.use_mock and self._redis is not None:
                # Define the Redis get operation as a closure
                async def redis_get_operation() -> Optional[Tuple[Any, Dict[str, Any], bool]]:
                    serialized_value = await self._redis.get(full_key)
                    if serialized_value is None:
                        return None
                        
                    # Found the value
                    result_value = self._deserialize_value(serialized_value)
                    result_metadata = {}
                    
                    # Get metadata if requested
                    if include_metadata:
                        serialized_metadata = await self._redis.get(f"{full_key}:metadata")
                        if serialized_metadata is not None:
                            result_metadata = self._deserialize_value(serialized_metadata)
                    
                    return result_value, result_metadata, True
                
                # Execute the operation safely
                redis_result = await self.safe_redis_operation("get", None, redis_get_operation)
                
                if redis_result is not None:
                    value, metadata, found_value = redis_result
                    status = "success"
                    
                    # Record metrics and return the value
                    self.metrics["operations"].labels(
                        operation="get",
                        status=status
                    ).inc()
                    
                    if include_metadata:
                        return value, metadata
                    return value
                else:
                    status = "not_found"
            
            # Try mock data if Redis not available, failed, or key not found in Redis
            if full_key in self._mock_data:
                mock_value, mock_metadata, ttl = self._mock_data[full_key]
                
                # Check if TTL expired (we don't implement actual expiry in the mock)
                if ttl is not None and ttl > 0:
                    # We don't have real time tracking in mock, so we can't actually expire
                    # In a real implementation, we'd check current time against TTL
                    pass
                    
                value = mock_value
                metadata = mock_metadata
                found_value = True
                
                # Determine the correct status based on context
                if status == "redis_failure":
                    status = "fallback"
                else:
                    status = "success" if self.config.use_mock else "fallback"
            
            # If we didn't find a value anywhere, status is "not_found"
            if not found_value:
                status = "not_found"
            
            # Record metrics based on final status
            self.metrics["operations"].labels(
                operation="get",
                status=status
            ).inc()
            
            # Return appropriate value based on whether metadata was requested
            if include_metadata:
                return (value, metadata) if found_value else (default, {})
            else:
                return value if found_value else default

    async def delete(self, key: str, memory_type: Optional[str] = None) -> bool:
        """Delete a key from memory.
        
        Args:
            key: Key to delete.
            memory_type: Optional memory type for metadata.
            
        Returns:
            True if key was found and deleted, False otherwise.
        """
        full_key = self._get_full_key(key)
        deleted = False
        status = "unknown"
        
        with self.metrics["latency"].labels(operation="delete").time():
            # Try Redis first if configured
            if not self.config.use_mock and self._redis is not None:
                # Define the Redis delete operation as a closure
                async def redis_delete_operation() -> bool:
                    pipeline = await self._redis.pipeline()
                    await pipeline.delete(full_key)
                    await pipeline.delete(f"{full_key}:metadata")
                    deleted_count = await pipeline.execute()
                    # If either the key or metadata was deleted, consider it a success
                    return any([count > 0 for count in deleted_count])
                
                # Execute the operation safely
                redis_deleted = await self.safe_redis_operation("delete", False, redis_delete_operation)
                
                if redis_deleted:
                    deleted = True
                    status = "success"
                    self.metrics["operations"].labels(
                        operation="delete",
                        status=status
                    ).inc()
                    return True
                else:
                    status = "not_found_in_redis"
            
            # Check mock data if Redis not available, failed, or key not found
            if full_key in self._mock_data:
                del self._mock_data[full_key]
                deleted = True
                status = "success" if self.config.use_mock else "fallback"
            
            # Record operation metrics
            self.metrics["operations"].labels(
                operation="delete",
                status=status
            ).inc()
            
            return deleted

    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory.
        
        Args:
            key: Key to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        full_key = self._get_full_key(key)
        exists = False
        
        with self.metrics["latency"].labels(operation="exists").time():
            # Check mock store first
            if self.config.use_mock:
                exists = full_key in self._mock_data
            # Try Redis if available
            elif self._redis is not None:
                # Define the Redis exists operation as a closure
                async def redis_exists_operation() -> bool:
                    return await self._redis.exists(full_key) > 0
                
                # Execute the operation safely
                exists = await self.safe_redis_operation("exists", False, redis_exists_operation)
            
            self.metrics["operations"].labels(
                operation="exists",
                status="success" if exists else "not_found"
            ).inc()
        
        return exists

    async def keys(self, pattern: str = "*", memory_type: Optional[MemoryType] = None, **kwargs: Any) -> List[str]:
        """Get keys matching a pattern.
        
        Args:
            pattern: Pattern to match.
            memory_type: Optional memory type to filter by.
            **kwargs: Additional arguments.
            
        Returns:
            List of keys matching the pattern.
        """
        full_pattern = self._get_full_key(pattern)
        keys: List[str] = []
        
        with self.metrics["latency"].labels(operation="keys").time():
            # Handle mock data case
            if self.config.use_mock:
                keys = [k for k in self._mock_data.keys() if fnmatch.fnmatch(k, full_pattern)]
                # Filter by memory type for mock data
                if memory_type:
                    filtered_keys = []
                    for key in keys:
                        _, metadata, _ = self._mock_data[key]
                        if metadata.get("memory_type") == memory_type:
                            filtered_keys.append(key)
                    keys = filtered_keys
            # Handle Redis case
            elif self._redis is not None:
                # Define the Redis keys operation as a closure
                async def redis_keys_operation() -> List[str]:
                    result_keys: List[str] = []
                    # Use SCAN to avoid blocking for large key sets
                    cursor = b'0'
                    while cursor != 0:
                        cursor, batch = await self._redis.scan(cursor, match=full_pattern.encode('utf-8'))
                        result_keys.extend([k.decode('utf-8') for k in batch])
                    
                    # Filter by memory type if specified
                    if memory_type:
                        filtered_keys = []
                        # Fetch metadata for each key
                        for key in result_keys:
                            metadata_bytes = await self._redis.get(key + ":metadata")
                            if metadata_bytes:
                                metadata = self._deserialize_value(metadata_bytes)
                                if metadata.get("memory_type") == memory_type:
                                    filtered_keys.append(key)
                        result_keys = filtered_keys
                    
                    return result_keys
                
                # Execute the operation safely
                keys = await self.safe_redis_operation("keys", [], redis_keys_operation)
            
            self.metrics["operations"].labels(
                operation="keys",
                status="success"
            ).inc()
        
        return [key[len(self.config.namespace):] for key in keys] # Return keys without namespace

    async def query(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        offset: int = 0,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query memory for entries matching a query.
        
        This implementation only supports querying mock data or fetching all keys
        from Redis and filtering in memory. A real implementation would use Redis Search.
        
        Args:
            query: Query string.
            memory_type: Optional memory type to filter by.
            limit: Maximum number of results to return.
            offset: Number of results to skip.
            **kwargs: Additional arguments.
            
        Returns:
            MemoryQueryResult containing matching entries.
        """
        results: List[MemoryEntry] = []
        total_count = 0
        
        with self.metrics["latency"].labels(operation="query").time():
            # Handle mock data case
            if self.config.use_mock:
                # Simple substring search on values in mock data
                all_entries = list(self._mock_data.items())
                if memory_type:
                    all_entries = [item for item in all_entries if item[1][1].get("memory_type") == memory_type]

                matching_entries = [
                    MemoryEntry(key=k[len(self.config.namespace):], value=v, metadata=m)
                    for k, (v, m, ttl) in all_entries
                    if query.lower() in str(v).lower() or query.lower() in str(m).lower()
                ]
                total_count = len(matching_entries)
                results = matching_entries[offset : offset + limit]
            # Handle Redis case
            elif self._redis is not None:
                # Define the Redis query operation as a closure
                async def redis_query_operation() -> Tuple[List[MemoryEntry], int]:
                    # Get all keys without namespaces
                    all_keys = await self.keys(pattern="*")

                    all_entries_redis: List[MemoryEntry] = []
                    for key in all_keys:
                        full_key = self._get_full_key(key)
                        value_bytes = await self._redis.get(full_key)
                        metadata_bytes = await self._redis.get(full_key + ":metadata")
                        if value_bytes is not None:
                            value = self._deserialize_value(value_bytes)
                            metadata = self._deserialize_value(metadata_bytes) if metadata_bytes else {}
                            if memory_type is None or metadata.get("memory_type") == memory_type:
                                all_entries_redis.append(MemoryEntry(key=key, value=value, metadata=metadata))

                    # Filter entries in memory (inefficient for large datasets)
                    matching_entries_redis = [
                        entry for entry in all_entries_redis
                        if query.lower() in str(entry.value).lower() or query.lower() in str(entry.metadata).lower()
                    ]
                    result_total = len(matching_entries_redis)
                    result_entries = matching_entries_redis[offset : offset + limit]
                    return result_entries, result_total
                
                # Execute the operation safely
                query_result = await self.safe_redis_operation("query", ([], 0), redis_query_operation)
                if query_result:
                    results, total_count = query_result

            self.metrics["operations"].labels(
                operation="query",
                status="success"
            ).inc()
        
        return MemoryQueryResult(entries=results, total=total_count)

    async def stats(self, **kwargs: Any) -> MemoryStats:
        """Get memory statistics.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            Statistics about memory usage.
        """
        total_entries = 0
        size_bytes = 0
        entries_by_type: Dict[str, int] = {}
        
        with self.metrics["latency"].labels(operation="stats").time():
            # Handle mock data case
            if self.config.use_mock:
                # Count entries in mock data
                total_entries = len(self._mock_data)
                # Get entries by type
                for _, (_, metadata, _) in self._mock_data.items():
                    memory_type = metadata.get("memory_type", "unknown")
                    memory_type_str = memory_type if isinstance(memory_type, str) else str(memory_type)
                    entries_by_type[memory_type_str] = entries_by_type.get(memory_type_str, 0) + 1
                # Estimate size (very rough)
                size_bytes = sum([len(pickle.dumps(v)) + len(pickle.dumps(m)) for _, (v, m, _) in self._mock_data.items()])
            # Handle Redis case
            elif self._redis is not None:
                # Define the Redis stats operation as a closure
                async def redis_stats_operation() -> Tuple[int, int, Dict[str, int]]:
                    # Use INFO command for Redis stats
                    info = await self._redis.info()
                    stats_total_entries = info.get('db0', {}).get('keys', 0) # Approximate for namespace
                    # Redis INFO provides total memory usage
                    stats_size_bytes = info.get('used_memory', 0)
                    
                    # For entries by type, check each key's metadata (expensive)
                    # Limit to a reasonable number of keys to avoid performance issues
                    memory_type_counts: Dict[str, int] = {}
                    keys = await self.keys()
                    max_keys_to_check = min(100, len(keys))  # Limit to 100 keys
                    sampled_keys = keys[:max_keys_to_check]
                    
                    for key in sampled_keys:
                        result = await self.get(key, include_metadata=True)
                        if isinstance(result, tuple):
                            _, metadata = result
                            memory_type = metadata.get("memory_type", "unknown")
                            memory_type_str = memory_type if isinstance(memory_type, str) else str(memory_type)
                            memory_type_counts[memory_type_str] = memory_type_counts.get(memory_type_str, 0) + 1
                    
                    # Extrapolate to full dataset - note this is approximate
                    stats_entries_by_type = {}
                    if max_keys_to_check > 0:
                        extrapolation_factor = stats_total_entries / max_keys_to_check
                        stats_entries_by_type = {k: int(v * extrapolation_factor) for k, v in memory_type_counts.items()}
                    
                    return stats_total_entries, stats_size_bytes, stats_entries_by_type
                
                # Execute the operation safely
                stats_result = await self.safe_redis_operation("stats", (0, 0, {}), redis_stats_operation)
                if stats_result:
                    total_entries, size_bytes, entries_by_type = stats_result
            
            self.metrics["operations"].labels(
                operation="stats",
                status="success"
            ).inc()
        
        # Update metrics
        self.metrics["size"].set(size_bytes)
        for memory_type, count in entries_by_type.items():
            self.metrics["entries"].labels(memory_type=memory_type).set(count)
        
        # Convert string keys to MemoryType enum values
        typed_entries: Dict[MemoryType, int] = {}
        for type_str, count in entries_by_type.items():
            try:
                # Try to convert string to enum
                memory_type = MemoryType(type_str)
                typed_entries[memory_type] = count
            except ValueError:
                # If not a valid enum value, store under SHORT_TERM as fallback
                self.logger.warning(f"Unknown memory type: {type_str}, using SHORT_TERM")
                typed_entries[MemoryType.SHORT_TERM] = typed_entries.get(MemoryType.SHORT_TERM, 0) + count
        
        return MemoryStats(
            total_entries=total_entries,
            size_bytes=size_bytes,
            entries_by_type=typed_entries
        )