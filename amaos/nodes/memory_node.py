"""Memory Node for AMAOS.

This module provides the Memory Node component responsible for abstracting
persistence (short/long-term memory).
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set, TypeVar, Union, cast, TYPE_CHECKING, Protocol, Callable
from enum import Enum
import traceback

from pydantic import BaseModel, Field, root_validator

# Import SemanticMemoryConfig directly for type checking
if TYPE_CHECKING:
    from amaos.memory.semantic_memory import SemanticMemory, SemanticMemoryConfig

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

class MemoryType(str, Enum):
    """Type of memory storage."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryNodeConfig(BaseModel):
    """Configuration for the Memory Node."""
    
    node_id: str = "memory_node"
    default_memory_type: MemoryType = MemoryType.SHORT_TERM
    ttl_short_term: int = 3600  # Time to live in seconds for short-term memory
    ttl_long_term: int = 2592000  # Time to live in seconds for long-term memory (30 days)
    enable_semantic_memory: bool = False  # Whether to enable semantic memory - default to False for tests
    semantic_memory_config: Optional[Dict[str, Any]] = None  # Use Dict to avoid forward ref issues


from amaos.core.node_protocol import Node, NodeTask, NodeResult

# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from amaos.memory.semantic_memory import SemanticMemory, SemanticMemoryConfig

class MemoryNode(Node):
    """Node responsible for abstracting persistence.
    
    This node handles:
    - Short-term and long-term memory storage
    - Memory retrieval with context-aware filtering
    - Memory persistence and expiration
    - Memory indexing and search
    """
    
    def __init__(self, config: Optional[MemoryNodeConfig] = None) -> None:
        """Initialize the Memory Node.
        
        Args:
            config: Configuration for the Memory Node.
        """
        # Initialize the Node base class first
        super().__init__()
        
        # Set configuration
        self.config = config or MemoryNodeConfig()
        self.logger = logging.getLogger(__name__)
        self.node_id = self.config.node_id
        
        # Initialize node configuration with appropriate ID and name
        from amaos.core.node import NodeConfig
        node_config = NodeConfig(
            id=self.node_id,
            name="Memory Node",
            description="Node for memory storage and retrieval, including semantic memory"
        )
        
        self.memories: Dict[MemoryType, Dict[str, Any]] = {
            MemoryType.SHORT_TERM: {},
            MemoryType.LONG_TERM: {},
            MemoryType.EPISODIC: {},
            MemoryType.SEMANTIC: {},
        }
        self.metrics: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "failures": 0,
            "fallbacks": 0,
            "semantic_stores": 0,
            "semantic_searches": 0,
        }
        
        # Initialize Redis and semantic memory
        self.redis: Optional[RedisProtocol] = None  # Should be set externally if using Redis
        self.redis_available = False
        
        # Initialize semantic memory if enabled
        self.semantic_memory = None
        if self.config.enable_semantic_memory:
            try:
                from amaos.memory.memory import Memory
                from amaos.memory.semantic_memory import SemanticMemory, SemanticMemoryConfig
                memory = Memory()
                
                # Convert dict config to proper SemanticMemoryConfig if needed
                sem_config = None
                if self.config.semantic_memory_config:
                    if isinstance(self.config.semantic_memory_config, dict):
                        sem_config = SemanticMemoryConfig(**self.config.semantic_memory_config)
                    else:
                        sem_config = self.config.semantic_memory_config
                
                self.semantic_memory = SemanticMemory(
                    config=sem_config,
                    memory=memory
                )
            except ImportError:
                self.logger.warning("Semantic memory module not available")
                self.semantic_memory = None

    async def initialize(self) -> None:
        """Initialize the node."""
        # Use explicit cast to make the super().initialize() call safe for mypy
        await cast(Any, super()).initialize()
        
        self.logger.info(f"Initializing {self.node_id}")
        # Check Redis availability if present
        if self.redis:
            try:
                # No need to assert, properly typed above
                # Check if ping is successful
                await self.redis.ping()
                self.redis_available = True
                self.logger.info("Redis connection successful.")
            except Exception as e:
                self.redis_available = False
                self.logger.warning(f"Redis connection failed: {e}. Falling back to in-memory storage.")
                
        # Initialize semantic memory if enabled
        if self.semantic_memory:
            self.logger.info("Semantic memory system enabled.")

    async def handle(self, task: NodeTask) -> NodeResult:
        # Generate a trace_id if not present in task metadata
        if task.metadata is None:
            task.metadata = {}
        if "trace_id" not in task.metadata:
            task.metadata["trace_id"] = str(uuid.uuid4())
            
        trace_id = task.metadata["trace_id"]
        self.logger.info(f"[trace:{trace_id}] Processing memory task: {task.task_type}")
        
        try:
            if task.task_type != "memory":
                return NodeResult(
                    success=False, 
                    result="Unsupported task_type for MemoryNode", 
                    error="unsupported_task_type", 
                    source=self.node_id
                )
                
            action = task.payload.get("action")
            key = task.payload.get("key")
            value = task.payload.get("value")
            memory_type = task.payload.get("memory_type")
            
            # Defensive: ensure key is str, memory_type is MemoryType
            if not isinstance(key, str):
                key = str(key) if key is not None else ""
            if memory_type is not None and not isinstance(memory_type, MemoryType):
                try:
                    memory_type = MemoryType(memory_type)
                except Exception:
                    memory_type = self.config.default_memory_type
            else:
                memory_type = memory_type or self.config.default_memory_type
                
            # Handle standard memory operations
            if action == "set":
                if key and value is not None:
                    ttl = task.payload.get("ttl")
                    success = await self.set(key, value, memory_type, ttl)
                    return NodeResult(
                        success=success,
                        result={"key": key, "stored": success},
                        source=self.node_id
                    )
                return NodeResult(
                    success=False,
                    result="Missing key or value",
                    error="missing_parameters",
                    source=self.node_id
                )
            elif action == "get":
                if key:
                    value = await self.get(key, memory_type)
                    return NodeResult(
                        success=value is not None,
                        result={"key": key, "value": value},
                        source=self.node_id
                    )
                return NodeResult(
                    success=False,
                    result="Missing key",
                    error="missing_parameters",
                    source=self.node_id
                )
            elif action == "delete":
                if key:
                    success = await self.delete(key, memory_type)
                    return NodeResult(
                        success=success,
                        result={"key": key, "deleted": success},
                        source=self.node_id
                    )
                return NodeResult(
                    success=False,
                    result="Missing key",
                    error="missing_parameters",
                    source=self.node_id
                )
            # Handle semantic memory operations
            elif action == "semantic_store":
                if not self.semantic_memory or not self.config.enable_semantic_memory:
                    return NodeResult(
                        success=False,
                        result="Semantic memory not enabled",
                        error="semantic_memory_disabled",
                        source=self.node_id
                    )
                
                text = task.payload.get("text")
                metadata = task.payload.get("metadata", {})
                
                if not text:
                    return NodeResult(
                        success=False,
                        result="Missing text for semantic storage",
                        error="missing_parameters",
                        source=self.node_id
                    )
                
                # Generate an ID if not provided
                if not key:
                    key = f"sem_{str(uuid.uuid4())}"
                
                success = await self.semantic_store(key, text, metadata)
                self.metrics["semantic_stores"] += 1
                
                return NodeResult(
                    success=success,
                    result={
                        "key": key, 
                        "stored": success,
                        "metadata": metadata
                    },
                    source=self.node_id
                )
            elif action == "semantic_get":
                if not self.semantic_memory or not self.config.enable_semantic_memory:
                    return NodeResult(
                        success=False,
                        result="Semantic memory not enabled",
                        error="semantic_memory_disabled",
                        source=self.node_id
                    )
                
                query = task.payload.get("query")
                limit = task.payload.get("limit", 5)
                metadata_filter = task.payload.get("metadata_filter", {})
                
                if not query:
                    return NodeResult(
                        success=False,
                        result="Missing query for semantic retrieval",
                        error="missing_parameters",
                        source=self.node_id
                    )
                
                results = await self.semantic_search(query, limit, metadata_filter)
                self.metrics["semantic_searches"] += 1
                
                return NodeResult(
                    success=True,
                    result={
                        "query": query,
                        "results": results,
                        "count": len(results)
                    },
                    source=self.node_id
                )
            else:
                return NodeResult(
                    success=False,
                    result=f"Unsupported action: {action}",
                    error="unsupported_action",
                    source=self.node_id
                )
        except Exception as e:
            self.logger.error(f"[trace:{trace_id}] Error handling task: {e}")
            return NodeResult(
                success=False,
                result=f"Error: {str(e)}",
                error="internal_error",
                source=self.node_id
            )

    async def start(self) -> None:
        """Start the node."""
        self.logger.info(f"Starting {self.node_id}")
        
    async def stop(self) -> None:
        """Stop the node."""
        self.logger.info(f"Stopping {self.node_id}")

    async def set(self, key: str, value: Any, memory_type: Optional[MemoryType] = None, ttl: Optional[int] = None) -> bool:
        memory_type = memory_type or self.config.default_memory_type
        ttl = ttl or self._get_ttl(memory_type)
        redis_success = False
        if self.redis is not None and self.redis_available:
            try:
                await self.redis.set(self._make_key(key, memory_type), value, ex=ttl)
                redis_success = True
            except Exception:
                self.metrics["failures"] += 1
                # No longer marking redis_available as False here, handle in initialize/ping
                # self.redis_available = False
        
        if not redis_success:
            # Fallback to in-memory
            self.metrics["fallbacks"] += 1
            self.memories[memory_type][key] = {
                "value": value,
                "expires_at": self._get_expiry_time(ttl) if ttl else None,
            }
        
        return True  # Return success indicator

    async def get(self, key: str, memory_type: Optional[MemoryType] = None) -> Optional[Any]:
        """Get a memory entry.
        
        Args:
            key: The key to retrieve.
            memory_type: The type of memory to retrieve from. Defaults to the configured default.
            
        Returns:
            The stored value if found, otherwise None.
        """
        memory_type = memory_type or self.config.default_memory_type
        
        # Track if we need to check in-memory storage
        check_in_memory = True
        
        # Try Redis first if available
        if self.redis is not None and self.redis_available:
            try:
                val = await self.redis.get(self._make_key(key, memory_type))
                if val is not None:
                    self.metrics["hits"] += 1
                    return val
                # If we successfully checked Redis and found nothing, no need to check in-memory
                check_in_memory = False
                self.metrics["misses"] += 1
                return None
            except Exception:
                self.metrics["failures"] += 1
                # We'll fall through to in-memory check since check_in_memory is still True
        
        # Only check in-memory if Redis wasn't available/failed or we're configured to not use Redis
        if check_in_memory:  # This silences mypy unreachable warning
            self.metrics["fallbacks"] += 1
            entry = self.memories[memory_type].get(key)
            if entry is not None:
                expires_at = entry.get("expires_at")
                if expires_at and expires_at < self._now():
                    del self.memories[memory_type][key]
                    self.metrics["misses"] += 1
                    return None
                self.metrics["hits"] += 1
                return entry["value"]
            self.metrics["misses"] += 1
        
        return None

    async def delete(self, key: str, memory_type: Optional[MemoryType] = None) -> bool:
        """Delete a memory entry.
        
        Args:
            key: The key to delete.
            memory_type: The type of memory to delete from. Defaults to the configured default.
            
        Returns:
            True if the key was deleted, False otherwise.
        """
        memory_type = memory_type or self.config.default_memory_type
        redis_success = False
        check_in_memory = True
        
        if self.redis is not None and self.redis_available:
            try:
                result = await self.redis.delete(self._make_key(key, memory_type))
                redis_success = bool(result)
                if redis_success:
                    # If Redis deletion was successful, no need to check in-memory
                    check_in_memory = False
                    return True
            except Exception:
                self.metrics["failures"] += 1
                # We'll fall through to in-memory check
        
        # Only check in-memory if Redis wasn't available/failed or we need to try both storages
        if check_in_memory:  # This silences mypy unreachable warning
            self.metrics["fallbacks"] += 1
            if key in self.memories[memory_type]:
                del self.memories[memory_type][key]
                return True
            
        return False  # Return False if not found in either storage

    async def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear all memories of a specific type.
        
        Args:
            memory_type: The type of memory to clear. Defaults to the configured default.
        """
        memory_type = memory_type or self.config.default_memory_type
        clear_in_memory = True
        
        # Try Redis first if available and implement pattern keys/delete or similar
        if self.redis is not None and self.redis_available:
            try:
                # We would implement Redis clearing pattern here if supported
                # e.g., something like: await self.redis.delete_pattern(f"{memory_type.value}:*")
                
                # For now we just note it's not implemented and fall back to in-memory
                # Note: This is a controlled exception path, not a failure
                self.metrics["fallbacks"] += 1
            except Exception:
                self.metrics["failures"] += 1
                self.metrics["fallbacks"] += 1
        
        # Always clear in-memory for now since Redis clear by type isn't implemented
        if clear_in_memory:  # This silences mypy unreachable warning
            self.memories[memory_type].clear()

    async def search(self, memory_type: MemoryType, filter_func: Callable[[Any], bool]) -> List[Any]:
        """Search for memory entries matching a filter function.
        
        Args:
            memory_type: The type of memory to search in.
            filter_func: A function that returns True for items to include.
            
        Returns:
            List of values matching the filter function.
        """
        # Only in-memory search supported for now
        results: List[Any] = []
        for entry in self.memories[memory_type].values():
            val = entry["value"]
            if filter_func(val):
                results.append(val)
        return results
        
    async def semantic_store(self, key: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store text in semantic memory with vector embeddings.
        
        Args:
            key: Unique identifier for the text
            text: The text content to store
            metadata: Optional metadata associated with the text
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.semantic_memory:
            self.logger.warning("Semantic memory not available.")
            return False
            
        return await self.semantic_memory.store(key, text, metadata or {})
        
    async def semantic_search(self, query: str, k: int = 5, 
                             metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search semantic memory for text similar to the query.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            metadata_filter: Optional filter criteria for metadata fields
            
        Returns:
            List of matching items with scores
        """
        if not self.semantic_memory:
            self.logger.warning("Semantic memory not available.")
            return []
            
        return await self.semantic_memory.search(query, k, metadata_filter)


    def _get_ttl(self, memory_type: MemoryType) -> Optional[int]:
        if memory_type == MemoryType.SHORT_TERM:
            return self.config.ttl_short_term
        elif memory_type == MemoryType.LONG_TERM:
            return self.config.ttl_long_term
        else:
            return None

    def _get_expiry_time(self, ttl: Optional[int]) -> Optional[float]:
        import time
        if ttl is not None:
            return self._now() + ttl
        return None

    def _now(self) -> float:
        import time
        return time.time()

    def _make_key(self, key: str, memory_type: MemoryType) -> str:
        return f"{memory_type.value}:{key}"

    def get_stats(self) -> Dict[str, Any]:
        """Get the stats of the memory node.
        
        Returns:
            Dictionary containing memory node statistics.
        """
        # Explicitly annotate stats as a dict for mypy
        stats: Dict[str, Any] = self.metrics.copy()
        
        # Add information about semantic memory if available
        if self.semantic_memory:
            # Explicitly annotate and create a new variable to avoid type errors
            from typing import Dict, Any
            semantic_stats_dict: Dict[str, Any] = self.semantic_memory.get_stats()
            stats["semantic_memory"] = semantic_stats_dict
            stats["semantic_memory_enabled"] = True
        else:
            stats["semantic_memory_enabled"] = False
            
        # Add Redis information
        stats["redis_available"] = self.redis_available
        
        # Add memory counts
        for memory_type, memories in self.memories.items():
            memory_count = len(memories)  # Store the count properly typed
            stats[f"{memory_type.value}_count"] = memory_count
            
        return stats