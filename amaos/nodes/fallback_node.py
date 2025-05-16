"""Fallback Node for AMAOS.

This module provides the Fallback Node component responsible for handling
structured retries, backups, and provider shifts.
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar, Generic, Callable, cast

from pydantic import BaseModel, Field


T = TypeVar('T')  # Generic type for fallback results


from enum import Enum
class FallbackStrategy(str, Enum):
    """Strategy for fallback handling."""
    RETRY = "retry"  # Retry the same operation
    ALTERNATE = "alternate"  # Try an alternate implementation
    DEGRADE = "degrade"  # Provide degraded functionality
    ABORT = "abort"  # Abort the operation


class FallbackResult(Generic[T]):
    """Result of a fallback operation."""
    
    success: bool
    result: Optional[T]
    error: Optional[Exception]
    strategy_used: Optional[FallbackStrategy]
    attempts: int
    
    def __init__(
        self,
        success: bool,
        result: Optional[T] = None,
        error: Optional[Exception] = None,
        strategy_used: Optional[FallbackStrategy] = None,
        attempts: int = 0,
    ) -> None:
        """Initialize a fallback result."""
        self.success = success
        self.result = result
        self.error = error
        self.strategy_used = strategy_used
        self.attempts = attempts


class FallbackNodeConfig(BaseModel):
    """Configuration for the Fallback Node."""
    
    node_id: str = "fallback_node"
    max_retries: int = 3
    retry_delay: float = 1.0  # Delay between retries in seconds
    max_alternate_attempts: int = 2
    default_strategy: FallbackStrategy = FallbackStrategy.RETRY


from amaos.core.node_protocol import Node, NodeTask, NodeResult
from typing import List

class FallbackNode(Node):
    """
    AMAOS FallbackNode: Handles structured retries, alternate nodes, and graceful degradation for agentic tasks.

    Args:
        nodes: List of nodes (primary first, then alternates)
        config: FallbackNodeConfig (max_retries, retry_delay, etc.)

    Features:
    - Retries tasks on failure (with exponential backoff)
    - Tries alternate nodes if retries are exhausted
    - Optionally degrades gracefully (returns a fallback result)
    - Logs all attempts and errors
    - Implements handle(task: NodeTask) -> NodeResult and get_stats()
    """
    def __init__(self, nodes: List[Node], config: Optional[FallbackNodeConfig] = None):
        self.nodes = nodes
        self.config = config or FallbackNodeConfig()
        self.logger = logging.getLogger(__name__)
        self.node_id = self.config.node_id
        self.stats = {i: {"attempts": 0, "failures": 0} for i in range(len(nodes))}

    async def initialize(self) -> None:
        for node in self.nodes:
            await node.initialize()
        self.logger.info(f"Initialized {self.node_id}")

    async def handle(self, task: NodeTask) -> NodeResult:
        # Try primary node with retries
        for attempt in range(1, self.config.max_retries + 1):
            self.stats[0]["attempts"] += 1
            try:
                result = await self.nodes[0].handle(task)
                if result.success:
                    return result
                else:
                    self.stats[0]["failures"] += 1
                    self.logger.warning(f"Primary node failed (attempt {attempt}): {result.error or result.result}")
            except Exception as e:
                self.stats[0]["failures"] += 1
                self.logger.exception(f"Primary node exception (attempt {attempt})")
            await asyncio.sleep(self.config.retry_delay * (2 ** (attempt - 1)))
        # Try alternates
        for i, node in enumerate(self.nodes[1:], start=1):
            for alt_attempt in range(1, self.config.max_alternate_attempts + 1):
                self.stats[i]["attempts"] += 1
                try:
                    result = await node.handle(task)
                    if result.success:
                        return result
                    else:
                        self.stats[i]["failures"] += 1
                        self.logger.warning(f"Alternate node {i} failed (attempt {alt_attempt}): {result.error or result.result}")
                except Exception as e:
                    self.stats[i]["failures"] += 1
                    self.logger.exception(f"Alternate node {i} exception (attempt {alt_attempt})")
                await asyncio.sleep(self.config.retry_delay)
        # Degrade gracefully
        self.logger.error(f"All fallbacks exhausted for task: {task}")
        return NodeResult(success=False, result="All retries and alternates failed", error="FALLBACK_EXHAUSTED")

    def get_stats(self) -> Dict[str, Any]:
        """Get the stats of the fallback node.
        
        Returns:
            Dictionary of stats.
        """
        # Convert int keys to str to match Dict[str, Any] return type
        string_keyed_stats: Dict[str, Any] = {}
        for key, value in self.stats.items():
            string_keyed_stats[str(key)] = value
        return string_keyed_stats
