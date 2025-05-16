"""
ReflectorNode for AMAOS: Passive observer node that wraps any Node, logs NodeTask and NodeResult, supports tracing, audit logs, memory replay (via MemoryNode), and optional mutation/enrichment hooks.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, cast, Union, TypeVar, Callable, Awaitable
from pydantic import BaseModel, Field
from amaos.core.node_protocol import Node, NodeTask, NodeResult
import asyncio
import logging

class ReflectorLogEntry(BaseModel):
    """Model for a log entry in the ReflectorNode.
    
    Captures input/output timestamps, task details, result, and metadata.
    """
    timestamp_in: str
    timestamp_out: str
    task: Dict[str, Any]
    result: Dict[str, Any]
    node_type: str
    success: bool
    duration_ms: float = Field(default=0.0, description="Task execution duration in milliseconds")

class ReflectorStats(BaseModel):
    """Statistics for ReflectorNode tracking."""
    total: int = 0
    success: int = 0
    failure: int = 0
    per_type: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class ReflectorNode(Node):
    """
    Passive observer node that wraps any Node, logs NodeTask and NodeResult, supports tracing, 
    audit logs, memory replay (via MemoryNode), and optional mutation/enrichment hooks.

    Args:
        wrapped: The Node to wrap.
        logger: Optional logger to use.
        memory_node: Optional memory node for persistence.
    """

    on_log_entry: Optional[Callable[[ReflectorLogEntry], Awaitable[None]]] = None  # Async event handler for log entries

    @property
    def id(self) -> str:
        """Unique identifier for the ReflectorNode (delegates to wrapped node if available)."""
        return getattr(self.wrapped, "id", "reflector")

    def __init__(self, wrapped: Node, logger: Optional[logging.Logger] = None, memory_node: Optional[Node] = None) -> None:
        self.wrapped: Node = wrapped
        self.logger: logging.Logger = logger or logging.getLogger("ReflectorNode")
        self.memory_node: Optional[Node] = memory_node
        self._log: List[ReflectorLogEntry] = []
        self.metrics: ReflectorStats = ReflectorStats()
        self.on_log_entry = None  # Ensure on_log_entry is always defined

    async def initialize(self) -> None:
        await self.wrapped.initialize()
        if self.memory_node:
            await self.memory_node.initialize()

    async def handle(self, task: NodeTask) -> NodeResult:
        t_in = datetime.utcnow().isoformat()
        result = await self.wrapped.handle(task)
        t_out = datetime.utcnow().isoformat()
        # Calculate duration in milliseconds
        try:
            # Use the datetime module that's already imported globally
            t_in_dt = datetime.fromisoformat(t_in)
            t_out_dt = datetime.fromisoformat(t_out)
            duration_ms = (t_out_dt - t_in_dt).total_seconds() * 1000
        except Exception:
            duration_ms = 0.0
            
        entry = ReflectorLogEntry(
            timestamp_in=t_in,
            timestamp_out=t_out,
            task=task.model_dump() if hasattr(task, "model_dump") else {"task": str(task)},
            result=result.model_dump() if hasattr(result, "model_dump") else {"result": str(result)},
            node_type=task.task_type,
            success=result.success,
            duration_ms=duration_ms
        )
        self.observe(entry)
        if self.on_log_entry:
            # Notify listeners (e.g., ReflectorStream)
            await self.on_log_entry(entry)
        if self.memory_node:
            mem_key = f"reflector_log:{t_in}"
            await self.memory_node.handle(NodeTask(task_type="memory", payload={"action": "set", "key": mem_key, "value": entry.model_dump() if hasattr(entry, "model_dump") else dict(entry=entry)}))
        return result

    def observe(self, entry: ReflectorLogEntry) -> None:
        """Process and store a log entry, updating metrics.
        
        Args:
            entry: The log entry to process and store.
        """
        self._log.append(entry)
        self.metrics.total += 1
        if entry.success:
            self.metrics.success += 1
        else:
            self.metrics.failure += 1
            
        t = entry.node_type
        if t not in self.metrics.per_type:
            self.metrics.per_type[t] = {"total": 0, "success": 0, "failure": 0}
            
        self.metrics.per_type[t]["total"] += 1
        if entry.success:
            self.metrics.per_type[t]["success"] += 1
        else:
            self.metrics.per_type[t]["failure"] += 1

        # Trigger async log event if set (for streaming)
        if self.on_log_entry:
            # Schedule the async callback (do not await in sync context)
            asyncio.create_task(self.on_log_entry(entry))  # type: ignore

    def get_log(self, filter_type: Optional[str] = None) -> List[ReflectorLogEntry]:
        filtered = self._log
        if filter_type:
            filtered = [e for e in filtered if e.node_type == filter_type]
        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the reflector's operation.
        
        Returns:
            Dictionary with reflector statistics.
        """
        return {
            "tasks_handled": len(self._log),
            "log_size": len(self._log),
            **self.metrics.model_dump()
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Expose stats as a dict for compatibility with ReflectorStream and others."""
        return self.metrics.model_dump()

    def clear_log(self) -> None:
        self._log.clear()

    def get_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent log entries as dicts, for API/streaming."""
        # Return the last `limit` logs, as dicts
        return [e.model_dump() for e in self._log[-limit:]]
