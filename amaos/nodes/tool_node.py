"""Tool Node for AMAOS.

This module provides the Tool Node component responsible for executing tools
or plugins (e.g., browser, file system).
"""

import logging
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


class ToolNodeConfig(BaseModel):
    """Configuration for the Tool Node."""
    
    node_id: str = "tool_node"


from amaos.models.tool import ToolTask, ToolResult
from types import TracebackType

from amaos.core.node_protocol import Node, NodeTask, NodeResult

class ToolNode(Node):
    """Node responsible for executing tools or plugins.
    
    This node handles:
    - Tool registration and discovery
    - Tool execution with proper error handling
    - Tool result formatting and validation
    """
    
    def __init__(self, config: Optional[ToolNodeConfig] = None) -> None:
        """Initialize the Tool Node.
        
        Args:
            config: Configuration for the Tool Node.
        """
        self.config = config or ToolNodeConfig()
        self.logger = logging.getLogger(__name__)
        self.node_id = self.config.node_id
        self._tools: Dict[str, Any] = {}
        self._stats: Dict[str, Dict[str, int]] = {}

    def register_tool(self, name: str, tool: Any) -> None:
        self._tools[name] = tool
        if name not in self._stats:
            self._stats[name] = {"calls": 0, "successes": 0, "failures": 0, "timeouts": 0}

    async def execute(self, task: ToolTask, timeout: float = 2.0, max_retries: int = 1) -> ToolResult:
        tool = self._tools.get(task.tool)
        self._stats.setdefault(task.tool, {"calls": 0, "successes": 0, "failures": 0, "timeouts": 0})
        self._stats[task.tool]["calls"] += 1
        if tool is None:
            return ToolResult(success=False, result={"error": f"Tool '{task.tool}' not found"})
        last_exc: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                import asyncio
                coro = tool.run(task)
                result = await asyncio.wait_for(coro, timeout=timeout)
                assert isinstance(result, ToolResult)
                self._stats[task.tool]["successes"] += 1
                return result
            except asyncio.TimeoutError:
                self._stats[task.tool]["timeouts"] += 1
                last_exc = asyncio.TimeoutError("Tool execution timed out")
            except Exception as exc:
                self._stats[task.tool]["failures"] += 1
                last_exc = exc
        err_msg = str(last_exc) if last_exc else "Unknown error"
        return ToolResult(success=False, result={"error": err_msg})

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        return {k: v.copy() for k, v in self._stats.items()}

    async def initialize(self) -> None:
        self.logger.info(f"Initializing {self.node_id}")

    async def handle(self, task: NodeTask) -> NodeResult:
        try:
            if task.task_type != "tool":
                return NodeResult(success=False, result="Unsupported task_type for ToolNode", error="unsupported_task_type", source=self.node_id)
            tool_task = ToolTask(**task.payload)
            result = await self.execute(tool_task)
            return NodeResult(
                success=result.success,
                result=result.result,
                source=self.node_id,
                error=result.error,
            )
        except Exception as e:
            return NodeResult(success=False, result="Exception in ToolNode", error=str(e), source=self.node_id)



    async def start(self) -> None:
        self.logger.info(f"Starting {self.node_id}")

    async def stop(self) -> None:
        self.logger.info(f"Stopping {self.node_id}")
