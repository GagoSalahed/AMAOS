"""Control Node for AMAOS.

This module provides the Control Node component responsible for orchestrating task flow,
validation, and status tracking.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, Literal

from pydantic import BaseModel, Field



class ControlNodeConfig(BaseModel):
    """Configuration for the Control Node."""
    
    node_id: str = "control_node"
    max_concurrent_tasks: int = 5


from amaos.core.node_protocol import Node, NodeTask, NodeResult
from typing import Dict, Optional

class ControlNode(Node):
    """Universal orchestrator node that routes NodeTasks to registered nodes by task_type, with retry and stats support."""
    def __init__(self, node_registry: Dict[str, Node], max_retries: int = 1):
        self.nodes = node_registry
        self.max_retries = max_retries

    async def initialize(self) -> None:
        for node in self.nodes.values():
            await node.initialize()

    async def handle(self, task: NodeTask) -> NodeResult:
        node = self.nodes.get(task.task_type)
        if not node:
            return NodeResult(success=False, result=f"Unknown node: {task.task_type}", error="NODE_NOT_FOUND")
        for attempt in range(1, self.max_retries + 2):  # 1 initial + retries
            result = await node.handle(task)
            if result.success:
                return result
        return NodeResult(success=False, result="All retries failed", error="RETRY_EXCEEDED")

    def get_stats(self) -> dict:
        return { node_type: node.get_stats() for node_type, node in self.nodes.items() }

class ChainedControlNode(ControlNode):
    """
    ControlNode that supports chaining/multi-step pipelines and optional metadata-driven routing.
    - If task_type == 'pipeline', executes steps in sequence, passing intermediate results.
    - Supports metadata such as max_tokens, role constraints, etc. for advanced routing.
    """
    async def handle(self, task: NodeTask) -> NodeResult:
        # Handle multi-step pipeline
        if task.task_type == "pipeline":
            steps = task.payload["steps"]
            if not steps:
                # Handle empty pipeline case
                return NodeResult(success=True, result="Pipeline is empty, no operations performed.")
            intermediate = None
            for step in steps:
                subtask = NodeTask(
                    task_type=step["type"],
                    payload=step["payload"],
                    metadata=step.get("metadata")
                )
                if intermediate:
                    # Pass previous result as input to next step
                    subtask.payload["input"] = intermediate.result
                # Optionally use metadata for routing (e.g., max_tokens, role)
                node = self.nodes.get(subtask.task_type)
                if node and subtask.metadata:
                    # Example: honor max_tokens or role constraints (extend as needed)
                    if "max_tokens" in subtask.metadata:
                        subtask.payload["max_tokens"] = subtask.metadata["max_tokens"]
                    if "role" in subtask.metadata:
                        subtask.payload["role"] = subtask.metadata["role"]
                intermediate = await super().handle(subtask)
                if not intermediate.success:
                    return intermediate
            # Ensure intermediate is not None before returning
            if intermediate is None:
                return NodeResult(success=False, result="Pipeline processing failed to produce a result.", error="PIPELINE_NO_RESULT")
            return intermediate
        # For non-pipeline, optionally use metadata for routing
        if task.metadata:
            node = self.nodes.get(task.task_type)
            if node:
                if "max_tokens" in task.metadata:
                    task.payload["max_tokens"] = task.metadata["max_tokens"]
                if "role" in task.metadata:
                    task.payload["role"] = task.metadata["role"]
        return await super().handle(task)
