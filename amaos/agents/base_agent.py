"""Base agent implementation for AMAOS.

This module provides the base agent implementation that can be extended by specific agents.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Set, TypeVar, cast

import prometheus_client as prom
from pydantic import BaseModel, Field

from amaos.core.node import Node, NodeCapability, NodeConfig, NodeState


class AgentCapability(NodeCapability):
    """Model representing a capability of an agent."""

    pass


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    enabled: bool = True
    auto_start: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    """Context for agent execution."""

    agent_id: str
    task_id: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Result of agent execution."""

    agent_id: str
    task_id: Optional[str] = None
    success: bool = True
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(Node):
    """Base implementation of an agent.
    
    This class provides a basic implementation that can be extended by specific agents.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the agent.
        
        Args:
            config: Configuration for the agent.
        """
        node_config = NodeConfig(
            id=config.id,
            name=config.name,
            description=config.description,
            enabled=config.enabled,
            auto_start=config.auto_start,
            config=config.config,
        )
        super().__init__(node_config)
        
        # Additional agent-specific metrics
        self.agent_metrics = self._setup_agent_metrics()

    def _setup_agent_metrics(self) -> Dict[str, Any]:
        """Set up agent-specific metrics.
        
        Returns:
            Dictionary of metrics.
        """
        metrics: Dict[str, Any] = {}
        
        # Task execution count
        metrics["tasks_executed"] = prom.Counter(
            "amaos_agent_tasks_executed",
            "Number of tasks executed by the agent",
            ["agent_id", "agent_name", "status"]
        )
        
        # Task execution time
        metrics["task_execution_time"] = prom.Histogram(
            "amaos_agent_task_execution_time",
            "Time taken to execute tasks",
            ["agent_id", "agent_name"]
        )
        
        # Memory usage
        metrics["memory_usage"] = prom.Gauge(
            "amaos_agent_memory_usage",
            "Memory usage by the agent",
            ["agent_id", "agent_name"]
        )
        
        return metrics

    async def initialize(self) -> None:
        """Initialize the agent."""
        await super().initialize()
        self.logger.info(f"Initializing agent: {self.name} ({self.id})")
        
        # Register default capabilities
        self.register_capability(
            AgentCapability(
                name="execute",
                description="Execute a task",
                parameters={
                    "context": "AgentContext object containing task details",
                },
            )
        )
        
        # Additional agent-specific initialization can be added in subclasses

    async def start(self) -> None:
        """Start the agent."""
        await super().start()
        self.logger.info(f"Starting agent: {self.name} ({self.id})")
        
        # Additional agent-specific startup can be added in subclasses

    async def stop(self) -> None:
        """Stop the agent."""
        await super().stop()
        self.logger.info(f"Stopping agent: {self.name} ({self.id})")
        
        # Additional agent-specific shutdown can be added in subclasses

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute a task.
        
        This method should be overridden by specific agents to implement their logic.
        
        Args:
            context: Context for the task execution.
            
        Returns:
            Result of the task execution.
        """
        self.logger.info(f"Executing task for agent: {self.name} ({self.id})")
        
        # Track metrics
        with self.agent_metrics["task_execution_time"].labels(
            agent_id=self.id,
            agent_name=self.name
        ).time():
            try:
                # Default implementation just returns success
                # Subclasses should override this method
                result = AgentResult(
                    agent_id=self.id,
                    task_id=context.task_id,
                    success=True,
                    outputs={},
                    metadata={"default_implementation": True},
                )
                
                self.agent_metrics["tasks_executed"].labels(
                    agent_id=self.id,
                    agent_name=self.name,
                    status="success"
                ).inc()
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error executing task: {e}")
                
                self.agent_metrics["tasks_executed"].labels(
                    agent_id=self.id,
                    agent_name=self.name,
                    status="failure"
                ).inc()
                
                return AgentResult(
                    agent_id=self.id,
                    task_id=context.task_id,
                    success=False,
                    error=str(e),
                    metadata={"error_type": type(e).__name__},
                )

    async def on_task_start(self, context: AgentContext) -> None:
        """Handle task start event.
        
        This method can be overridden by specific agents to handle task start events.
        
        Args:
            context: Context for the task.
        """
        self.logger.debug(f"Task started: {context.task_id}")

    async def on_task_complete(self, result: AgentResult) -> None:
        """Handle task complete event.
        
        This method can be overridden by specific agents to handle task complete events.
        
        Args:
            result: Result of the task.
        """
        self.logger.debug(f"Task completed: {result.task_id} (success: {result.success})")

    async def on_error(self, context: AgentContext, error: Exception) -> None:
        """Handle error event.
        
        This method can be overridden by specific agents to handle error events.
        
        Args:
            context: Context for the task.
            error: Error that occurred.
        """
        self.logger.error(f"Error in task {context.task_id}: {error}")

    def get_capabilities(self) -> Set[NodeCapability]:
        """Get all capabilities of the agent.
        
        Returns:
            Set of node capabilities.
        """
        # Cast to the correct type for the superclass
        return super().get_capabilities()
