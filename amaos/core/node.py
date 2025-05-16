"""Node implementation for AMAOS.

This module provides the base node implementation that serves as the foundation
for the agent system in AMAOS.
"""

# mypy: disable-error-code="assignment"

import asyncio
import logging
import uuid
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, cast, Union, Awaitable, Coroutine, TypeAlias, Type, Protocol

import prometheus_client as prom
from pydantic import BaseModel, Field

T = TypeVar("T")

# Simple callables for standard typing
NodeHandler = Callable[['Node', Dict[str, Any]], None]
AsyncNodeHandler = Callable[['Node', Dict[str, Any]], Coroutine[Any, Any, None]]
AnyNodeHandler = Union[NodeHandler, AsyncNodeHandler]

class NodeState(Enum):
    """Enum representing the possible states of a node."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class NodeCapability(BaseModel):
    """Model representing a capability of a node."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class NodeConfig(BaseModel):
    """Configuration for a node."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    enabled: bool = True
    auto_start: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class Node:
    """Base node implementation.
    
    This class provides the foundation for all nodes in the AMAOS system,
    including agents and other components.
    """

    def __init__(self, config: NodeConfig) -> None:
        """Initialize the node.
        
        Args:
            config: Configuration for the node.
        """
        self.config = config
        self.id = config.id
        self.name = config.name
        self.state = NodeState.UNINITIALIZED
        self.logger = logging.getLogger(f"node.{self.id}")
        
        # Initialize event handlers
        self._event_handlers: Dict[str, List[NodeHandler]] = {}
        # Use AnyNodeHandler to handle mixed types - safer for mypy
        self._async_event_handlers: Dict[str, List[AnyNodeHandler]] = {}
        
        # Node capabilities
        self._capabilities: Set[NodeCapability] = set()
        
        # Metrics
        self.metrics = self._setup_metrics()

    def _setup_metrics(self) -> Dict[str, Any]:
        """Set up metrics for the node.
        
        Returns:
            Dictionary of metrics.
        """
        metrics: Dict[str, Any] = {}
        
        # Node state metric
        metrics["state"] = prom.Gauge(
            "amaos_node_state",
            "Current state of the node",
            ["node_id", "node_name", "state"]
        )
        
        # Event count metric
        metrics["events_processed"] = prom.Counter(
            "amaos_node_events_processed",
            "Number of events processed by the node",
            ["node_id", "node_name", "event_type"]
        )
        
        # Event processing time metric
        metrics["event_processing_time"] = prom.Histogram(
            "amaos_node_event_processing_time",
            "Time taken to process events",
            ["node_id", "node_name", "event_type"]
        )
        
        return metrics

    def _update_state_metric(self) -> None:
        """Update the state metric for the node."""
        # Clear previous state
        for state in NodeState:
            try:
                self.metrics["state"].remove(self.id, self.name, state.name)
            except (KeyError, ValueError):
                pass
                
        # Set current state
        self.metrics["state"].labels(
            node_id=self.id,
            node_name=self.name,
            state=self.state.name
        ).set(1)

    async def initialize(self) -> None:
        """Initialize the node.
        
        This method should be overridden by specific nodes to perform initialization.
        """
        self.state = NodeState.INITIALIZING
        self._update_state_metric()
        self.logger.info(f"Initializing node: {self.name} ({self.id})")
        
        # Perform node-specific initialization here
        
        self.state = NodeState.INITIALIZED
        self._update_state_metric()
        self.logger.info(f"Node initialized: {self.name} ({self.id})")

    async def start(self) -> None:
        """Start the node.
        
        This method should be overridden by specific nodes to start the node.
        """
        if self.state != NodeState.INITIALIZED:
            self.logger.warning(
                f"Cannot start node {self.name} ({self.id}): not initialized (state: {self.state})"
            )
            return
            
        self.state = NodeState.STARTING
        self._update_state_metric()
        self.logger.info(f"Starting node: {self.name} ({self.id})")
        
        # Perform node-specific startup here
        
        self.state = NodeState.RUNNING
        self._update_state_metric()
        self.logger.info(f"Node started: {self.name} ({self.id})")

    async def stop(self) -> None:
        """Stop the node.
        
        This method should be overridden by specific nodes to stop the node.
        """
        if self.state != NodeState.RUNNING:
            self.logger.warning(
                f"Cannot stop node {self.name} ({self.id}): not running (state: {self.state})"
            )
            return
            
        self.state = NodeState.STOPPING
        self._update_state_metric()
        self.logger.info(f"Stopping node: {self.name} ({self.id})")
        
        # Perform node-specific shutdown here
        
        self.state = NodeState.STOPPED
        self._update_state_metric()
        self.logger.info(f"Node stopped: {self.name} ({self.id})")

    def register_capability(self, capability: NodeCapability) -> None:
        """Register a capability for the node.
        
        Args:
            capability: Capability to register.
        """
        self._capabilities.add(capability)
        self.logger.debug(f"Registered capability: {capability.name}")

    def unregister_capability(self, capability_name: str) -> None:
        """Unregister a capability from the node.
        
        Args:
            capability_name: Name of the capability to unregister.
        """
        self._capabilities = {
            cap for cap in self._capabilities if cap.name != capability_name
        }
        self.logger.debug(f"Unregistered capability: {capability_name}")

    def has_capability(self, capability_name: str) -> bool:
        """Check if the node has a specific capability.
        
        Args:
            capability_name: Name of the capability to check.
            
        Returns:
            True if the node has the capability, False otherwise.
        """
        return any(cap.name == capability_name for cap in self._capabilities)

    def get_capabilities(self) -> Set[NodeCapability]:
        """Get all capabilities of the node.
        
        Returns:
            Set of node capabilities.
        """
        return self._capabilities.copy()

    def register_event_handler(self, event_name: str, handler: NodeHandler) -> None:
        """Register an event handler.
        
        Args:
            event_name: Name of the event.
            handler: Handler function for the event.
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
        self.logger.debug(f"Registered event handler for: {event_name}")



    def register_async_event_handler(
        self, event_name: str, handler: AsyncNodeHandler
    ) -> None:
        """Register an async event handler.
        
        Args:
            event_name: Name of the event.
            handler: Async handler function for the event.
        """
        if event_name not in self._async_event_handlers:
            self._async_event_handlers[event_name] = []
        # Now that we've defined proper Protocols, no need for casting
        self._async_event_handlers[event_name].append(handler)
        self.logger.debug(f"Registered async event handler for: {event_name}")

    def unregister_event_handler(self, event_name: str, handler: NodeHandler) -> None:
        """Unregister an event handler.
        
        Args:
            event_name: Name of the event.
            handler: Handler function for the event.
        """
        if event_name in self._event_handlers:
            self._event_handlers[event_name] = [
                h for h in self._event_handlers[event_name] if h != handler
            ]
            self.logger.debug(f"Unregistered event handler for: {event_name}")

    def unregister_async_event_handler(
        self, event_name: str, handler: AsyncNodeHandler
    ) -> None:
        """Unregister an async event handler.
        
        Args:
            event_name: Name of the event.
            handler: Async handler function for the event.
        """
        if event_name in self._async_event_handlers:
            self._async_event_handlers[event_name] = [
                h for h in self._async_event_handlers[event_name] if h != handler
            ]
            self.logger.debug(f"Unregistered async event handler for: {event_name}")

    def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Emit an event.
        
        Args:
            event_name: Name of the event.
            event_data: Data for the event.
        """
        self.logger.debug(f"Emitting event: {event_name}")
        
        # Update metrics
        self.metrics["events_processed"].labels(
            node_id=self.id,
            node_name=self.name,
            event_type=event_name
        ).inc()
        
        # Call synchronous handlers
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                with self.metrics["event_processing_time"].labels(
                    node_id=self.id,
                    node_name=self.name,
                    event_type=event_name
                ).time():
                    handler(self, event_data)
        
        # Call asynchronous handlers
        if event_name in self._async_event_handlers:
            # Use try/except to handle different types of handlers safely
            for handler in self._async_event_handlers[event_name]:
                try:
                    # Check if it's a coroutine function (async def)
                    if asyncio.iscoroutinefunction(handler):
                        # Create a task directly
                        coro = handler(self, event_data)  
                        asyncio.create_task(coro)
                    else:
                        # It must be an AsyncNodeHandler (callable returning coroutine)
                        async_handler = cast(AsyncNodeHandler, handler)
                        coro = async_handler(self, event_data)
                        asyncio.create_task(coro)
                except Exception as e:
                    self.logger.error(f"Error calling async handler for event {event_name}: {e}")


    async def _call_async_handler(
        self, event_name: str, handler: AsyncNodeHandler, event_data: Dict[str, Any]
    ) -> None:
        """Call an async event handler.
        
        Args:
            event_name: Name of the event.
            handler: Async handler function for the event.
            event_data: Data for the event.
        """
        with self.metrics["event_processing_time"].labels(
            node_id=self.id,
            node_name=self.name,
            event_type=event_name
        ).time():
            await handler(self, event_data)

# Type definitions are now at the top of the file for better visibility