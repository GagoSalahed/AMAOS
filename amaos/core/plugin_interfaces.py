"""Plugin interfaces for AMAOS.

This module defines the interfaces for plugins in the AMAOS system.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel


class PluginState(Enum):
    """Enum representing the possible states of a plugin."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class PluginMetadata(BaseModel):
    """Metadata for a plugin."""

    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = []
    tags: List[str] = []


class PluginConfig(BaseModel):
    """Configuration for a plugin."""

    enabled: bool = True
    auto_start: bool = False
    config: Dict[str, Any] = {}


T = TypeVar("T")
EventHandler = Callable[[Dict[str, Any]], None]


@runtime_checkable
class PluginInterface(Protocol):
    """Protocol defining the interface for plugins."""

    metadata: PluginMetadata
    config: PluginConfig
    state: PluginState

    async def initialize(self) -> None:
        """Initialize the plugin."""
        ...

    async def start(self) -> None:
        """Start the plugin."""
        ...

    async def stop(self) -> None:
        """Stop the plugin."""
        ...

    def register_event_handler(self, event_name: str, handler: EventHandler) -> None:
        """Register an event handler."""
        ...

    def unregister_event_handler(self, event_name: str, handler: EventHandler) -> None:
        """Unregister an event handler."""
        ...

    def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Emit an event."""
        ...


class BasePluginInterface(ABC):
    """Abstract base class for plugins."""

    def __init__(
        self, metadata: PluginMetadata, config: Optional[PluginConfig] = None
    ) -> None:
        """Initialize the plugin.

        Args:
            metadata: Metadata for the plugin.
            config: Configuration for the plugin.
        """
        self.metadata = metadata
        self.config = config or PluginConfig()
        self.state = PluginState.UNINITIALIZED
        self._event_handlers: Dict[str, List[EventHandler]] = {}

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin."""
        self.state = PluginState.INITIALIZING
        # Implementation should set state to INITIALIZED when done

    @abstractmethod
    async def start(self) -> None:
        """Start the plugin."""
        self.state = PluginState.STARTING
        # Implementation should set state to RUNNING when done

    @abstractmethod
    async def stop(self) -> None:
        """Stop the plugin."""
        self.state = PluginState.STOPPING
        # Implementation should set state to STOPPED when done

    def register_event_handler(self, event_name: str, handler: EventHandler) -> None:
        """Register an event handler.

        Args:
            event_name: Name of the event.
            handler: Handler function for the event.
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    def unregister_event_handler(self, event_name: str, handler: EventHandler) -> None:
        """Unregister an event handler.

        Args:
            event_name: Name of the event.
            handler: Handler function for the event.
        """
        if event_name in self._event_handlers:
            self._event_handlers[event_name] = [
                h for h in self._event_handlers[event_name] if h != handler
            ]

    def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Emit an event.

        Args:
            event_name: Name of the event.
            event_data: Data for the event.
        """
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                handler(event_data)
