"""Base plugin implementation for AMAOS.

This module provides a base implementation of a plugin that can be extended by specific plugins.
"""

import logging
from typing import Any, Dict, Optional

from amaos.core.plugin_interfaces import (
    BasePluginInterface,
    PluginConfig,
    PluginMetadata,
    PluginState,
)


class BasePlugin(BasePluginInterface):
    """Base implementation of a plugin.
    
    This class provides a basic implementation of the BasePluginInterface that can be
    extended by specific plugins.
    """

    def __init__(
        self, metadata: PluginMetadata, config: Optional[PluginConfig] = None
    ) -> None:
        """Initialize the plugin.
        
        Args:
            metadata: Metadata for the plugin.
            config: Configuration for the plugin.
        """
        super().__init__(metadata, config)
        self.logger = logging.getLogger(f"plugin.{metadata.name}")

    async def initialize(self) -> None:
        """Initialize the plugin.
        
        This method should be overridden by specific plugins to perform initialization.
        """
        self.state = PluginState.INITIALIZING
        self.logger.info(f"Initializing plugin: {self.metadata.name}")
        
        # Perform plugin-specific initialization here
        
        self.state = PluginState.INITIALIZED
        self.logger.info(f"Plugin initialized: {self.metadata.name}")

    async def start(self) -> None:
        """Start the plugin.
        
        This method should be overridden by specific plugins to start the plugin.
        """
        if self.state != PluginState.INITIALIZED:
            self.logger.warning(
                f"Cannot start plugin {self.metadata.name}: not initialized (state: {self.state})"
            )
            return
            
        self.state = PluginState.STARTING
        self.logger.info(f"Starting plugin: {self.metadata.name}")
        
        # Perform plugin-specific startup here
        
        self.state = PluginState.RUNNING
        self.logger.info(f"Plugin started: {self.metadata.name}")

    async def stop(self) -> None:
        """Stop the plugin.
        
        This method should be overridden by specific plugins to stop the plugin.
        """
        if self.state != PluginState.RUNNING:
            self.logger.warning(
                f"Cannot stop plugin {self.metadata.name}: not running (state: {self.state})"
            )
            return
            
        self.state = PluginState.STOPPING
        self.logger.info(f"Stopping plugin: {self.metadata.name}")
        
        # Perform plugin-specific shutdown here
        
        self.state = PluginState.STOPPED
        self.logger.info(f"Plugin stopped: {self.metadata.name}")

    def handle_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Handle an event.
        
        This method can be overridden by specific plugins to handle events.
        
        Args:
            event_name: Name of the event.
            event_data: Data for the event.
        """
        self.logger.debug(f"Received event: {event_name}")
        # Default implementation does nothing
