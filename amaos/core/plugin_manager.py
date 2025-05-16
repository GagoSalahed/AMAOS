"""Plugin manager for AMAOS.

This module provides a plugin manager for loading, initializing, and managing plugins.
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set, Type, cast

import prometheus_client as prom
from pydantic import BaseModel

from amaos.core.plugin_interfaces import BasePluginInterface, PluginInterface, PluginMetadata, PluginState


class PluginManagerConfig(BaseModel):
    """Configuration for the plugin manager."""

    plugin_dirs: List[str] = ["plugins"]
    auto_discover: bool = True
    auto_initialize: bool = True
    auto_start: bool = False


class PluginManager:
    """Plugin manager for loading, initializing, and managing plugins."""

    def __init__(self, config: Optional[PluginManagerConfig] = None) -> None:
        """Initialize the plugin manager.

        Args:
            config: Configuration for the plugin manager.
        """
        self.config = config or PluginManagerConfig()
        self.plugins: Dict[str, PluginInterface] = {}
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.plugin_count = prom.Gauge(
            "amaos_plugin_manager_plugin_count", "Number of plugins registered"
        )
        self.plugin_state = prom.Gauge(
            "amaos_plugin_manager_plugin_state", 
            "State of plugins", 
            ["plugin_name", "state"]
        )
        self.plugin_load_errors = prom.Counter(
            "amaos_plugin_manager_load_errors", 
            "Number of plugin load errors",
            ["plugin_name"]
        )

    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        if self.config.auto_discover:
            await self.discover_plugins()
        
        if self.config.auto_initialize:
            await self.initialize_all_plugins()
            
        if self.config.auto_start:
            await self.start_all_plugins()

    async def discover_plugins(self) -> None:
        """Discover plugins in the plugin directories."""
        for plugin_dir in self.config.plugin_dirs:
            if not os.path.exists(plugin_dir):
                self.logger.warning(f"Plugin directory {plugin_dir} does not exist")
                continue
                
            # Add plugin directory to path if not already there
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
                
            # Find all Python files in the directory
            for filename in os.listdir(plugin_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    module_name = filename[:-3]  # Remove .py extension
                    await self._load_plugin_from_module(module_name)

    async def _load_plugin_from_module(self, module_name: str) -> None:
        """Load a plugin from a module.
        
        Args:
            module_name: Name of the module to load.
        """
        try:
            module = importlib.import_module(module_name)
            
            # Find all plugin classes in the module
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj) 
                    and issubclass(obj, BasePluginInterface) 
                    and obj != BasePluginInterface
                ):
                    plugin_class = cast(Type[PluginInterface], obj)
                    
                    # Create an instance of the plugin
                    plugin_instance = self._instantiate_plugin(plugin_class)
                    if plugin_instance:
                        self.register_plugin(plugin_instance)
        
        except Exception as e:
            self.logger.error(f"Error loading plugin from module {module_name}: {e}")
            self.plugin_load_errors.labels(plugin_name=module_name).inc()

    def _instantiate_plugin(self, plugin_class: Type[PluginInterface]) -> Optional[PluginInterface]:
        """Instantiate a plugin from a class.
        
        Args:
            plugin_class: Class to instantiate.
            
        Returns:
            An instance of the plugin, or None if instantiation failed.
        """
        try:
            # Check if the class has required attributes
            if not hasattr(plugin_class, "metadata") or not isinstance(
                getattr(plugin_class, "metadata"), PluginMetadata
            ):
                self.logger.warning(
                    f"Plugin class {plugin_class.__name__} does not have valid metadata"
                )
                return None
                
            # Create an instance of the plugin
            plugin_instance = plugin_class()
            return plugin_instance
            
        except Exception as e:
            self.logger.error(f"Error instantiating plugin {plugin_class.__name__}: {e}")
            self.plugin_load_errors.labels(plugin_name=plugin_class.__name__).inc()
            return None

    def register_plugin(self, plugin: PluginInterface) -> None:
        """Register a plugin with the plugin manager.
        
        Args:
            plugin: Plugin to register.
        """
        plugin_name = plugin.metadata.name
        
        if plugin_name in self.plugins:
            self.logger.warning(f"Plugin {plugin_name} already registered, replacing")
            
        self.plugins[plugin_name] = plugin
        self.plugin_count.set(len(self.plugins))
        self._update_plugin_state_metric(plugin)
        
        self.logger.info(f"Registered plugin: {plugin_name} v{plugin.metadata.version}")

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin from the plugin manager.
        
        Args:
            plugin_name: Name of the plugin to unregister.
        """
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            
            # Clear metrics for this plugin
            for state in PluginState:
                self.plugin_state.remove(plugin_name, state.name)
                
            del self.plugins[plugin_name]
            self.plugin_count.set(len(self.plugins))
            
            self.logger.info(f"Unregistered plugin: {plugin_name}")
        else:
            self.logger.warning(f"Cannot unregister plugin {plugin_name}: not found")

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to get.
            
        Returns:
            The plugin instance, or None if not found.
        """
        return self.plugins.get(plugin_name)

    def get_plugins_by_tag(self, tag: str) -> List[PluginInterface]:
        """Get plugins by tag.
        
        Args:
            tag: Tag to filter plugins by.
            
        Returns:
            List of plugin instances with the specified tag.
        """
        return [
            plugin for plugin in self.plugins.values() 
            if tag in plugin.metadata.tags
        ]

    async def initialize_plugin(self, plugin_name: str) -> bool:
        """Initialize a plugin.
        
        Args:
            plugin_name: Name of the plugin to initialize.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            self.logger.warning(f"Cannot initialize plugin {plugin_name}: not found")
            return False
            
        try:
            if plugin.state != PluginState.UNINITIALIZED:
                self.logger.warning(
                    f"Plugin {plugin_name} is already initialized (state: {plugin.state})"
                )
                return True
                
            await plugin.initialize()
            self._update_plugin_state_metric(plugin)
            
            self.logger.info(f"Initialized plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing plugin {plugin_name}: {e}")
            return False

    async def initialize_all_plugins(self) -> None:
        """Initialize all registered plugins."""
        for plugin_name in list(self.plugins.keys()):
            await self.initialize_plugin(plugin_name)

    async def start_plugin(self, plugin_name: str) -> bool:
        """Start a plugin.
        
        Args:
            plugin_name: Name of the plugin to start.
            
        Returns:
            True if starting was successful, False otherwise.
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            self.logger.warning(f"Cannot start plugin {plugin_name}: not found")
            return False
            
        try:
            if plugin.state == PluginState.RUNNING:
                self.logger.warning(f"Plugin {plugin_name} is already running")
                return True
                
            if plugin.state == PluginState.UNINITIALIZED:
                success = await self.initialize_plugin(plugin_name)
                if not success:
                    return False
                    
            await plugin.start()
            self._update_plugin_state_metric(plugin)
            
            self.logger.info(f"Started plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting plugin {plugin_name}: {e}")
            return False

    async def start_all_plugins(self) -> None:
        """Start all registered plugins."""
        for plugin_name in list(self.plugins.keys()):
            if self.plugins[plugin_name].config.auto_start:
                await self.start_plugin(plugin_name)

    async def stop_plugin(self, plugin_name: str) -> bool:
        """Stop a plugin.
        
        Args:
            plugin_name: Name of the plugin to stop.
            
        Returns:
            True if stopping was successful, False otherwise.
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            self.logger.warning(f"Cannot stop plugin {plugin_name}: not found")
            return False
            
        try:
            if plugin.state != PluginState.RUNNING:
                self.logger.warning(
                    f"Plugin {plugin_name} is not running (state: {plugin.state})"
                )
                return True
                
            await plugin.stop()
            self._update_plugin_state_metric(plugin)
            
            self.logger.info(f"Stopped plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping plugin {plugin_name}: {e}")
            return False

    async def stop_all_plugins(self) -> None:
        """Stop all registered plugins."""
        for plugin_name in list(self.plugins.keys()):
            await self.stop_plugin(plugin_name)

    def _update_plugin_state_metric(self, plugin: PluginInterface) -> None:
        """Update the plugin state metric.
        
        Args:
            plugin: Plugin to update the metric for.
        """
        plugin_name = plugin.metadata.name
        
        # Clear previous state
        for state in PluginState:
            try:
                self.plugin_state.remove(plugin_name, state.name)
            except (KeyError, ValueError):
                pass
                
        # Set current state
        self.plugin_state.labels(
            plugin_name=plugin_name, 
            state=plugin.state.name
        ).set(1)
