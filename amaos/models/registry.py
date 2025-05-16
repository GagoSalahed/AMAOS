"""Registry module for AMAOS.

This module provides a central registry of models and tasks used in the AMAOS system.
"""

from typing import Dict, Type, Mapping, Any, Callable, TypeVar, cast
from pydantic import BaseModel

from .tool import ToolTask, ToolResult
from .config import BaseConfig, SystemConfig
from amaos.memory.interface import MemoryEntry as MemoryItem, MemoryQueryResult
from amaos.core.node_protocol import NodeTask, NodeResult

# Type variable for registry entries that are subclasses of BaseModel
T = TypeVar('T', bound=BaseModel)

# Core model registry
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "llm": NodeTask,
    "tool": ToolTask,
    "memory": MemoryItem,
    "config": SystemConfig,
}

# Registry for result types
RESULT_REGISTRY: Dict[str, Type[BaseModel]] = {
    "llm": NodeResult,
    "tool": ToolResult,
    "memory": MemoryQueryResult,
}

def register_model(model_type: str, model_class: Type[T]) -> None:
    """Register a model class for a specific type.
    
    Args:
        model_type: The type identifier for the model.
        model_class: The model class to register.
    """
    MODEL_REGISTRY[model_type] = model_class

def get_model(model_type: str) -> Type[BaseModel]:
    """Get the model class for a specific type.
    
    Args:
        model_type: The type identifier for the model.
        
    Returns:
        The registered model class.
        
    Raises:
        KeyError: If the model type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"Model type '{model_type}' not registered.")
    return MODEL_REGISTRY[model_type]

def register_result_type(model_type: str, result_class: Type[T]) -> None:
    """Register a result model class for a specific type.
    
    Args:
        model_type: The type identifier for the result model.
        result_class: The result model class to register.
    """
    RESULT_REGISTRY[model_type] = result_class

def get_result_type(model_type: str) -> Type[BaseModel]:
    """Get the result model class for a specific type.
    
    Args:
        model_type: The type identifier for the result model.
        
    Returns:
        The registered result model class.
        
    Raises:
        KeyError: If the result model type is not registered.
    """
    if model_type not in RESULT_REGISTRY:
        raise KeyError(f"Result type '{model_type}' not registered.")
    return RESULT_REGISTRY[model_type]
