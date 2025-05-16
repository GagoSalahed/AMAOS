"""Tool models for AMAOS.

This module defines models for tool tasks and results used in the AMAOS system.
"""

from typing import Any, Dict, Optional, List, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


class ToolInput(BaseModel):
    """Base model for tool inputs, which can be extended by specific tools."""
    
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for extensibility
        validate_assignment=True  # Validate on attribute assignment
    )


class ToolTask(BaseModel):
    """Model representing a tool task.
    
    This model is used to represent a request to execute a specific tool
    with the provided input parameters.
    """
    
    tool: str = Field(
        description="The name of the tool to execute"
    )
    input: Dict[str, Any] = Field(
        default_factory=dict,
        description="The input parameters for the tool"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional context for tool execution"
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        description="Optional timeout for tool execution in seconds"
    )
    

class ToolResult(BaseModel):
    """Model representing the result of a tool execution.
    
    This model captures the success/failure status, result data,
    and any error information from tool execution.
    """
    
    success: bool = Field(
        description="Whether the tool execution was successful"
    )
    result: Dict[str, Any] = Field(
        description="The result data from the tool execution"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the tool execution failed"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error that occurred, if any"
    )
    execution_time: Optional[float] = Field(
        default=None,
        description="Time taken to execute the tool, in seconds"
    )


class ToolRegistry(BaseModel):
    """Model for storing information about available tools."""
    
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters the tool accepts"
    )
    required_permissions: List[str] = Field(
        default_factory=list,
        description="Permissions required to use this tool"
    )
    is_async: bool = Field(
        default=True, 
        description="Whether this tool executes asynchronously"
    )
