"""User Input Node for AMAOS.

This module provides the User Input Node component responsible for accepting
human input, approval, and correction.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol, Callable, Union, Literal

from pydantic import BaseModel, Field


from enum import Enum

class InputType(str, Enum):
    """Type of user input."""
    TEXT = "text"  # Free-form text input
    CHOICE = "choice"  # Selection from predefined choices
    APPROVAL = "approval"  # Yes/no approval
    CORRECTION = "correction"  # Correction of previous output
    FILE = "file"  # File upload


class UserInputRequest(BaseModel):
    """Request for user input."""
    
    request_id: str
    input_type: InputType
    prompt: str
    choices: Optional[List[str]] = None
    timeout: Optional[float] = None
    default_value: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserInputResponse(BaseModel):
    """Response from user input."""
    
    request_id: str
    input_type: InputType
    value: Any
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserInputNodeConfig(BaseModel):
    """Configuration for the User Input Node."""
    
    node_id: str = "user_input_node"
    default_timeout: float = 300.0  # Default timeout in seconds
    auto_approve_safe_actions: bool = False


class UserInputNode:
    """Node responsible for handling user input, approval, and correction.
    
    This node handles:
    - Collecting user input in various formats
    - Managing approval workflows
    - Processing corrections and feedback
    - Timeout and default value handling
    """
    
    def __init__(self, config: Optional[UserInputNodeConfig] = None) -> None:
        """Initialize the User Input Node.
        
        Args:
            config: Configuration for the User Input Node.
        """
        self.config = config or UserInputNodeConfig()
        self.logger = logging.getLogger(__name__)
        self.node_id = self.config.node_id
        self.pending_requests: Dict[str, UserInputRequest] = {}
        self.input_callbacks: Dict[str, Callable[[UserInputResponse], None]] = {}
        
    async def initialize(self) -> None:
        """Initialize the node."""
        self.logger.info(f"Initializing {self.node_id}")
        
    async def start(self) -> None:
        """Start the node."""
        self.logger.info(f"Starting {self.node_id}")
        
    async def stop(self) -> None:
        """Stop the node."""
        self.logger.info(f"Stopping {self.node_id}")
        
    async def request_input(self, request: UserInputRequest) -> UserInputResponse:
        """Request input from the user.
        
        Args:
            request: Input request details.
            
        Returns:
            User input response.
        """
        self.logger.info(f"Requesting user input: {request.prompt}")
        
        # Store the request
        self.pending_requests[request.request_id] = request
        
        # In a real implementation, we would wait for user input
        # For now, just return a mock response after a short delay
        await asyncio.sleep(0.5)
        
        # Mock user response
        if request.input_type == InputType.CHOICE and request.choices:
            value = request.choices[0]
        elif request.input_type == InputType.APPROVAL:
            value = "approved"
        elif request.input_type == InputType.TEXT:
            value = "Mock user input"
        else:
            value = request.default_value or "Mock user input"
            
        response = UserInputResponse(
            request_id=request.request_id,
            input_type=request.input_type,
            value=value,
            timestamp=asyncio.get_event_loop().time(),
            metadata={"mock": True},
        )
        
        # Remove the request
        self.pending_requests.pop(request.request_id, None)
        
        return response
        
    async def request_approval(
        self, action_description: str, timeout: Optional[float] = None
    ) -> bool:
        """Request approval from the user for an action.
        
        Args:
            action_description: Description of the action to approve.
            timeout: Timeout in seconds.
            
        Returns:
            True if approved, False otherwise.
        """
        # If auto-approve is enabled for safe actions, check if this is a safe action
        if self.config.auto_approve_safe_actions and self._is_safe_action(action_description):
            self.logger.info(f"Auto-approving safe action: {action_description}")
            return True
            
        # Otherwise, request explicit approval
        request = UserInputRequest(
            request_id=f"approval_{id(action_description)}",
            input_type=InputType.APPROVAL,
            prompt=f"Approve action: {action_description}",
            timeout=timeout or self.config.default_timeout,
        )
        
        response = await self.request_input(request)
        return bool(response.value)
        
    def _is_safe_action(self, action_description: str) -> bool:
        """Check if an action is safe for auto-approval.
        
        Args:
            action_description: Description of the action.
            
        Returns:
            True if the action is safe, False otherwise.
        """
        # In a real implementation, we would have more sophisticated logic
        # For now, just check for some keywords
        unsafe_keywords = ["delete", "remove", "drop", "reset", "overwrite", "format"]
        return not any(keyword in action_description.lower() for keyword in unsafe_keywords)
