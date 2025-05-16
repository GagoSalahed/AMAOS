"""
AdaptiveControlNode for AMAOS.

This module provides an adaptive routing node that can dynamically adjust routing decisions
based on historical performance, feedback, and other metrics.
"""
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, Set, Tuple, TypeVar, Union, cast, Callable, overload, Mapping, TypedDict
from pydantic import BaseModel, Field, root_validator, validator
import json

from amaos.core.node_protocol import Node, NodeTask, NodeResult
from amaos.nodes.control_node import ControlNode
from amaos.utils.context_logger import ContextAwareLogger
from amaos.utils.context_tracker import ContextTracker

# Type variables for generic node handling
T_Node = TypeVar('T_Node', bound=Node)
T_Result = TypeVar('T_Result', bound=NodeResult)
T_Task = TypeVar('T_Task', bound=NodeTask)

# Handler type definitions
NodeHandler = Callable[[Node, NodeTask], NodeResult]
AsyncNodeHandler = Callable[[Node, NodeTask], Callable[[], NodeResult]]


class RoutingStrategy(str, Enum):
    """Routing strategy for adaptive node.
    
    Defines possible strategies for routing tasks to nodes.
    """
    ROUND_ROBIN = "round_robin"  # Simple round-robin scheduling
    PERFORMANCE = "performance"  # Based on success rate
    LATENCY = "latency"  # Based on response time
    FEEDBACK = "feedback"  # Based on feedback score
    ADAPTIVE = "adaptive"  # Combination of all factors


class PerformanceMetrics(BaseModel):
    """Performance metrics for a node."""
    
    success_count: int = Field(default=0, description="Number of successful task executions")
    failure_count: int = Field(default=0, description="Number of failed task executions")
    total_latency: float = Field(default=0.0, description="Cumulative latency across all executions")
    min_latency: Optional[float] = Field(default=None, description="Minimum observed latency")
    max_latency: Optional[float] = Field(default=None, description="Maximum observed latency")
    feedback_sum: float = Field(default=0.0, description="Sum of feedback scores")
    feedback_count: int = Field(default=0, description="Count of feedback submissions")
    last_used: Optional[float] = Field(default=None, description="Timestamp of last execution")
    
    @property
    def total_count(self) -> int:
        """Get total execution count.
        
        Returns:
            Total number of task executions (success + failure)
        """
        return self.success_count + self.failure_count
        
    @property
    def success_rate(self) -> float:
        """Get success rate.
        
        Returns:
            Success rate as fraction (0.0-1.0) or 0.0 if no executions
        """
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
        
    @property
    def avg_latency(self) -> float:
        """Get average latency.
        
        Returns:
            Average latency in seconds or 0.0 if no executions
        """
        if self.total_count == 0:
            return 0.0
        return self.total_latency / self.total_count
        
    @property
    def avg_feedback(self) -> float:
        """Get average feedback score.
        
        Returns:
            Average feedback score (0.0-1.0) or 0.0 if no feedback
        """
        if self.feedback_count == 0:
            return 0.0
        return self.feedback_sum / self.feedback_count


class AdaptiveNodeConfig(BaseModel):
    """Configuration for adaptive control node.
    
    Defines behavior parameters for the adaptive routing algorithm and persistence.
    """
    
    node_id: str = Field(default="adaptive_control_node", description="Unique identifier for this node")
    default_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.ADAPTIVE, 
        description="Default routing strategy when not specified in task metadata"
    )
    history_size: int = Field(
        default=1000, 
        description="Maximum number of historical task executions to retain", 
        ge=1, 
        le=100000
    )
    latency_weight: float = Field(
        default=0.3, 
        description="Weight given to latency in adaptive routing decisions",
        ge=0.0, 
        le=1.0
    )
    success_weight: float = Field(
        default=0.4, 
        description="Weight given to success rate in adaptive routing decisions",
        ge=0.0, 
        le=1.0
    )
    feedback_weight: float = Field(
        default=0.3, 
        description="Weight given to feedback in adaptive routing decisions",
        ge=0.0, 
        le=1.0
    )
    persist_metrics: bool = Field(
        default=True, 
        description="Whether to persist metrics to disk between runs"
    )
    metrics_file: str = Field(
        default="adaptive_metrics.json", 
        description="File path for persisted metrics"
    )
    learning_rate: float = Field(
        default=0.1, 
        description="Rate at which to adjust weights based on new data",
        ge=0.0, 
        le=1.0
    )
    
    @validator("latency_weight", "success_weight", "feedback_weight")
    def validate_weights(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate that weights sum approximately to 1.0.
        
        Args:
            v: The current weight value
            values: Values of other fields
            
        Returns:
            The validated weight value
            
        Raises:
            ValueError: If weights don't sum approximately to 1.0
        """
        # Only run this check when processing the feedback_weight (last of the three)
        field_name = list(cls.__fields__.keys())[-1]
        field_info = list(cls.__fields__.values())[-1]
        if field_info.alias != 'feedback_weight':
            return v
            
        # Get all three weights
        latency = values.get('latency_weight', 0.3)
        success = values.get('success_weight', 0.4)
        feedback = v
        
        # Check if sum is approximately 1.0 (allowing small float precision errors)
        total = latency + success + feedback
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weight values must sum to 1.0, got {total}")
            
        return v


class TaskHistory(BaseModel):
    """History entry for a task execution.
    
    Records details about a specific task execution for analysis and adaptation.
    """
    
    task_id: str = Field(..., description="Unique identifier for the task")
    task_type: str = Field(..., description="Type of task executed")
    node_id: str = Field(..., description="ID of the node that executed the task")
    success: bool = Field(..., description="Whether the task execution succeeded")
    latency: float = Field(..., description="Execution time in seconds", ge=0.0)
    timestamp: float = Field(..., description="Time when task was executed (epoch seconds)")
    end_time: Optional[float] = Field(default=None, description="Time when task finished (epoch seconds)")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    feedback: Optional[float] = Field(
        default=None, 
        description="Optional feedback score (0.0-1.0)",
        ge=0.0, 
        le=1.0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional execution metadata"
    )


class AdaptiveControlNode(ControlNode, Generic[T_Node]):
    """Adaptive control node that learns routing preferences over time.
    
    This node extends the basic ControlNode with:
    - Performance metrics tracking
    - Multiple routing strategies
    - Feedback incorporation
    - Task history for analysis
    
    Type Parameters:
        T: The type of Node this AdaptiveControlNode can route to
    """
    
    def __init__(
        self, 
        node_registry: Mapping[str, T_Node], 
        config: Optional[AdaptiveNodeConfig] = None,
        max_retries: int = 1
    ) -> None:
        """Initialize adaptive control node.
        
        Args:
            node_registry: Registry of nodes by task type
            config: Configuration for adaptive control
            max_retries: Maximum retries for failed tasks
        """
        # ControlNode expects Dict[str, Node], but we have Mapping[str, T_Node]
        # T_Node is a subtype of Node, so this is safe, but mypy needs help
        super().__init__(cast(Dict[str, Node], node_registry), max_retries)
        
        # Set up configuration and tracking
        self.config = config or AdaptiveNodeConfig()
        
        # Set up context-aware logger
        self.logger = ContextAwareLogger(f"adaptive_node.{self.config.node_id}")
        
        # Type-safe metrics and history storage
        self.metrics: Dict[str, Dict[str, PerformanceMetrics]] = {}
        self.history: List[TaskHistory] = []
        self.node_id_map: Dict[str, str] = {}
        self.last_node: Dict[str, int] = {}  # Track by index, not ID
        
        # Feedback storage
        self.feedback_store: Dict[str, float] = {}
        
        # Initialize metrics for each registered node
        for task_type, node in self.nodes.items():
            node_id = self._get_safe_node_id(node, task_type)  # type: ignore
            if task_type not in self.metrics:
                self.metrics[task_type] = {}
            if node_id not in self.metrics[task_type]:
                self.metrics[task_type][node_id] = PerformanceMetrics()
                
        # Load metrics from disk if enabled
        if self.config.persist_metrics:
            self._load_metrics()
        
        # Load persisted metrics if available
        if self.config.persist_metrics:
            self._load_metrics()
            
    async def initialize(self) -> None:
        """Initialize the node and all registered nodes.
        
        This method is called during system startup to initialize all registered nodes
        and prepare the metrics tracking system.
        """
        # Initialize all child nodes first
        await super().initialize()
        
        initialized_nodes = 0
        
        # Initialize metrics for all nodes with robust type handling
        for task_type, node in self.nodes.items():
            node_id = self._get_safe_node_id(node, task_type)  # type: ignore
            if task_type not in self.metrics:
                self.metrics[task_type] = {}
            if node_id not in self.metrics[task_type]:
                self.metrics[task_type][node_id] = PerformanceMetrics()
            initialized_nodes += 1
                
        # Log initialization with context information
        with ContextTracker.context({"action": "init", "node_id": self.config.node_id}):
            self.logger.info(f"Initialized adaptive node with {initialized_nodes} registered nodes")
        
    async def handle(self, task: NodeTask) -> NodeResult:
        """Handle a task with adaptive routing.
        
        This method routes the incoming task to the most appropriate node based on
        the selected routing strategy, tracks performance metrics, and handles errors.
        
        Args:
            task: Task to handle, must contain valid task_type
            
        Returns:
            Result of task execution with source information
        """
        start_time = time.time()
        task_id = getattr(task, "task_id", str(id(task)))
        task_type = task.task_type
        
        # Create a context for this task execution
        context_data = {
            "task_id": task_id,
            "task_type": task_type,
            "node_id": self.config.node_id
        }
        
        # Set up tracking for this task
        history_entry = TaskHistory(
            task_id=task_id,
            task_type=task_type,
            node_id=self.config.node_id,
            timestamp=start_time,
            success=False,  # Will be updated later
            latency=0.0     # Will be updated later
        )
        
        try:
            with ContextTracker.context(context_data):
                # Get candidate nodes for this task type
                candidate_nodes = self._get_nodes_for_task_type(task_type)
                
                if not candidate_nodes:
                    self.logger.warning(f"No handlers found for task type: {task_type}")
                    return NodeResult(
                        success=False,
                        result={"error": f"No handlers found for task type: {task_type}"},
                        source=self.config.node_id
                    )
                
                # Determine routing strategy from task metadata or config
                strategy_name = None
                if task.metadata is not None:
                    strategy_name = task.metadata.get("routing_strategy", self.config.default_strategy)
                else:
                    strategy_name = self.config.default_strategy
                    
                if isinstance(strategy_name, str) and strategy_name in RoutingStrategy.__members__:
                    strategy = RoutingStrategy(strategy_name)
                else:
                    strategy = self.config.default_strategy
                
                # Select node based on strategy
                selected_node = await self._select_node(task_type, candidate_nodes, strategy)
                
                if not selected_node:
                    self.logger.error(f"Failed to select node for task type: {task_type}")
                    return NodeResult(
                        success=False,
                        result={"error": f"Failed to select node for task type: {task_type}"},
                        source=self.config.node_id
                    )
                
                # Get node ID for metrics tracking
                selected_node_id = self._get_safe_node_id(selected_node, task_type)
                
                # Update context with selected node
                context_data["selected_node"] = selected_node_id
                
                # Execute task on selected node
                self.logger.info(f"Routing task {task_id} to node {selected_node_id} using strategy {strategy.value}")
                result = await selected_node.handle(task)
                
                # Update history entry with execution details
                execution_time = time.time() - start_time
                history_entry.success = result.success
                history_entry.latency = execution_time
                
                # Update metrics for selected node
                self._update_metrics(
                    task_type=task_type,
                    node_id=selected_node_id,
                    success=result.success,
                    latency=execution_time
                )
                
                # Append to history, respecting max size
                self.history.append(history_entry)
                if len(self.history) > self.config.history_size:
                    self.history = self.history[-self.config.history_size:]
                
                # Save metrics if configured
                if self.config.persist_metrics:
                    self._save_metrics()
                
                # Return result with extra tracking information
                if result.metadata is None:
                    result_metadata: Dict[str, Any] = {}
                else:
                    result_metadata = dict(result.metadata)
                
                result_metadata.update({
                    "routing_strategy": strategy.value,
                    "execution_time": execution_time,
                    "selected_node": selected_node_id
                })
                
                # Create new result with updated metadata
                return NodeResult(
                    success=result.success,
                    result=result.result,
                    source=result.source or self.config.node_id,
                    metadata=result_metadata
                )
                    
        except Exception as e:
            self.logger.error(f"Error handling task {task_id}: {str(e)}", exc_info=True)
            return NodeResult(
                success=False,
                result={"error": f"Error in adaptive node: {str(e)}"},
                source=self.config.node_id
            )
        finally:
            # Always update history
            execution_time = time.time() - start_time
            self.logger.debug(f"Task {task_id} processed in {execution_time:.4f}s")
    
    def _get_nodes_for_task_type(self, task_type: str) -> List[T_Node]:
        """Get all nodes that can handle a specific task type.
        
        Args:
            task_type: Task type to find handlers for
            
        Returns:
            List of nodes that can handle the task type
        """
        candidate_nodes: List[T_Node] = []
        
        # Check if task type exists in node registry
        if task_type not in self.nodes:
            self.logger.warning(f"No node registered for task type: {task_type}")
            return []
            
        # Get node entry from registry
        node_entry = self.nodes.get(task_type)
        
        if node_entry is not None:
            candidate_nodes.append(node_entry)  # type: ignore # Node vs T_Node issue
                
        return candidate_nodes
            
    def _get_safe_node_id(self, node: T_Node, fallback: str = "unknown") -> str:
        """Get a node ID safely with proper type checking.
        
        Args:
            node: The node to get ID from
            fallback: Fallback ID if no better ID is found
            
        Returns:
            A string identifier for the node
        """
        # Cast to Node type for safe access to common Node properties
        node_base = cast(Node, node)
        # Try object identity first (most reliable)
        obj_id = id(node_base)
        obj_id_str = str(obj_id)
        if obj_id_str in self.node_id_map:
            return self.node_id_map[obj_id_str]
        
        # Try node.id attribute
        if hasattr(node_base, "id") and isinstance(getattr(node_base, "id"), str):
            self.node_id_map[obj_id_str] = getattr(node_base, "id")
            return self.node_id_map[obj_id_str]
            
        # Try node.config.node_id
        if hasattr(node_base, "config"):
            config = getattr(node_base, "config")
            if hasattr(config, "node_id") and isinstance(getattr(config, "node_id"), str):
                self.node_id_map[obj_id_str] = getattr(config, "node_id")
                return self.node_id_map[obj_id_str]
                
        # Fall back to object ID + task type
        self.node_id_map[obj_id_str] = f"{fallback}_{obj_id}"
        return self.node_id_map[obj_id_str]

    def _update_metrics(self, task_type: str, node_id: str, success: bool, latency: float) -> None:
        """Update performance metrics for a node.
        
        Args:
            task_type: Task type that was executed
            node_id: ID of the node that executed the task
            success: Whether execution was successful
            latency: Execution time in seconds
        """
        # Initialize metrics dictionary for task type if needed
        if task_type not in self.metrics:
            self.metrics[task_type] = {}
            
        # Initialize metrics for node if needed
        if node_id not in self.metrics[task_type]:
            self.metrics[task_type][node_id] = PerformanceMetrics()
            
        # Get the metrics object
        metrics = self.metrics[task_type][node_id]
        
        # Update success/failure counts
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
            
        # Update latency metrics
        metrics.total_latency += latency
        
        # Update min/max latency
        if metrics.min_latency is None or latency < metrics.min_latency:
            metrics.min_latency = latency
            
        if metrics.max_latency is None or latency > metrics.max_latency:
            metrics.max_latency = latency
            
        # Update last used timestamp
        metrics.last_used = time.time()
        
    def _save_metrics(self) -> bool:
        """Save metrics to persistent storage if configured.
        
        Returns:
            Whether metrics were successfully saved
        """
        if not hasattr(self.config, "metrics_file") or not self.config.metrics_file:
            self.logger.warning("No metrics file configured, skipping save")
            return False
            
        try:
            # Convert metrics to serializable form
            metrics_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
            
            for task_type, nodes in self.metrics.items():
                metrics_data[task_type] = {}
                for node_id, metrics in nodes.items():
                    metrics_data[task_type][node_id] = metrics.dict()
            
            # Save to file
            with open(self.config.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            self.logger.debug(f"Saved metrics to {self.config.metrics_file}")
            return True
            
        except (IOError, OSError, TypeError) as e:
            self.logger.error(f"Failed to save metrics: {str(e)}", exc_info=True)
            return False
            
    async def provide_feedback(self, task_id: str, score: float) -> bool:
        """Provide feedback for a task execution.
        
        Args:
            task_id: Task ID to provide feedback for
            score: Feedback score (0-1 range)
            
        Returns:
            True if feedback was applied, False otherwise
        """
        # Find task in history
        matching_tasks = [t for t in self.history if t.task_id == task_id]
        
        if not matching_tasks:
            self.logger.warning(f"No task found with ID {task_id} for feedback")
            return False
            
        # Apply feedback to most recent matching task
        task = matching_tasks[-1]
        task.feedback = score
        
        # Update metrics if available
        if task.task_type in self.metrics and task.node_id in self.metrics[task.task_type]:
            metrics = self.metrics[task.task_type][task.node_id]
            metrics.feedback_sum += score
            metrics.feedback_count += 1
            
            # Save metrics
            if self.config.persist_metrics:
                self._save_metrics()
                
            self.logger.info(f"Applied feedback {score} to task {task_id}")
            return True
            
        return False
    
    async def _select_node(
        self, 
        task_type: str, 
        nodes: List[T_Node], 
        strategy: RoutingStrategy
    ) -> Optional[T_Node]:
        """Select the best node for a task based on the strategy.
        
        This method implements the core routing logic, using different strategies
        to select the optimal node based on historical performance data.
        
        Args:
            task_type: Type of task to route
            nodes: List of candidate nodes that can handle the task
            strategy: Routing strategy to use for selection
            
        Returns:
            Selected node or None if no suitable node found
        """
        if not nodes:
            self.logger.warning(f"No nodes available for task type: {task_type}")
            return None
            
        # If only one node, return it immediately (optimization)
        if len(nodes) == 1:
            return nodes[0]
            
        # Apply strategy with type safety
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(task_type, nodes)
            
        elif strategy == RoutingStrategy.PERFORMANCE:
            return self._select_by_performance(task_type, nodes)
            
        elif strategy == RoutingStrategy.LATENCY:
            return self._select_by_latency(task_type, nodes)
            
        elif strategy == RoutingStrategy.FEEDBACK:
            return self._select_by_feedback(task_type, nodes)
            
        else:  # ADAPTIVE
            return self._select_adaptive(task_type, nodes)
    
    def _select_round_robin(self, task_type: str, nodes: List[T_Node]) -> T_Node:
        """Select node using round-robin strategy.
        
        Args:
            task_type: Type of task
            nodes: List of candidate nodes
            
        Returns:
            Selected node using round-robin approach
        """
        # Simple round-robin with proper dict handling
        if task_type not in self.last_node:
            self.last_node[task_type] = -1
            
        # Handle edge case when nodes list has changed
        next_idx = (self.last_node[task_type] + 1) % len(nodes)
        self.last_node[task_type] = next_idx
        return nodes[next_idx]
    
    def _select_by_performance(self, task_type: str, nodes: List[T_Node]) -> T_Node:
        """Select node with best success rate.
        
        Args:
            task_type: Type of task
            nodes: List of candidate nodes
            
        Returns:
            Node with best success rate or first node if no data
        """
        best_node: Optional[T_Node] = None
        best_score: float = -1.0
        
        for node in nodes:
            node_id = self._get_safe_node_id(node, task_type)
            
            # Get metrics for this node
            if task_type in self.metrics and node_id in self.metrics[task_type]:
                metrics = self.metrics[task_type][node_id]
                score = metrics.success_rate
                
                # Select node with highest success rate
                if score > best_score:
                    best_score = score
                    best_node = node
                    
        # If we couldn't find a best node, fall back to first one
        if best_node is None and nodes:
            best_node = nodes[0]
            
        return best_node or nodes[0]  # Type safety: ensure non-null return
    
    def _select_by_latency(self, task_type: str, nodes: List[T_Node]) -> T_Node:
        """Select node with best (lowest) latency.
        
        Args:
            task_type: Type of task
            nodes: List of candidate nodes
            
        Returns:
            Node with best latency or first node if no data
        """
        best_node: Optional[T_Node] = None
        best_latency: float = float('inf')
        
        for node in nodes:
            node_id = self._get_safe_node_id(node, task_type)
            
            # Get metrics for this node
            if task_type in self.metrics and node_id in self.metrics[task_type]:
                metrics = self.metrics[task_type][node_id]
                
                # Only consider nodes with at least one successful execution
                if metrics.total_count > 0:
                    latency = metrics.avg_latency
                    
                    # Select node with lowest latency
                    if latency < best_latency:
                        best_latency = latency
                        best_node = node
                        
        # If we couldn't find a best node, fall back to first one
        if best_node is None:
            best_node = nodes[0]
            
        return best_node
    
    def _select_by_feedback(self, task_type: str, nodes: List[T_Node]) -> T_Node:
        """Select node with best feedback score.
        
        Args:
            task_type: Type of task
            nodes: List of candidate nodes
            
        Returns:
            Node with best feedback or first node if no data
        """
        best_node: Optional[T_Node] = None
        best_score: float = -1.0
        
        for node in nodes:
            node_id = self._get_safe_node_id(node, task_type)
            
            # Get metrics for this node
            if task_type in self.metrics and node_id in self.metrics[task_type]:
                metrics = self.metrics[task_type][node_id]
                
                # Only consider nodes with at least one feedback
                if metrics.feedback_count > 0:
                    score = metrics.avg_feedback
                    
                    # Select node with highest feedback score
                    if score > best_score:
                        best_score = score
                        best_node = node
                    
        # If we couldn't find a best node, fall back to first one
        if best_node is None:
            best_node = nodes[0]
            
        return best_node
    
    def _select_adaptive(self, task_type: str, nodes: List[T_Node]) -> T_Node:
        """Select node using combined weighted metrics.
        
        Args:
            task_type: Type of task
            nodes: List of candidate nodes
            
        Returns:
            Node with best combined score or first node if no data
        """
        best_node: Optional[T_Node] = None
        best_score: float = -1.0
        
        for node in nodes:
            node_id = self._get_safe_node_id(node, task_type)
            
            # Get metrics for this node
            if task_type in self.metrics and node_id in self.metrics[task_type]:
                metrics = self.metrics[task_type][node_id]
                
                # Calculate combined score based on configured weights
                score = 0.0
                
                # Add success rate component
                if metrics.total_count > 0:
                    success_component = metrics.success_rate * self.config.success_weight
                    score += success_component
                
                # Add latency component (inverse - lower is better)
                if metrics.total_count > 0 and metrics.avg_latency > 0:
                    # Normalize latency to 0-1 scale (1 is best/fastest)
                    # We use a simple inverse function with a cap to avoid division by zero
                    MAX_REASONABLE_LATENCY = 10.0  # 10 seconds is considered very slow
                    normalized_latency = max(0.0, 1.0 - (metrics.avg_latency / MAX_REASONABLE_LATENCY))
                    latency_component = normalized_latency * self.config.latency_weight
                    score += latency_component
                
                # Add feedback component
                if metrics.feedback_count > 0:
                    feedback_component = metrics.avg_feedback * self.config.feedback_weight
                    score += feedback_component
                
                # Update best node if this one has a better score
                if score > best_score:
                    best_score = score
                    best_node = node
                    
        # If we couldn't find a best node, fall back to first one
        if best_node is None:
            return nodes[0]
                
        return best_node
    
    def _load_metrics(self) -> bool:
        """Load metrics from disk.
        
        Deserializes performance metrics from the configured file and updates the metrics
        dictionary with the loaded values.
        
        Returns:
            Boolean indicating whether the load operation was successful
        """
        if not self.config.persist_metrics:
            return False
            
        metrics_file_path = self.config.metrics_file
        if not os.path.exists(metrics_file_path):
            self.logger.info(f"No metrics file found at {metrics_file_path}")
            return False
            
        try:
            with open(metrics_file_path, 'r') as f:
                metrics_data = json.load(f)
                
            # Convert loaded data back to PerformanceMetrics objects
            for task_type, nodes in metrics_data.items():
                if task_type not in self.metrics:
                    self.metrics[task_type] = {}
                    
                for node_id, data in nodes.items():
                    self.metrics[task_type][node_id] = PerformanceMetrics(**data)
                    
            self.logger.info(f"Loaded metrics from {metrics_file_path}")
            return True
            
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to load metrics: {str(e)}", exc_info=True)
            return False
            
    async def clear_history(self) -> None:
        """Clear task history."""
        self.history = []
        self.logger.info("Task history cleared")
        
    async def reset_metrics(self, task_type: Optional[str] = None, node_id: Optional[str] = None) -> bool:
        """Reset performance metrics.
        
        This method resets the performance metrics for the specified task type and node ID.
        If task_type is None, resets metrics for all task types.
        If node_id is None, resets metrics for all nodes of the specified task type.
        If both are None, resets all metrics.
        
        Args:
            task_type: Optional task type to reset metrics for
            node_id: Optional node ID to reset metrics for
            
        Returns:
            Boolean indicating if any metrics were reset
        """
        # Create context for logging
        context = {"action": "reset_metrics"}
        if task_type is not None:
            context["task_type"] = task_type
        if node_id is not None:
            context["node_id"] = node_id
            
        with ContextTracker.context(context):
            if task_type is None and node_id is None:
                # Reset all metrics
                self.metrics = {}
                self.logger.info("Reset all metrics")
                return True
                
            elif task_type is not None and node_id is None:
                # Reset metrics for specific task type
                if task_type in self.metrics:
                    self.metrics[task_type] = {}
                    self.logger.info(f"Reset metrics for task type: {task_type}")
                    return True
                else:
                    self.logger.warning(f"No metrics found for task type: {task_type}")
                    return False
                    
            elif task_type is not None and node_id is not None:
                # Reset metrics for specific task type and node
                if task_type in self.metrics and node_id in self.metrics[task_type]:
                    self.metrics[task_type][node_id] = PerformanceMetrics()
                    self.logger.info(f"Reset metrics for task type: {task_type}, node: {node_id}")
                    return True
                else:
                    self.logger.warning(f"No metrics found for task type: {task_type}, node: {node_id}")
                    return False
                    
            else:  # task_type is None and node_id is not None
                # Reset metrics for specific node across all task types
                reset_count = 0
                for task_type, nodes in self.metrics.items():
                    if node_id in nodes:
                        self.metrics[task_type][node_id] = PerformanceMetrics()
                        reset_count += 1
                        
                if reset_count > 0:
                    self.logger.info(f"Reset metrics for node: {node_id} across {reset_count} task types")
                    return True
                else:
                    self.logger.warning(f"No metrics found for node: {node_id}")
                    return False
        
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics.
        
        Returns detailed statistics about this node's performance metrics,
        including aggregated stats across all nodes and task-specific breakdowns.
        
        Returns:
            Dictionary containing node statistics and metrics
        """
        # Create base stats with explicit typing
        stats: Dict[str, Any] = {
            "node_id": self.config.node_id,
            "strategy": self.config.default_strategy.value,
            "history_size": len(self.history),
            "metrics": {},
            "weights": {
                "success": self.config.success_weight,
                "latency": self.config.latency_weight,
                "feedback": self.config.feedback_weight
            },
            "node_count": len(self.nodes)
        }
        
        # Calculate aggregated metrics for each task type
        for task_type, nodes in self.metrics.items():
            stats["metrics"][task_type] = {
                "total_tasks": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "avg_feedback": 0.0,
                "nodes": {}
            }
            
            task_success_count = 0
            task_failure_count = 0
            task_latency_sum = 0.0
            task_feedback_sum = 0.0
            task_feedback_count = 0
            
            # Aggregate metrics across all nodes for this task type
            for node_id, metrics in nodes.items():
                # Add node-specific metrics
                stats["metrics"][task_type]["nodes"][node_id] = {
                    "success_count": metrics.success_count,
                    "failure_count": metrics.failure_count,
                    "total_count": metrics.total_count,
                    "success_rate": metrics.success_rate,
                    "avg_latency": metrics.avg_latency,
                    "avg_feedback": metrics.avg_feedback,
                    "last_used": metrics.last_used
                }
                
                # Update task-level aggregates
                task_success_count += metrics.success_count
                task_failure_count += metrics.failure_count
                task_latency_sum += metrics.total_latency
                task_feedback_sum += metrics.feedback_sum
                task_feedback_count += metrics.feedback_count
                
            # Calculate task-level aggregated metrics
            total_tasks = task_success_count + task_failure_count
            stats["metrics"][task_type]["total_tasks"] = total_tasks
            stats["metrics"][task_type]["success_count"] = task_success_count
            stats["metrics"][task_type]["failure_count"] = task_failure_count
            
            if total_tasks > 0:
                stats["metrics"][task_type]["success_rate"] = task_success_count / total_tasks
                stats["metrics"][task_type]["avg_latency"] = task_latency_sum / total_tasks
                
            if task_feedback_count > 0:
                stats["metrics"][task_type]["avg_feedback"] = task_feedback_sum / task_feedback_count
            
        return stats
        
    async def get_node_recommendations(self, task_type: str) -> Dict[str, float]:
        """Get node recommendations for a task type.
        
        This method analyzes the performance metrics for all nodes that can handle
        the specified task type and returns a dictionary of node IDs to scores.
        
        Args:
            task_type: Task type to get recommendations for
            
        Returns:
            Dictionary mapping node IDs to recommendation scores (0-1)
        """
        recommendations: Dict[str, float] = {}
        
        # Get candidate nodes
        candidate_nodes = self._get_nodes_for_task_type(task_type)
        
        if not candidate_nodes:
            self.logger.warning(f"No nodes available for task type: {task_type}")
            return recommendations
            
        # Calculate scores for each node
        for node in candidate_nodes:
            node_id = self._get_safe_node_id(node, task_type)
            
            # Default score for nodes with no metrics
            score = 0.5
            
            # Get metrics if available
            if task_type in self.metrics and node_id in self.metrics[task_type]:
                metrics = self.metrics[task_type][node_id]
                
                # Only calculate score if we have some data
                if metrics.total_count > 0:
                    # Calculate combined score using adaptive strategy weights
                    success_component = metrics.success_rate * self.config.success_weight
                    
                    # Normalize latency (lower is better)
                    MAX_REASONABLE_LATENCY = 10.0
                    normalized_latency = max(0.0, 1.0 - (metrics.avg_latency / MAX_REASONABLE_LATENCY))
                    latency_component = normalized_latency * self.config.latency_weight
                    
                    # Add feedback component if available
                    feedback_component = 0.0
                    if metrics.feedback_count > 0:
                        feedback_component = metrics.avg_feedback * self.config.feedback_weight
                        
                    # Combine components
                    score = success_component + latency_component + feedback_component
                    
            # Add to recommendations
            recommendations[node_id] = score
            
        return recommendations
