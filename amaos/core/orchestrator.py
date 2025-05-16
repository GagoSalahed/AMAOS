"""Orchestrator implementation for AMAOS.

This module provides an orchestrator for routing tasks to agents and managing task execution.
"""

import asyncio
import logging
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import prometheus_client as prom
from pydantic import BaseModel, Field

from amaos.agents.base_agent import AgentContext, AgentResult, BaseAgent
from amaos.core.node import Node, NodeConfig, NodeState


class TaskStatus(str, Enum):
    """Enum representing the status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Enum representing the type of a task."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class TaskConfig(BaseModel):
    """Configuration for a task."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    agent_id: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Model representing a task."""

    config: TaskConfig
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[AgentResult] = None
    created_at: float = Field(default_factory=lambda: asyncio.get_event_loop().time())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Orchestrator"
    description: str = "Task orchestrator for AMAOS"
    max_concurrent_tasks: int = 10
    default_timeout: float = 60.0
    retry_failed_tasks: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    config: Dict[str, Any] = Field(default_factory=dict)


class Orchestrator(Node):
    """Orchestrator for routing tasks to agents and managing task execution."""

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        """Initialize the orchestrator.
        
        Args:
            config: Configuration for the orchestrator.
        """
        self.orchestrator_config = config or OrchestratorConfig()
        
        node_config = NodeConfig(
            id=self.orchestrator_config.id,
            name=self.orchestrator_config.name,
            description=self.orchestrator_config.description,
            config=self.orchestrator_config.config,
        )
        
        super().__init__(node_config)
        
        # Registered agents
        self.agents: Dict[str, BaseAgent] = {}
        
        # Tasks
        self.tasks: Dict[str, Task] = {}
        
        # Task queue and semaphore
        self.task_queue: asyncio.Queue[str] = asyncio.Queue()
        self.task_semaphore = asyncio.Semaphore(self.orchestrator_config.max_concurrent_tasks)
        
        # Task worker task
        self.task_worker_task: Optional[asyncio.Task[None]] = None
        
        # Additional metrics
        self.orchestrator_metrics = self._setup_orchestrator_metrics()

    def _setup_orchestrator_metrics(self) -> Dict[str, Any]:
        """Set up orchestrator-specific metrics.
        
        Returns:
            Dictionary of metrics.
        """
        metrics: Dict[str, Any] = {}
        
        # Task count metrics
        metrics["tasks"] = prom.Gauge(
            "amaos_orchestrator_tasks",
            "Number of tasks by status",
            ["status"]
        )
        
        # Task execution time
        metrics["task_execution_time"] = prom.Histogram(
            "amaos_orchestrator_task_execution_time",
            "Time taken to execute tasks",
            ["agent_id", "status"]
        )
        
        # Agent count
        metrics["agents"] = prom.Gauge(
            "amaos_orchestrator_agents",
            "Number of registered agents"
        )
        
        # Queue size
        metrics["queue_size"] = prom.Gauge(
            "amaos_orchestrator_queue_size",
            "Size of the task queue"
        )
        
        return metrics

    def _update_task_metrics(self) -> None:
        """Update task metrics."""
        # Count tasks by status
        status_counts: Dict[TaskStatus, int] = {status: 0 for status in TaskStatus}
        
        for task in self.tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
            
        # Update metrics
        for status, count in status_counts.items():
            self.orchestrator_metrics["tasks"].labels(status=status).set(count)
            
        # Update queue size
        self.orchestrator_metrics["queue_size"].set(self.task_queue.qsize())

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        await super().initialize()
        self.logger.info(f"Initializing orchestrator: {self.name} ({self.id})")
        
        # Initialize metrics
        self._update_task_metrics()
        self.orchestrator_metrics["agents"].set(0)

    async def start(self) -> None:
        """Start the orchestrator."""
        await super().start()
        self.logger.info(f"Starting orchestrator: {self.name} ({self.id})")
        
        # Start task worker
        self.task_worker_task = asyncio.create_task(self._task_worker())

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self.logger.info(f"Stopping orchestrator: {self.name} ({self.id})")
        
        # Cancel task worker
        if self.task_worker_task:
            self.task_worker_task.cancel()
            try:
                await self.task_worker_task
            except asyncio.CancelledError:
                pass
            self.task_worker_task = None
            
        # Cancel all running tasks
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.RUNNING:
                self.logger.info(f"Cancelling task: {task_id}")
                task.status = TaskStatus.CANCELLED
                task.error = "Orchestrator stopped"
                
        await super().stop()

    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.
        
        Args:
            agent: Agent to register.
        """
        self.logger.info(f"Registering agent: {agent.name} ({agent.id})")
        
        # Initialize the agent if not already initialized
        if agent.state == NodeState.UNINITIALIZED:
            await agent.initialize()
            
        # Register the agent
        self.agents[agent.id] = agent
        
        # Update metrics
        self.orchestrator_metrics["agents"].set(len(self.agents))

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator.
        
        Args:
            agent_id: ID of the agent to unregister.
        """
        if agent_id in self.agents:
            self.logger.info(f"Unregistering agent: {agent_id}")
            del self.agents[agent_id]
            
            # Update metrics
            self.orchestrator_metrics["agents"].set(len(self.agents))
        else:
            self.logger.warning(f"Cannot unregister agent {agent_id}: not found")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to get.
            
        Returns:
            The agent, or None if not found.
        """
        return self.agents.get(agent_id)

    async def submit_task(self, task_config: TaskConfig) -> str:
        """Submit a task for execution.
        
        Args:
            task_config: Configuration for the task.
            
        Returns:
            ID of the submitted task.
            
        Raises:
            ValueError: If the agent is not registered.
        """
        self.logger.info(f"Submitting task: {task_config.name} ({task_config.id})")
        
        # Check if the agent is registered
        if task_config.agent_id and task_config.agent_id not in self.agents:
            raise ValueError(f"Agent {task_config.agent_id} not registered")
            
        # Create the task
        task = Task(config=task_config)
        self.tasks[task_config.id] = task
        
        # Update metrics
        self._update_task_metrics()
        
        # Add to queue if no dependencies
        if not task_config.dependencies:
            await self.task_queue.put(task_config.id)
            
        return task_config.id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: ID of the task to cancel.
            
        Returns:
            True if the task was cancelled, False if it wasn't found or already completed.
        """
        if task_id not in self.tasks:
            self.logger.warning(f"Cannot cancel task {task_id}: not found")
            return False
            
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.logger.warning(
                f"Cannot cancel task {task_id}: already {task.status}"
            )
            return False
            
        self.logger.info(f"Cancelling task: {task_id}")
        task.status = TaskStatus.CANCELLED
        task.error = "Task cancelled"
        
        # Update metrics
        self._update_task_metrics()
        
        return True

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: ID of the task to get.
            
        Returns:
            The task, or None if not found.
        """
        return self.tasks.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get tasks by status.
        
        Args:
            status: Status to filter tasks by.
            
        Returns:
            List of tasks with the specified status.
        """
        return [task for task in self.tasks.values() if task.status == status]

    async def _task_worker(self) -> None:
        """Worker that processes tasks from the queue."""
        self.logger.info("Task worker started")
        
        while True:
            try:
                # Get a task from the queue
                task_id = await self.task_queue.get()
                
                # Process the task
                await self._process_task(task_id)
                
                # Mark the task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info("Task worker cancelled")
                break
                
            except Exception as e:
                self.logger.error(f"Error in task worker: {e}")
                
        self.logger.info("Task worker stopped")

    async def _process_task(self, task_id: str) -> None:
        """Process a task.
        
        Args:
            task_id: ID of the task to process.
        """
        if task_id not in self.tasks:
            self.logger.warning(f"Cannot process task {task_id}: not found")
            return
            
        task = self.tasks[task_id]
        
        # Check if the task is already completed or cancelled
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.logger.warning(
                f"Cannot process task {task_id}: already {task.status}"
            )
            return
            
        # Check if dependencies are met
        for dep_id in task.config.dependencies:
            if dep_id not in self.tasks:
                self.logger.warning(
                    f"Cannot process task {task_id}: dependency {dep_id} not found"
                )
                task.status = TaskStatus.FAILED
                task.error = f"Dependency {dep_id} not found"
                self._update_task_metrics()
                return
                
            dep_task = self.tasks[dep_id]
            
            if dep_task.status != TaskStatus.COMPLETED:
                self.logger.warning(
                    f"Cannot process task {task_id}: dependency {dep_id} not completed"
                )
                # Re-queue the task for later
                await self.task_queue.put(task_id)
                return
                
        # Acquire semaphore to limit concurrent tasks
        async with self.task_semaphore:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = asyncio.get_event_loop().time()
            self._update_task_metrics()
            
            # Get the agent
            agent_id = task.config.agent_id
            if not agent_id:
                self.logger.warning(f"Cannot process task {task_id}: no agent specified")
                task.status = TaskStatus.FAILED
                task.error = "No agent specified"
                self._update_task_metrics()
                return
                
            agent = self.get_agent(agent_id)
            if not agent:
                self.logger.warning(
                    f"Cannot process task {task_id}: agent {agent_id} not found"
                )
                task.status = TaskStatus.FAILED
                task.error = f"Agent {agent_id} not found"
                self._update_task_metrics()
                return
                
            # Create agent context
            context = AgentContext(
                agent_id=agent_id,
                task_id=task_id,
                inputs=task.config.inputs,
                metadata=task.config.metadata,
            )
            
            # Execute the task
            try:
                # Notify agent of task start
                await agent.on_task_start(context)
                
                # Execute the task with timeout
                timeout = task.config.timeout or self.orchestrator_config.default_timeout
                
                with self.orchestrator_metrics["task_execution_time"].labels(
                    agent_id=agent_id,
                    status="success"
                ).time():
                    result = await asyncio.wait_for(agent.execute(context), timeout)
                    
                # Update task status
                task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                task.result = result
                task.error = result.error
                task.completed_at = asyncio.get_event_loop().time()
                
                # Notify agent of task completion
                await agent.on_task_complete(result)
                
                # Check for dependent tasks
                await self._check_dependent_tasks(task_id)
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {task_id} timed out after {timeout} seconds")
                
                # Update task status
                task.status = TaskStatus.FAILED
                task.error = f"Task timed out after {timeout} seconds"
                task.completed_at = asyncio.get_event_loop().time()
                
                # Update metrics
                self.orchestrator_metrics["task_execution_time"].labels(
                    agent_id=agent_id,
                    status="timeout"
                ).observe(timeout)
                
                # Retry if configured
                await self._handle_task_retry(task)
                
            except Exception as e:
                self.logger.error(f"Error executing task {task_id}: {e}")
                
                # Update task status
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = asyncio.get_event_loop().time()
                
                # Update metrics
                self.orchestrator_metrics["task_execution_time"].labels(
                    agent_id=agent_id,
                    status="error"
                ).observe(
                    task.completed_at - (task.started_at or task.completed_at)
                )
                
                # Notify agent of error
                await agent.on_error(context, e)
                
                # Retry if configured
                await self._handle_task_retry(task)
                
            finally:
                # Update metrics
                self._update_task_metrics()

    async def _handle_task_retry(self, task: Task) -> None:
        """Handle task retry.
        
        Args:
            task: Task to retry.
        """
        # Check if retry is enabled
        if not self.orchestrator_config.retry_failed_tasks:
            return
            
        # Check if max retries reached
        max_retries = min(
            task.config.retry_count or 0,
            self.orchestrator_config.max_retries
        )
        
        if task.retries >= max_retries:
            self.logger.warning(
                f"Task {task.config.id} failed after {task.retries} retries"
            )
            return
            
        # Increment retry count
        task.retries += 1
        
        # Calculate retry delay
        retry_delay = task.config.retry_delay or self.orchestrator_config.retry_delay
        
        self.logger.info(
            f"Retrying task {task.config.id} in {retry_delay} seconds "
            f"(retry {task.retries}/{max_retries})"
        )
        
        # Reset task status
        task.status = TaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.error = None
        
        # Update metrics
        self._update_task_metrics()
        
        # Schedule retry
        await asyncio.sleep(retry_delay)
        await self.task_queue.put(task.config.id)

    async def _check_dependent_tasks(self, task_id: str) -> None:
        """Check if any tasks depend on the completed task.
        
        Args:
            task_id: ID of the completed task.
        """
        # Find tasks that depend on this task
        dependent_tasks = [
            task for task in self.tasks.values()
            if task_id in task.config.dependencies
            and task.status == TaskStatus.PENDING
        ]
        
        for task in dependent_tasks:
            # Check if all dependencies are met
            all_deps_met = True
            
            for dep_id in task.config.dependencies:
                if dep_id not in self.tasks:
                    all_deps_met = False
                    break
                    
                dep_task = self.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    all_deps_met = False
                    break
                    
            # If all dependencies are met, add to queue
            if all_deps_met:
                self.logger.info(
                    f"All dependencies met for task {task.config.id}, adding to queue"
                )
                await self.task_queue.put(task.config.id)
