# AMAOS - AI Modular Asynchronous Orchestration System

AMAOS is a modular, type-safe framework for building and orchestrating AI agent systems with robust fallback mechanisms, plugin support, and comprehensive monitoring.

## Core Features

- **Plugin System**: Extensible plugin architecture with lifecycle management
- **Agent Framework**: Modular agent system with multiple agent types including ReAct pattern
- **LLM Manager**: Smart model selection with automatic fallback capabilities
- **Memory System**: Flexible memory abstraction with Redis support and semantic memory
- **Task Orchestration**: Dynamic task routing with adaptive learning capabilities
- **Type Safety**: Fully type-annotated with mypy strict mode support
- **Metrics & Observability**: Comprehensive tracing and monitoring with streaming dashboards

## New Features

### Semantic Memory System
- Vector-based memory with FAISS integration for semantic similarity search
- Automatic fallback to keyword search when embedding fails
- Persisted vector indices with incremental updates
- Memory node extension with `semantic_store` and `semantic_get` operations
- Configurable embedding models and index types

### Reactive Agent Pattern
- First-class ReAct pattern implementation with think-act-observe loop
- Extensible tool registry with calculator and search mock tools
- Step-by-step reasoning with observation feedback loop
- Configurable for maximum iterations and specialized prompting
- Task history with detailed step tracing

### Adaptive Control Node
- Performance-based task routing with multiple strategies
- Learning from historical performance, latency, and feedback
- Persisted metrics for continuous improvement
- Multiple fallback strategies with configurable retry logic
- Comprehensive statistics for node selection optimization

### ReflectorStream Interface
- Real-time WebSocket streaming of system events
- FastAPI endpoints for log filtering and retrieval
- Trace ID propagation across node boundaries
- Filterable stream by node type, trace ID, or task type
- Support for historical log retrieval and live updates

### Context-Aware Logging
- Context propagation across async boundaries
- Trace ID generation and tracking
- Structured context data in log messages
- Enhanced debugging capabilities

## Getting Started

### Installation

```bash
# Install the core package
pip install -e .

# Install with LLM support
pip install -e ".[llm]"

# Install with memory support
pip install -e ".[memory]"

# Install with semantic memory support
pip install -e ".[semantic]"

# Install development dependencies
pip install -e ".[dev]"
```

### Using Semantic Memory

```python
import asyncio
from amaos.nodes.memory_node import MemoryNode, MemoryNodeConfig
from amaos.core.node_protocol import NodeTask, NodeResult

async def main():
    # Initialize memory node with semantic memory enabled
    config = MemoryNodeConfig(
        node_id="memory_node",
        enable_semantic_memory=True
    )
    memory_node = MemoryNode(config)
    await memory_node.initialize()
    
    # Store text with semantic embedding
    store_task = NodeTask(
        task_type="memory",
        payload={
            "action": "semantic_store",
            "key": "doc1",
            "text": "AMAOS is a modular AI orchestration system",
            "metadata": {"category": "documentation"}
        }
    )
    result = await memory_node.handle(store_task)
    print(f"Store result: {result.success}")
    
    # Search for semantically similar content
    search_task = NodeTask(
        task_type="memory",
        payload={
            "action": "semantic_get",
            "query": "What is an AI framework?",
            "limit": 3,
            "metadata_filter": {"category": "documentation"}
        }
    )
    result = await memory_node.handle(search_task)
    print(f"Search results: {result.result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Creating a Reactive Agent

```python
import asyncio
from amaos.agents.reactive_agent import ReactiveAgent, ReactiveAgentConfig
from amaos.agents.base_agent import AgentContext

async def main():
    # Configure reactive agent
    config = ReactiveAgentConfig(
        name="reasoning_agent",
        max_iterations=5,
        llm_node_id="llm"
    )
    agent = ReactiveAgent(config)
    await agent.initialize()
    
    # Register a custom tool
    async def weather_tool(location: str):
        return f"The weather in {location} is sunny and 75°F"
    
    agent.register_tool(
        name="weather",
        description="Get weather information for a location",
        parameters={"location": "City or location name"},
        function=weather_tool
    )
    
    # Create context and execute task
    context = AgentContext(
        agent_id=agent.id,
        task_id="task123",
        inputs={
            "task": "Plan a picnic for tomorrow. I need to know the weather and decide what to bring."
        },
        metadata={"trace_id": "trace123"}
    )
    
    # Execute the agent
    result = await agent.execute(context)
    
    # Print the result with step trace
    print(f"Answer: {result.outputs['answer']}")
    print("\nReasoning Steps:")
    for i, step in enumerate(result.outputs["steps"]):
        print(f"Step {i+1}:")
        print(f"  Thought: {step['thought']}")
        print(f"  Action: {step['action']}")
        print(f"  Action Input: {step['action_input']}")
        print(f"  Observation: {step['observation']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Adaptive Control Node

```python
import asyncio
from amaos.nodes.adaptive_control_node import AdaptiveControlNode, AdaptiveNodeConfig, RoutingStrategy
from amaos.nodes.llm_node import LLMNode, LLMNodeConfig
from amaos.core.node_protocol import NodeTask, NodeResult

async def main():
    # Create multiple LLM nodes
    llm1 = LLMNode(LLMNodeConfig(name="primary_llm", providers=["openai"]))
    llm2 = LLMNode(LLMNodeConfig(name="backup_llm", providers=["anthropic"]))
    
    # Create adaptive control node
    config = AdaptiveNodeConfig(
        node_id="adaptive_router",
        default_strategy=RoutingStrategy.ADAPTIVE,
        latency_weight=0.4,
        success_weight=0.4,
        feedback_weight=0.2
    )
    
    # Register nodes by task type
    router = AdaptiveControlNode(
        node_registry={"llm": llm1, "backup_llm": llm2},
        config=config
    )
    await router.initialize()
    
    # Create a task
    task = NodeTask(
        task_type="llm",
        payload={"prompt": "Explain AMAOS in one paragraph"},
        metadata={
            "routing_strategy": RoutingStrategy.PERFORMANCE,
            "trace_id": "trace_abc123"
        }
    )
    
    # Process task with adaptive routing
    result = await router.handle(task)
    print(f"Result from: {result.source}")
    print(f"Content: {result.result}")
    
    # Provide feedback to improve routing
    await router.provide_feedback(task.metadata["trace_id"], 0.9)
    
    # Check statistics
    stats = router.get_stats()
    print(f"Routing stats: {stats['adaptive']['metrics']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the ReflectorStream Dashboard

```python
import asyncio
from amaos.nodes.reflector_node import ReflectorNode
from amaos.stream.reflector_stream import ReflectorStream, StreamConfig
from amaos.nodes.llm_node import LLMNode

async def main():
    # Create a reflector wrapping an LLM node
    llm = LLMNode()
    reflector = ReflectorNode(llm)
    await reflector.initialize()
    
    # Configure stream
    config = StreamConfig(
        host="127.0.0.1",
        port=8000,
        log_buffer_size=1000,
        enable_history=True
    )
    
    # Create and start stream
    stream = ReflectorStream(config=config, reflector_node=reflector)
    
    # This will start a server in the background
    server_task = asyncio.create_task(stream.start())
    
    print(f"ReflectorStream started at http://{config.host}:{config.port}")
    print("Connect with a WebSocket client to ws://127.0.0.1:8000/ws")
    
    # Keep the server running
    await server_task

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

AMAOS follows a modular architecture with the following key components:

- **Core**: Base system components including Node, Plugin interfaces, and Orchestrator
- **Agents**: Agent implementations including ReactiveAgent with ReAct pattern
- **Nodes**: Specialized processing units (LLM, Memory, Tool, Reflector, etc.)
- **Memory**: Memory systems including vector-based semantic memory
- **Stream**: Real-time event streaming and visualization
- **Plugins**: Extensible plugin system for adding functionality
- **Utils**: Utility functions including context-aware logging

## Trace ID Propagation

All operations in AMAOS now support trace ID propagation for comprehensive request tracing. This allows tracking a request across different nodes, creating a complete view of system behavior. Trace IDs are automatically generated if not provided and included in all log messages and event streams.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_memory_node.py

# Run with coverage
pytest --cov=amaos
```

### Type Checking

```bash
# Run mypy type checking
mypy amaos
```

## Future Directions

See the [AMAOS_ROADMAP.md](AMAOS_ROADMAP.md) file for planned enhancements and features.

Potential focus areas for the next phase include:
- Knowledge graph integration for structured knowledge representation
- Multi-agent collaboration frameworks
- Self-optimizing agent patterns
- Dynamic workflow generation

## License

MIT License
