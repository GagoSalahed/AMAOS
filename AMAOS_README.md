# 🧠 AMAOS - AI Modular Asynchronous Orchestration System

AMAOS is a robust, modular framework for building and orchestrating AI systems with asynchronous processing, smart fallback mechanisms, and comprehensive monitoring capabilities.

## 🌟 Overview

AMAOS provides a flexible architecture for creating AI applications with a focus on reliability, extensibility, and observability. The system is built around a node-based design where different components (LLMs, tools, memory, etc.) can be orchestrated to work together seamlessly.

Key design philosophies:
- **Asynchronous by default**: Built for non-blocking operations
- **Graceful degradation**: Multi-level fallback mechanisms
- **Observability**: Comprehensive metrics and logging
- **Modularity**: Plug-and-play components with standardized interfaces
- **Type safety**: Fully type-annotated with Python 3.11+ compatibility

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         AMAOS SYSTEM                            │
├─────────────┬─────────────┬──────────────┬────────────┬─────────┤
│  CORE       │  NODES      │  AGENTS      │  MEMORY    │ PLUGINS │
├─────────────┼─────────────┼──────────────┼────────────┼─────────┤
│ Node        │ LLMNode     │ BaseAgent    │ Memory     │ Plugin  │
│ Protocol    │ ToolNode    │ ExampleAgent │ Interface  │ Manager │
│ Orchestrator│ MemoryNode  │ ...          │ Redis      │ ...     │
│ ...         │ ...         │              │ ...        │         │
└─────────────┴─────────────┴──────────────┴────────────┴─────────┘
        │             │            │             │          │
        └─────────────┴────────────┴─────────────┴──────────┘
                                │
                    ┌───────────▼───────────┐
                    │   EVENT-DRIVEN BUS    │
                    └───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  METRICS & MONITORING │
                    └───────────────────────┘
```

## 🧩 Module Responsibilities

### Core (`amaos/core/`)
- **node.py**: Base Node implementation with event system and lifecycle management
- **node_protocol.py**: Core interfaces for node communication
- **orchestrator.py**: Task scheduling and routing
- **plugin_interfaces.py**: Interfaces for extending the system
- **plugin_manager.py**: Dynamic loading and management of plugins
- **llm_manager.py**: Smart LLM provider selection and management

### Nodes (`amaos/nodes/`)
- **control_node.py**: Routing and pipeline management
- **fallback_node.py**: Graceful degradation with fallback strategies
- **guardrail_node.py**: Input/output validation and safety boundaries
- **llm_node.py**: LLM integration with multiple provider support
- **memory_node.py**: State persistence and retrieval
- **reflector_node.py**: Logging, tracing, and replay capabilities
- **tool_node.py**: External tool/API integration
- **user_input_node.py**: User interaction handling

### Agents (`amaos/agents/`)
- **base_agent.py**: Foundation for all agent types
- **example_agent.py**: Reference implementation

### Memory (`amaos/memory/`)
- **interface.py**: Common interface for memory implementations
- **memory.py**: Redis-based implementation with fallbacks

### Utils (`amaos/utils/`)
- **errors.py**: Standardized error handling
- **logging.py**: Enhanced logging utilities

## 🔌 Plugin Support & Extension API

AMAOS provides a flexible plugin system that allows for extending functionality without modifying core code:

### Plugin Development

```python
from amaos.plugins.base_plugin import BasePlugin
from amaos.core.plugin_interfaces import PluginLifecycle

class CustomPlugin(BasePlugin, PluginLifecycle):
    """Custom plugin implementation."""
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        self.logger.info("Initializing custom plugin")
        
    async def start(self) -> None:
        """Start the plugin."""
        self.logger.info("Starting custom plugin")
        
    async def stop(self) -> None:
        """Stop the plugin."""
        self.logger.info("Stopping custom plugin")
```

### Node Extension

Creating custom nodes by extending the base Node class:

```python
from amaos.core.node import Node, NodeConfig
from amaos.core.node_protocol import NodeTask, NodeResult

class CustomNode(Node):
    """Custom node implementation."""
    
    async def initialize(self) -> None:
        """Initialize the node."""
        await super().initialize()
        self.logger.info("Custom node initialized")
        
    async def handle(self, task: NodeTask) -> NodeResult:
        """Handle a task."""
        self.logger.info(f"Handling task: {task.task_type}")
        # Custom task handling logic
        return NodeResult(success=True, result={"message": "Task handled"})
        
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {"custom_stat": 123}
```

## 🚀 How to Run / Test / Extend

### Installation

```bash
# Basic installation
pip install -e .

# With LLM support
pip install -e ".[llm]"

# With memory support
pip install -e ".[memory]"

# Development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
import asyncio
from amaos.core.node import Node, NodeConfig
from amaos.nodes.llm_node import LLMNode, LLMNodeConfig
from amaos.nodes.fallback_node import FallbackNode
from amaos.core.node_protocol import NodeTask, NodeResult

async def main():
    # Create an LLM node with OpenAI
    llm_config = LLMNodeConfig(
        name="primary_llm",
        providers=["openai"],
        api_keys={"openai": "your-api-key"},
    )
    primary_llm = LLMNode(llm_config)
    
    # Create a fallback LLM node
    fallback_config = LLMNodeConfig(
        name="fallback_llm",
        providers=["anthropic"],
        api_keys={"anthropic": "your-api-key"},
    )
    fallback_llm = LLMNode(fallback_config)
    
    # Create a fallback node
    fallback_node = FallbackNode(
        primary=primary_llm,
        fallbacks=[fallback_llm],
    )
    
    # Initialize the nodes
    await primary_llm.initialize()
    await fallback_llm.initialize()
    await fallback_node.initialize()
    
    # Create a task
    task = NodeTask(
        task_type="llm",
        payload={"prompt": "Explain AMAOS in one paragraph"},
        metadata={"max_tokens": 100}
    )
    
    # Process the task with automatic fallback
    result = await fallback_node.handle(task)
    
    # Print the result
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Source: {result.source}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=amaos

# Run specific test file
pytest amaos/tests/test_llm_node.py
```

### Creating a Custom Control Flow

```python
from amaos.nodes.control_node import ChainedControlNode
from amaos.core.node_protocol import Node, NodeTask

async def setup_pipeline():
    # Create nodes
    llm_node = await create_llm_node()
    memory_node = await create_memory_node()
    tool_node = await create_tool_node()
    
    # Create a pipeline
    pipeline = ChainedControlNode({
        "llm": llm_node,
        "memory": memory_node,
        "tool": tool_node
    })
    
    await pipeline.initialize()
    return pipeline

async def process_user_input(pipeline, user_input):
    # Create task for processing
    task = NodeTask(
        task_type="llm",
        payload={"prompt": user_input},
        metadata={"memory_id": "user_session_123"}
    )
    
    # Process through pipeline
    result = await pipeline.handle(task)
    return result
```

## 🔮 Future Vision

AMAOS is evolving toward a more autonomous, self-optimizing system with these key directions:

1. **Adaptive Intelligence**: Nodes that learn from usage patterns and adapt behavior
2. **Distributed Processing**: Scale-out capability for high-throughput applications
3. **Knowledge Graph Integration**: Structured knowledge representation and reasoning
4. **Explanation Generation**: Auto-generated explanations for system decisions
5. **Multi-Agent Collaboration**: Frameworks for agent teams with specialized roles
6. **Real-time Streaming**: First-class support for streaming I/O across all components

The roadmap includes:
- Enhanced semantic memory with vector embeddings
- Dynamic pipeline generation based on task requirements
- Improved observability with real-time dashboards
- Self-tuning capabilities for optimal resource allocation

## 📊 System Stats & Monitoring

AMAOS includes comprehensive metrics collection across all components:

- **Node State**: Track lifecycle state of all system nodes
- **Event Processing**: Measure event throughput and latency
- **Task Execution**: Monitor task success rates and processing times
- **Memory Operations**: Track memory access patterns and hit rates
- **LLM Performance**: Provider-specific performance and error tracking

All metrics are Prometheus-compatible and can be visualized with Grafana dashboards.

## 🛠️ Development

### Directory Structure

```
amaos/           # Core package
├── core/        # Core system components
├── agents/      # Agent implementations
├── nodes/       # Node implementations
├── memory/      # Memory management
├── models/      # Data models
├── plugins/     # Plugin system
├── utils/       # Utilities
└── tests/       # Test suite

docs/            # Documentation
├── api/         # API documentation
└── examples/    # Example implementations
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to AMAOS.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
