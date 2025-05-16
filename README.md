# AMAOS - Advanced Multi-Agent Orchestration System

AMAOS is a modular, type-safe framework for building and orchestrating AI agent systems with robust fallback mechanisms, plugin support, and comprehensive monitoring.

## Core Features

- **Plugin System**: Extensible plugin architecture with lifecycle management
- **Agent Framework**: Modular agent system with BaseAgent implementation
- **LLM Manager**: Smart model selection with automatic fallback capabilities
- **Memory System**: Flexible memory abstraction with Redis support
- **Task Orchestrator**: Sequential and dependency-based task routing
- **Type Safety**: Fully type-annotated with mypy strict mode support
- **Metrics**: Prometheus-ready metrics collection for observability

## Getting Started

### Installation

```bash
# Install the core package
pip install -e .

# Install with LLM support
pip install -e ".[llm]"

# Install with memory support
pip install -e ".[memory]"

# Install development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
import asyncio
from amaos.agents.example_agent import ExampleAgent, ExampleAgentConfig
from amaos.core.orchestrator import Orchestrator, TaskConfig

async def main():
    # Initialize orchestrator
    orchestrator = Orchestrator()
    await orchestrator.initialize()
    
    # Create and register an agent
    agent_config = ExampleAgentConfig(
        name="example_agent",
        prompt_template="You are a helpful assistant. Answer: {question}"
    )
    agent = ExampleAgent(agent_config)
    await orchestrator.register_agent(agent)
    
    # Create a task
    task_config = TaskConfig(
        name="answer_question",
        agent_id=agent.id,
        inputs={"question": "What is AMAOS?"}
    )
    
    # Submit the task
    task_id = await orchestrator.submit_task(task_config)
    
    # Wait for task completion
    while True:
        task = orchestrator.get_task(task_id)
        if task.status in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)
    
    # Print the result
    if task.result:
        print(f"Answer: {task.result.outputs.get('answer', '')}")
    else:
        print(f"Task failed: {task.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

AMAOS follows a modular architecture with the following key components:

- **Core**: Base system components including Node, Plugin interfaces, and Orchestrator
- **Agents**: Agent implementations extending the BaseAgent class
- **Memory**: Memory systems for storing and retrieving data
- **Plugins**: Extensible plugin system for adding functionality
- **Utils**: Utility functions and helpers

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_plugin_manager.py

# Run with coverage
pytest --cov=amaos
```

### Type Checking

```bash
# Run mypy type checking
mypy amaos
```

## License

MIT License

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [Developer Guide](docs/DEVELOPER_GUIDE.md) - For contributors
- [User Guide](docs/USER_GUIDE.md) - For end users
- [Build Guide](docs/BUILD_GUIDE.md) - Build and deployment instructions
- [Contributing Guide](docs/CONTRIBUTING.md) - Contribution guidelines

## Directory Structure

```
amaos/           # Core package
├── core/        # Core system components
├── agents/      # Agent implementations
├── fallback/    # Fallback mechanisms
├── tools/       # Tool integrations
├── memory/      # Memory management
├── judge/       # Evaluation system
├── config/      # Configuration
├── deploy/      # Deployment scripts
├── utils/       # Utilities
└── dashboard/   # Monitoring dashboard

deploy/          # Deployment configuration
├── docker/      # Docker configuration
├── kubernetes/  # Kubernetes manifests
└── grafana/     # Monitoring dashboards

docs/            # Documentation
├── runbooks/    # Operational runbooks
└── api/         # API documentation

tests/           # Test suite
└── benchmarks/  # Performance tests
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CrewAI
- LangGraph
- AutoGen
- AGiXT
- SuperAGI
