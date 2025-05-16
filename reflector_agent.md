# ReflectorAgent Usage Guide

The ReflectorAgent is a meta-agent for AMAOS that observes agent behavior, memory entries, and task outcomes, providing insights and suggestions for improvement.

## Overview

ReflectorAgent acts as a system observer and analyzer, helping you understand and improve your agent ecosystem by:

1. Collecting observations about agent behaviors
2. Analyzing system performance
3. Detecting anomalies
4. Suggesting improvements
5. Critiquing its own analyses through a debate mechanism

## Local Model Support

ReflectorAgent uses the LLMManager to generate insights. Recent updates have enhanced the LLMManager to support:

1. Function-based local models (like llama3-local, mistral-local)
2. A mock model for testing and demos

### Using the Mock Model for Testing

If you don't have local LLM models available or are experiencing issues with them, the system now includes a `mock-model` configured in `config/model_priority.yaml` with the highest priority. This ensures the ReflectorAgent can always run, even in environments without GPU access or installed models.

The mock model returns simulated responses suitable for testing and development.

## Running the ReflectorAgent

You can run the ReflectorAgent directly as a module:

```bash
python -m amaos.agents.reflector_agent
```

This will run a demonstration of the agent's capabilities.

Alternatively, you can integrate the ReflectorAgent into your AMAOS system:

```python
from amaos.agents.reflector_agent import ReflectorAgent

# Create a ReflectorAgent
reflector = ReflectorAgent(
    id="system-reflector",
    name="System Reflector",
    description="Analyzes system behavior and suggests improvements",
    schedule_interval=3600  # Run analysis hourly
)

# Initialize the agent
await reflector.initialize()

# Record observations about your system
await reflector.record_observation(
    "agent_behaviors",
    {
        "agent_id": "agent-1",
        "action": "task_completed",
        "task_id": "task-123",
        "execution_time": 1.5
    }
)

# Get insights from the reflector
analysis = await reflector.run_scheduled_analysis()
```

## Testing with Mock Model

For testing environments, use the provided test script:

```bash
python examples/reflector_agent_test.py
```

This script demonstrates the ReflectorAgent with the mock model.

## Next Steps for Development

1. **Add custom models**: Extend the model_loader to include your own local models
2. **Create specialized analyzers**: Develop domain-specific analysis methods for the ReflectorAgent
3. **Integrate with monitoring**: Connect the ReflectorAgent with your monitoring system
4. **Implement auto-remediation**: Allow the ReflectorAgent to automatically fix simple issues
