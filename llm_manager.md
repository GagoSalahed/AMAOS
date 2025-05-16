# LLMManager: Smart Model Selection for AMAOS

The `LLMManager` is a powerful component in AMAOS that provides smart, cost-aware model selection and fallback handling for LLM tasks. It helps applications balance performance, cost, and reliability when working with language models.

## Key Features

- 🔄 **Automatic Model Fallback**: If one model fails or times out, automatically fall back to the next available model
- 💰 **Cost-Aware Model Selection**: Define cost parameters for models to optimize for cost-effectiveness
- 🌎 **Local & Cloud Support**: Seamlessly work with both local models (Ollama) and cloud providers (Claude, OpenAI, Gemini)
- 🔌 **Online/Offline Mode**: Work in offline mode with local models only
- 📊 **Usage Metrics**: Track usage, costs, and performance across different models
- 🛡️ **Error Handling**: Gracefully handle timeouts, API errors, and other failure modes
- 🔍 **Model Choice**: Allow forcing specific models when needed

## Installation

The LLMManager is part of AMAOS. You'll need to install the appropriate dependencies for the models you want to use:

```bash
# For all providers
pip install "amaos[llm]"

# Or install specific providers
pip install "amaos[ollama,openai,anthropic,google]"
```

## Basic Usage

```python
from amaos.core.llm_manager import LLMManager

# Initialize the manager
manager = LLMManager()

# Simple completion
result = await manager.complete("Explain what AMAOS is.")

# Print the result
print(f"Response from {result.model_name}:")
print(result.content)
print(f"Tokens: {result.tokens}")
print(f"Cost: ${result.cost:.6f}")
```

## Configuration

The LLMManager can be configured in several ways:

### 1. Environment Variables

Set API keys and other configuration in your environment:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
GOOGLE_API_KEY=...
QWEN_API_KEY=...
```

### 2. Configuration File

Create a `model_priority.yaml` file to define model priorities and settings:

```yaml
models:
  - name: llama3-local
    provider: ollama
    source: local
    priority: 10
    max_tokens: 2048
    temperature: 0.7
    timeout: 15.0

  - name: claude-3-haiku
    provider: anthropic
    source: cloud
    priority: 40
    max_tokens: 4096
    temperature: 0.7
    timeout: 20.0
    cost_per_1k_input: 0.25
    cost_per_1k_output: 0.75
    api_key_env: ANTHROPIC_API_KEY
```

## Advanced Usage

### Local-Only Mode

```python
# Initialize in local-only mode
manager = LLMManager(local_only=True)

# Now only local models will be used
result = await manager.complete("What are the benefits of local LLMs?")
```

### Forcing a Specific Model

```python
# Force the use of a specific model
result = await manager.complete(
    "Generate a creative story idea.",
    force_model="claude-3-sonnet"
)
```

### Custom System Prompts

```python
# Use a custom system prompt
result = await manager.complete(
    "Write a poem about AI.",
    system_prompt="You are a creative poetry assistant. Write in the style of Emily Dickinson."
)
```

### Adjusting Parameters

```python
# Customize temperature and max tokens
result = await manager.complete(
    "Explain quantum computing.",
    temperature=0.2,  # Lower temperature for more factual responses
    max_tokens=500    # Limit response length
)
```

## Using with Node Architecture

AMAOS provides a node implementation that integrates with the graph-based architecture:

```python
from amaos.core.nodes.llm_manager_node import LLMManagerNode
from amaos.core.graph_orchestrator import GraphOrchestrator

# Create orchestrator
orchestrator = GraphOrchestrator()
await orchestrator.initialize()

# Create LLM manager node
llm_node = LLMManagerNode(
    id="main_llm",
    name="Main LLM",
    config={
        "config_path": "config/model_priority.yaml",
        "local_only": False,
        "enable_fallback": True
    }
)

# Register node
await orchestrator.register_node(llm_node)

# Create a task graph using the node...
```

## Metrics and Monitoring

The LLMManager tracks usage statistics:

```python
# Get usage stats
stats = manager.get_stats()
print(f"Total calls: {stats['calls']}")
print(f"Successes: {stats['successes']}")
print(f"Failures: {stats['failures']}")
print(f"Timeouts: {stats['timeouts']}")

# Get model-specific stats
for model_name, model_stats in stats['by_model'].items():
    print(f"{model_name}: {model_stats['calls']} calls, {model_stats['avg_latency']:.2f}s avg latency")
```

## Model Support

The LLMManager currently supports the following models/providers:

### Local Models
- Ollama (llama3, mistral, qwen, etc.)

### Cloud Models
- Anthropic Claude (3, 3.5, etc.)
- OpenAI (GPT-3.5-Turbo, GPT-4o, etc.)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash)
- Qwen (via API)

## Architecture Diagram

```
┌──────────────────┐     ┌─────────────┐
│   Application    │────▶│ LLMManager  │
└──────────────────┘     └──────┬──────┘
                               │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Local Models    │ │Cloud Models  │ │ Fallback Models  │
│ (Ollama, etc.)   │ │(Claude, GPT) │ │ (Lower priority) │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

## Extending for New Models

To add support for new model providers, subclass the LLMManager and implement the provider-specific call method.
