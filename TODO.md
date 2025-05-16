# 🧠 GTP #0 — AMAOS MASTER TODO

This document tracks the full roadmap of AMAOS: from architecture convergence to advanced orchestration, and finally intelligent self-improving agents.

---

## ✅ PHASE 1: Core Foundation (COMPLETED)

* [x] Unified Node Protocol (NodeTask, NodeResult, Node ABC)
* [x] Refactored LLMNode
* [x] Refactored ToolNode (plugin execution)
* [x] Refactored MemoryNode with fallback + metrics
* [x] NodeProtocol test suite (test\_node\_convergence)
* [x] ToolNode test suite
* [x] MemoryNode test suite (MockRedis, TTL, fallback)
* [x] Type-safety with mypy strict
* [x] Pydantic v2 compatibility
* [x] Clear directory layout: nodes/, plugins/, utils/, models/, tests/

### 🔮 PHASE 1: TEST PHASE

* [x] All core tests pass with full coverage (llm, tool, memory)
* [x] mypy --strict success
* [x] Pytest markers for timeouts in all async tests

---

## 🔄 PHASE 2: Orchestration + Observability (IN PROGRESS)

### 🎛 GTP #13 — ControlNode (Task Router)

* [x] Accepts generic NodeTask
* [x] Routes to appropriate node by `task_type`
* [x] Handles retry logic (per node)
* [x] Returns NodeResult
* [x] Optional chaining / multi-step pipeline support
* [~] Optional routing metadata (e.g. max_tokens, role constraints) — partial support (metadata injected, can be extended)
* [x] Test suite for routing / fallback / retries / unknown types

**Summary:**
- Universal ControlNode and ChainedControlNode implemented.
- Supports dynamic routing, retries, and multi-step pipelines.
- Metadata-driven routing supported (extendable for more constraints).
- Full test coverage for all routing, chaining, and fallback logic.

> **Technologies**: Pure Python async routing, dictionary-based node registry, extensible policy config (e.g., retry, constraints)

### 🪞 GTP #14 — ReflectorNode (Passive Observer)

* [x] Wraps any Node, logs incoming NodeTask and outgoing NodeResult
* [x] Supports tracing, audit logs, side-channel memory (via callback)
* [x] Optional mutation/enrichment of tasks/results
* [x] Integration with MemoryNode for memory replay
* [x] Test suite: logs, integrity, time-stamped memory replay

**Summary:**
- ReflectorNode wraps any Node, logs all tasks/results with timestamps.
- Supports side-channel journaling (memory callback for replay/audit).
- Supports mutation/enrichment hooks for NodeTask and NodeResult (pre/post handle).
- Integrates with MemoryNode for persistent replay/audit.
- Fully tested for logging, memory callback, memory replay, and integrity.

> **Technologies**: Logging, task cloning, asyncio wrapper injection, optional Redis/memory tap for journaling

---

## 🧠 GTP #15 — CrewManager / AgentManager (COMPLETE)

* [x] Maintain a roster of specialized agents (LLMNode, ToolNode, etc.)
* [x] Use ControlNode to route subtasks
* [x] Support role-based, goal-oriented planning (metadata, assignment)
* [x] Test suite for registration, routing, and stats
* [x] Advanced agent planning, fallback, and dynamic discovery

**Summary:**
- CrewManager maintains a dynamic roster of agents with roles/goals.
- Supports registration, deregistration, role/goal assignment.
- Routes all tasks via ControlNode or directly to agents.
- Fully tested for registration, routing, and stats.

---

## 🦾 GTP #16 — FallbackNode (COMPLETE)

* [x] Structured retries with exponential backoff
* [x] Alternate node fallback (provider shift)
* [x] Graceful degradation and error reporting
* [x] Orchestrates agentic tasks with stats and logging
* [x] Full test suite for all fallback scenarios

**Summary:**
- FallbackNode wraps primary and alternate nodes, retries on failure, and degrades gracefully.
- Tracks attempts/failures, logs all fallback events, and exposes stats for orchestration.
- Fully tested for all fallback scenarios (primary, alternate, all-fail).

---

## 🧬 NEXT UP

* [x] Semantic memory search with FAISS (GTP #17)
* [x] AdaptiveControlNode for dynamic routing
* [x] ReactiveAgent with ReAct pattern
* [x] ReflectorStream for real-time event monitoring
* [x] Context-aware logging
* [ ] Comprehensive test coverage for new components:
  * [ ] test_semantic_memory.py: Verify embedding, indexing, and search functionality
  * [ ] test_adaptive_control_node.py: Test routing strategies, metrics, and feedback
  * [ ] test_reactive_agent.py: Test tools execution, reflection, and multi-step reasoning
  * [ ] test_reflector_stream.py: Verify WebSocket streaming and event filtering
  * [ ] test_context_logger.py: Test context propagation across async boundaries

### 🔮 PHASE 2: TEST PHASE

* [ ] test\_control\_node.py: Validate dynamic routing, retry fallbacks
* [ ] test\_reflector\_node.py: Confirm NodeTask/NodeResult logging
* [ ] Cross-node coordination test (simulate pipeline)

---

## ⚡ PHASE 3: Intelligent Agent Layer (PLANNED)

### 🧠 GTP #15 — CrewManager / AgentManager

* [ ] Maintain a roster of specialized agents (LLMNode, ToolNode, etc.)
* [ ] Use ControlNode to route subtasks
* [ ] Support role-based, goal-oriented planning
* [ ] Memory conditioning based on NodeResults
* [ ] Integrate Reflector for memory-backed thinking

> **Technologies**: Agent graph abstraction, dynamic role resolution, memory-conditioned dispatch

### ↺ GTP #16 — Self-Healing / Retry Graph

* [ ] Define FallbackNode with retry policies
* [ ] Support per-model and per-capability fallback graphs
* [ ] Visualization / inspection tools

> **Technologies**: DAG fallback structure, retry thresholds, graph traversal heuristics

### 🗂️ GTP #17 — Memory Enhancement (PARTIAL COMPLETION)

* [x] Add semantic search over long-term memory with FAISS
* [x] Fallback to keyword search when embedding fails
* [x] SemanticMemory class with configurable index types (FLAT, IVF, HNSW)
* [ ] Fix mypy type errors in semantic_memory.py
* [ ] TTL decay and reinforcement learning (via rewards)
* [ ] Episode chunking and compression (to Redis or VectorDB)

> **Technologies**: FAISS vector indexing, SentenceTransformers for embeddings, Redis persistence, TTL-driven garbage collection

### 🔮 PHASE 3: TEST PHASE

* [ ] test\_agent\_manager.py: verify task delegation
* [ ] test\_fallback\_graph.py: simulate retry scenarios
* [ ] test\_memory\_enhancement.py: semantic match and TTL decay

---

## 🛡️ PHASE 3.5: Type Safety and Code Quality

### 🔍 Type Safety Improvements

* [ ] Fix all mypy errors in semantic_memory.py:
  * [ ] Add proper type stubs for faiss and sentence_transformers
  * [ ] Clean up unreachable code statements
  * [ ] Fix attribute access issues with SemanticMemoryConfig
  * [ ] Add proper type annotations for all variables
* [ ] Fix Any return types in ReactiveAgent
* [ ] Ensure safe parameter handling in AdaptiveControlNode
* [ ] Add proper error handling for async context propagation
* [ ] Create explicit TypeVars for generic container types

### 📊 Code Quality

* [ ] Add docstring test examples for key public methods
* [ ] Standardize error handling across all modules
* [ ] Enforce consistent naming conventions
* [ ] Refactor duplicate code in metrics collection
* [ ] Add parameter validation for all critical functions

> **Technologies**: mypy, stub files, Python typings module, doctest, consistent error handling patterns

## 🥚 PHASE 4: Stability, Observability, and Extensibility

### 🧪 Tests & Mocks

* [ ] test\_reflector\_node.py
* [ ] test\_control\_node.py (advanced pipelines)
* [ ] test\_self\_healing\_node.py

### 📊 Monitoring

* [ ] Integrate Prometheus metrics for node executions
* [ ] Runtime logging per node (logger per class)
* [ ] Optional tracing via OpenTelemetry or lightweight tracer

### 🧹 Plugins

* [ ] Add browser tool (headless Playwright integration)
* [ ] Add filesystem tool (read/write/delete/search)
* [ ] Add LLM Completion plugin (local + cloud options)

> **Technologies**: Playwright, Python's pathlib/fs, plugin loading via importlib, completion adapters for Ollama/OpenAI

### 🔮 PHASE 4: TEST PHASE

* [ ] Full integration test across Control, LLM, Tool, Memory
* [ ] Load test plugin execution
* [ ] Monitor performance metrics and retry behavior

---

## 🧠 BONUS IDEAS

* [ ] NyxShell — Command-line interface to invoke ControlNode
* [ ] Web UI using Streamlit or Next.js
* [ ] TaskQueue (queue + deduplication + monitoring)
* [ ] Local fine-tuning hook for LLM feedback improvement

---

## 🔀 How to Use This File

Every time we complete a GTP or milestone, update this file. This is the **single source of truth** for AMAOS progress.

If you want me to automatically parse or transform it to Obsidian, markdown planner, or issue tracker — say the word.

Stay god-tier.

— Nyx
