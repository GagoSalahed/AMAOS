# AMAOS Memory Tracing & Snapshot Utilities

This document explains how to use the `MemoryTracer` plugin and `MemorySnapshot` utility for monitoring, diagnosing, and backing up AMAOS agent memory (Redis/fakeredis).

## Overview
- **MemoryTracer:** Tracks Redis key usage, detects anomalies, and can be attached to agent lifecycles (e.g., ReflectorAgent).
- **MemorySnapshot:** Exports/imports Redis state to/from JSON for backup, migration, or debugging.

---

## 1. MemoryTracer Plugin

### Features
- Tracks all Redis keys and usage frequency over time.
- Detects:
  - Rapid memory/key growth
  - Repeated (persistent) keys
  - Unused (single-access) keys
- Generates time-series logs (`memory_tracer_timeseries.jsonl`)
- Can run in background or be called manually
- Optional: Cleans keys by pattern or TTL
- Integrates with agent lifecycle (e.g., ReflectorAgent)

### Usage Example
```python
from amaos.utils.memory_tracer_plugin import MemoryTracer

tracer = MemoryTracer(trace_interval=10, history_size=100)
tracer.start_tracing()  # Background tracing
# ... run your agents ...
tracer.stop_tracing()

# Manual trace & report
stats = tracer.trace_memory()
report = tracer.report()
print(report)

# Clean up agent keys with TTL=0
tracer.clean_keys(pattern="agent:*", ttl_zero_only=True)
```

#### Integration with ReflectorAgent
Pass your agent to the tracer:
```python
tracer = MemoryTracer(agent=my_reflector_agent)
```
If the agent implements `on_memory_trace(stats)`, it will be notified each cycle.

---

## 2. MemorySnapshot Utility

### Features
- Exports all Redis keys/values to JSON
- Restores from JSON snapshot
- Works with Redis and fakeredis

### Usage Example
```python
from amaos.utils.memory_snapshot import MemorySnapshot

snap = MemorySnapshot()
snap.snapshot_memory("backup.json")  # Export
# ... clear or modify Redis ...
snap.restore_snapshot("backup.json")  # Import
```

---

## 3. Testing
Run the provided tests (pytest):
```sh
pytest tests/test_memory_tracer.py
pytest tests/test_memory_snapshot.py
```

---

## 4. Recommendations
- Use MemoryTracer in development and troubleshooting to catch memory leaks or key misuse.
- Automate snapshots for backup or migration.
- Clean up unused or orphaned keys regularly in resource-constrained environments.
- Review time-series logs for historical trends and anomaly detection.

---

## 5. Manual Run Steps

### MemoryTracer
```python
from amaos.utils.memory_tracer_plugin import MemoryTracer
tracer = MemoryTracer(trace_interval=5)
tracer.start_tracing()
# ... let it run ...
tracer.stop_tracing()
print(tracer.report())
```

### MemorySnapshot
```python
from amaos.utils.memory_snapshot import MemorySnapshot
snap = MemorySnapshot()
snap.snapshot_memory("current_snapshot.json")
# ... modify Redis ...
snap.restore_snapshot("current_snapshot.json")
```

---

## 6. Notes
- All utilities are offline-capable and require no external services.
- Logs and snapshots are plain JSON for easy inspection.
- For async/advanced Redis, extend `get_redis_client()` as needed.
