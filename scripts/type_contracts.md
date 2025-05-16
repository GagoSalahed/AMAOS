# Type Contracts for AMAOS

> **Purpose**: This document establishes the strict typing rules and contracts that all AMAOS code must adhere to.
> These contracts are enforced by mypy and verified by contract tests.

## General Rules

### Core Principles
- All functions must have explicit type annotations for parameters and return values
- Use `Optional[T]` for any value that could be `None`
- Avoid `Any` unless absolutely necessary and document the reason
- All imports for typing purposes should be conditional:
  ```python
  from typing import Dict, List, Optional, Union, TYPE_CHECKING
  if TYPE_CHECKING:
      from .specific_module import SpecificType
  ```

### Type Annotations
- For class methods, always annotate `self` and `cls` properly
- For coroutines, use `async def func() -> Awaitable[ResultType]`
- Use Protocol classes for duck typing instead of TypeVars where possible
- Always use the most specific type possible (e.g., `Dict[str, int]` instead of `Dict`)

## Module-Specific Contracts

### NodeResult Contract
- `NodeResult` must include a `metadata` field with the following structure:
  ```python
  metadata: Dict[str, Any]  # Consider using TypedDict for stricter typing
  ```
- All NodeResult instances must be immutable after creation
- Accessing a non-existent metadata key should return None, not raise KeyError

### ReflectorNode Contract
- ReflectorNode must implement the following methods:
  - `reflect() -> NodeResult`
  - `validate_input(input: Any) -> bool`
  - `get_reflection_metadata() -> Dict[str, Any]`

### Control Node Contract
- All Control Nodes must implement:
  - `process(input_data: Dict[str, Any]) -> NodeResult`
  - `handle_error(error: Exception) -> NodeResult`

### Memory Interface Contract
- All Memory implementations must implement the Memory Protocol
- Memory operations must be type-safe and handle serialization/deserialization
- Memory operations must not modify input data

## Error Handling Contracts
- Functions that can fail should return Union[Result, Error] instead of raising exceptions
- All custom exceptions must inherit from a base AMAOS exception class
- Exception messages must be descriptive and follow a consistent format

## Common Pitfalls to Avoid
- Avoid recursive types without proper forward declarations
- Be careful with `isinstance()` and type narrowing
- Don't use mutable objects as default parameters
- Avoid Union[X, None] in favor of Optional[X]

> **Note**: These contracts are automatically enforced by mypy and the contract tests.
> Breaking these contracts will cause CI/CD pipelines to fail.
