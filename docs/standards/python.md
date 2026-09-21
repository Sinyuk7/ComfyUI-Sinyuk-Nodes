# Python Engineering Standards

These rules apply to all Python code in this repository.

## 1. Type Safety

All Python code must pass Pyright in strict mode.

Use:

```python
from __future__ import annotations
```

for every Python module.

All public functions and methods must have explicit parameter and return types.

Use modern type syntax:

```python
str | None
list[str]
dict[str, int]
tuple[str, ...]
```

Do not use unparameterized containers such as `list`, `dict`, or `tuple`.

Avoid `Any` and `Unknown` propagation. Dynamic or untyped external data must be validated and converted to a concrete internal type at the boundary.

Do not use `cast()`, `# type: ignore`, or broader types merely to silence Pyright errors.

## 2. Runtime Validation

Static typing does not replace runtime validation.

Validate data entering from external boundaries, including:

* HTTP and WebSocket responses
* JSON
* files and paths
* environment variables
* configuration
* third-party SDK responses

Do not add redundant runtime type checks for values already guaranteed by internal typed code.

## 3. Data Models

Use structured types when data has a known shape.

Prefer:

```text
dataclass   → internal data models
TypedDict   → dictionary / JSON structures
Enum        → semantic fixed values
Literal     → small fixed value sets
Protocol    → structural interfaces
```

Avoid long-lived `dict[str, Any]`.

Required fields should fail explicitly when missing. Use `.get()` only for genuinely optional fields.

## 4. ComfyUI V3

This project supports ComfyUI V3 only.

Nodes must use:

```python
io.ComfyNode
io.Schema
io.NodeOutput
ComfyExtension
```

Do not introduce legacy V1 APIs such as:

```text
INPUT_TYPES
RETURN_TYPES
FUNCTION
CATEGORY
NODE_CLASS_MAPPINGS
```

A node should only handle:

```text
schema
→ input adaptation
→ feature/common call
→ NodeOutput
```

Keep business logic outside node classes.

Published `node_id` values are part of the workflow compatibility contract and must remain stable.

## 5. Dependency Direction

The dependency direction is:

```text
nodes
  ↓
features
  ↓
common
```

`nodes/` contains ComfyUI adapters.

`features/` contains feature-specific business logic.

`common/` contains genuinely reusable infrastructure.

The following dependencies are forbidden:

```text
common → features
common → nodes
features → nodes
```

`common` must not become a miscellaneous utility directory.

## 6. Framework Boundaries

ComfyUI-specific code should remain inside:

```text
compat/
nodes/
extension.py
registry.py
```

`common/` must not depend on ComfyUI.

`features/` should normally remain independent from concrete ComfyUI node classes.

Core functionality should be importable and testable without starting ComfyUI.

## 7. Shared Infrastructure

Reuse shared implementations for cross-cutting functionality such as:

```text
image conversion / compression / resizing
HTTP requests / retries / downloads
filesystem and paths
JSON parsing
error normalization
```

Do not duplicate these implementations inside individual nodes.

Only move code into `common/` after it is genuinely reusable across multiple features.

Do not introduce speculative frameworks such as `BaseNode`, `BaseService`, provider factories, repositories, or dependency-injection systems without a demonstrated need.

## 8. Imports and Dependencies

Use `pathlib.Path` for filesystem paths.

Import order and formatting are enforced by Ruff.

Use `TYPE_CHECKING` for imports required only by static type checking:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
```

Do not hide dependency or import failures with broad fallback imports.

## 9. Error Handling

Never silently swallow exceptions.

Do not return fabricated fallback values merely to avoid errors.

Catch only exceptions the current layer can meaningfully handle.

Preserve exception chains when translating errors:

```python
try:
    ...
except SomeLibraryError as exc:
    raise NetworkError("Request failed") from exc
```

## 10. Testing

Code should be independently testable.

Lightweight nodes are tested by functional group:

```text
nodes/simple/string.py
tests/nodes/simple/test_string.py
```

Complex features receive dedicated tests:

```text
features/image_processing/
tests/features/image_processing/
```

Prioritize tests for:

```text
normal behavior
edge cases
invalid input
output type and structure
```

Network tests must use mocks and must not depend on live external APIs.

Business logic should normally be tested without starting ComfyUI.

## 11. Quality Gate

Before considering Python changes complete, the relevant checks must pass:

```bash
ruff check .
ruff format --check .
pyright
pytest
```

Fix the underlying code instead of weakening linting or type-checking rules.
