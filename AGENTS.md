Before modifying Python code, read and follow:
- docs/standards/python.md

## Minimal Change Policy

Keep this repository small, focused, and inexpensive to maintain.

Make the smallest coherent change required for the current task. Do not
refactor unrelated code, clean up unrelated files, introduce speculative
abstractions, or expand a task into adjacent improvements. Prefer modifying
an existing owner over creating a new layer, file, or abstraction.

### Artifact Creation Gate

Before adding a file, abstraction, document, helper, or test, confirm:

1. What concrete current problem requires it?
2. Why can an existing file, module, or test not own it?
3. What stable behavior or repository rule would be lost without it?

If the artifact only anticipates a possible future need, duplicates existing
information, documents obvious code, or provides no meaningful behavioral
protection, do not add it.

### Documentation

Do not create documentation by default. Prefer updating the existing
canonical document. Documentation is justified only for durable information
that cannot be expressed clearly in code, tests, or an existing document.

Do not create design documents, implementation notes, progress or build logs,
planning files, architecture documents, additional README files, or duplicate
documentation unless explicitly requested or clearly required by a substantial
long-lived change. Keep documentation concise and remove obsolete or repeated
content during related changes.

### Tests

Add only the smallest tests that protect meaningful behavior, regressions,
important edge cases, public contracts, workflow compatibility, or reusable
core logic. Do not add tests merely to increase coverage or test static
constants, trivial declarations, framework boilerplate, or implementation
details already covered by higher-value behavior tests.

During refactors, update existing tests instead of duplicating them unless
both behaviors intentionally remain supported. Prefer focused tests grouped by
functional module.

### Validation and Refactoring

Run the narrowest validation that gives confidence: changed-code tests first,
then relevant type and lint checks, and broader validation only when shared
infrastructure or repository-wide contracts changed.

Refactors must reduce or preserve maintenance cost. Prefer deletion and
consolidation over additional compatibility layers, artifacts, or speculative
infrastructure.

When uncertain whether to add structure, do not add it yet. Add structure only
when current code demonstrates a concrete need.

## Ownership and Growth

Avoid uncontrolled growth of nodes, modules, and abstractions.

Every new capability must have one clear owner.

### Before Adding a Node

Do not create a new node merely because the implementation differs from an existing node.

Create a new node only when it represents a distinct workflow operation, responsibility, input/output contract, side effect, or execution lifecycle.

Extend an existing node only when the new behavior:

* belongs to the same responsibility
* preserves the meaning of existing inputs and outputs
* preserves existing default behavior
* does not require turning the node into a collection of unrelated modes

Code differences alone do not justify a new node. Workflow semantics do.

Avoid both:

* multiple nodes representing the same capability
* large nodes that accumulate unrelated behavior through mode switches

Existing published node contracts should remain stable. Do not repurpose an existing node to provide a different capability.

### Code Ownership

Place code according to its responsibility:

```text
nodes/      ComfyUI workflow adapters
features/   feature-specific behavior
common/     shared low-level infrastructure
compat/     framework/version boundaries
```

Keep dependencies flowing:

```text
nodes → features → common
```

Do not duplicate the same responsibility across layers.

### Common Code

Do not place code in `common/` preemptively.

Start feature-specific logic inside its owning feature.

Promote code to `common/` only when multiple independent features genuinely require the same underlying capability.

Prefer moving proven shared behavior to `common/` over designing generic utilities in advance.

### Growth Gate

Before adding a node, module, abstraction, or shared utility, determine:

1. What responsibility is being added?
2. Which existing component owns that responsibility?
3. Can the existing owner support it without changing its contract or becoming less cohesive?
4. Is a new user-visible workflow operation actually required?
5. Is the proposed shared code already needed by multiple independent consumers?

If ownership is unclear, do not add the new structure until the responsibility is clarified.

