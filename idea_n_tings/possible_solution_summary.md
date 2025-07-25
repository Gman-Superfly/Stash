# Concurrency Solutions in the Abstractions Framework

This document provides detailed summaries of two key concurrency solutions implemented in the Abstractions framework. These address challenges in asynchronous, event-driven entity systems for distributed functional data processing. The solutions focus on optimistic concurrency control (OCC) without locks, ensuring immutability, traceability, and reactivity while handling conflicts, retries, and prioritization.

The framework treats data as immutable entities with provenance and lineage, transformed via pure functions in a callable registry. These solutions integrate seamlessly with the entity-native model, event system, and hierarchical operations.

## Solution 1: Monotonic Counter + Nanosecond Timestamps

### Problem It Solves
In an asynchronous entity system (built on ECS principles with immutable entities, tree hierarchies, and event-driven reactivity), multiple operations (e.g., versioning, forking, or merging changes via `version_entity`) can concurrently target the same shared entity or root tree. This leads to classic concurrency issues:
- **Races and Lost Updates**: Two async ops (e.g., E1 multiplying a value, E2 dividing) might snapshot the same initial state (A1), compute independently, and attempt to merge back. If E1 finishes first (updating to A5), E2's merge uses stale data, causing incorrect results or overwrites.
- **No Strict Ordering Without Locks**: Without synchronization, ops can't reliably detect or resolve conflicts, especially in distributed setups where clock skew or event cascades amplify timing variability.
- **Event-Driven Amplification**: Reactive events (e.g., `@on(ModifiedEvent)`) trigger bursts of async ops, increasing conflict likelihood without a way to timestamp/order them precisely.
- **Framework-Specific Risks**: In the immutable model, rebuilds (e.g., `build_entity_tree`) and diffs (e.g., `find_modified_entities`) are expensive; unresolved conflicts waste resources on repeated failures, and hard failures violate the "don't fail, reject via retry/event" preference.

This results in data corruption, infinite loops in retries, or system instability in high-contention scenarios (e.g., rapid cascades from multi-entity outputs or unions).

### Solution Details
The solution introduces a **composite key** for optimistic concurrency control (OCC) directly on the base `Entity` class, inherited by all entities (e.g., `Student`, `OperationEntity`):
- **`version: int = 0`**: A monotonic incrementing counter, bumped atomically on every successful update/fork/merge (e.g., in `update_ecs_ids` via `self.version += 1`).
- **`last_modified: int = Field(default_factory=time.monotonic_ns)`**: A high-resolution, monotonic timestamp in nanoseconds (from Python's `time.monotonic_ns()`—non-decreasing since an arbitrary point like boot time), updated on each change (e.g., `self.last_modified = time.monotonic_ns()`).

**How It Works**:
- **Snapshot on Op Start**: When an op begins (e.g., in `version_entity`), snapshot the target's current `version` and `last_modified`.
- **Conflict Detection on Merge**: Before applying changes, compare snapshots to the current state. Mismatch (e.g., version increased or last_modified newer) indicates interference—trigger retry with fresh data (re-fetch tree, rebuild).
- **Tie-Breaking**: If versions match (rare near-simultaneous ops), use last_modified (smaller = earlier) to decide winner or order.
- **Integration**: Added to base `Entity` for all (inherited); used in retry loops with backoff (e.g., `await asyncio.sleep(0.01 * (2 ** retries))`). On max retries, soft-fail via event.
- **Code Example**:
  ```python
  import time
  from pydantic import Field

  class Entity(BaseModel):
      # ... existing fields
      version: int = 0
      last_modified: int = Field(default_factory=time.monotonic_ns)

  # In update_ecs_ids (forking):
  self.version += 1
  self.last_modified = time.monotonic_ns()

  # In version_entity (async retry loop):
  while retries <= max_retries:
      # ... build trees, compute diff
      if old_root.version != entity.version or old_root.last_modified != entity.last_modified:
          retries += 1
          if retries > max_retries:
              # Emit rejection event (e.g., OperationRejectedEvent)
              return False
          await asyncio.sleep(0.01 * (2 ** retries))  # Exponential backoff
          continue
      # Proceed with merge
  ```

### Benefits
- **Lock-Free**: Pure OCC—cheap reads, detects conflicts post-compute.
- **Precise Ordering**: Monotonicity avoids clock issues; ns resolution minimizes ties.
- **Scales Async/Distributed**: Works across nodes (monotonic per-process, but combined with UUIDs/lineage for global trace).
- **Fits Framework Philosophy**: Entity-native (fields on Entity), no queues—retries use async yields; provenance (lineage_id) preserved.

### Limitations
- Doesn't prioritize ops; all retry equally, which can lead to resource waste or starvation in mixed-priority workloads. This is addressed by the second solution.

## Solution 2: Hierarchy of Operation Entities

### Problem It Solves
While the monotonic/timestamp solution handles basic conflict detection/retries, it treats all ops equally, causing issues in diverse, event-reactive workloads:
- **No Prioritization**: High-importance "structural" tasks (e.g., tree rebuilds, root promotions) compete equally with low-pri ones (e.g., minor updates from events), leading to starvation (high-pri fails due to persistent low-pri interference) or inefficiency (equal retries waste resources on low-pri).
- **Resource Waste in Contention**: In bursts (e.g., events triggering parallel async ops on shared entities), flat retries amplify rebuilds/diffs without adapting—high-pri tasks might exhaust retries while low-pri hog CPU.
- **No Contextual Inheritance**: In nested cascades (e.g., a structural pipeline triggering sub-ops), sub-ops don't inherit importance, risking rejection of critical chains.
- **Unbounded Loops/Fairness**: Variable retries (more for structural) without coordination could loop indefinitely on systemic contention, or low-pri ops delay high-pri ones, violating "structural takes priority."
- **Framework-Specific Risks**: Reactivity (events like `ModifiedEvent` creating op bursts) and distributed nature (addressing across nodes) amplify this; without structure, retries don't scale to "must-succeed" vs. "can-fail" ops.

This leads to unpredictable outcomes, resource imbalance, and potential livelock in high-load scenarios, while still needing to "retry more for certain tasks" without failing outright.

### Solution Details
Introduce a hierarchy of operation entities (subclasses of `Entity`) to model tasks as prioritized, traceable entities. Each op is an `OperationEntity` instance (promoted to root), with subclasses overriding defaults for behavior. Hierarchy via `parent_op_id` to form trees/chains.

- **Base `OperationEntity`**:
  - Fields: `op_type` (e.g., "version"), `priority` (1-10), `target_entity_id` (op's focus), `retry_count`, `max_retries`, `parent_op_id` (for chains), `status` ("pending"/"retrying"/etc.).
  - Inherits monotonic versioning from base `Entity`.

- **Subclasses**:
  - `StructuralOperation`: High pri (10), more retries (20)—for must-succeed (e.g., core changes).
  - `NormalOperation`: Medium pri (5), standard retries (5).
  - `LowPriorityOperation`: Low pri (2), few retries (3)—can yield/fail quickly.

**How It Works**:
- **Op Creation**: In `version_entity`, create op entity (e.g., `op = StructuralOperation(target_entity_id=entity.ecs_id)`), emit `OperationStartedEvent`.
- **Retry Loop in `version_entity`**: Use `op.max_retries`/`op.priority` for limits/backoff. On conflict, emit `OperationConflictEvent` with op details.
- **Event-Driven Resolution** (`@on(OperationConflictEvent)` handler):
  - Fetch conflicting ops on same `target_entity_id`.
  - Compare priorities: High-pri preempts low-pri (emit `OperationRejectedEvent` to abort low-pri).
  - Use hierarchy: Traverse `parent_op_id` to form effective pri (e.g., max of chain); inherit for sub-ops in cascades.
  - Ties: Resolve via timestamps/lineage (e.g., earlier op wins).
- **On Exhaustion**: Emit `OperationRejectedEvent`, soft-fail—high-pri might trigger resource alerts.
- **Integration**: Ops as entities are queryable (via registry); events carry pri for adaptive dispatch (e.g., process high-pri first in bus queues).

**Code Example**:
```python
from uuid import UUID
from typing import Optional
from pydantic import Field

class OperationEntity(Entity):
    op_type: str = Field(default="", description="Type of operation (e.g., 'version_entity')")
    priority: int = Field(default=5, description="Priority level (1-10; higher = more important)")
    target_entity_id: UUID  # The entity this op targets
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=5, description="Max retries before rejection")
    parent_op_id: Optional[UUID] = Field(default=None, description="Parent operation for hierarchy")
    status: str = Field(default="pending")

class StructuralOperation(OperationEntity):
    priority: int = Field(default=10)  # Override to high
    max_retries: int = Field(default=20)  # More retries for persistence

class NormalOperation(OperationEntity):
    priority: int = Field(default=5)  # Medium
    max_retries: int = Field(default=5)

class LowPriorityOperation(OperationEntity):
    priority: int = Field(default=2)  # Low
    max_retries: int = Field(default=3)  # Fewer retries to avoid contention
```

### Benefits
- **Prioritized Retries**: Structural ops retry more/resource-preferentially; low-pri yield.
- **No Locks**: Optimistic + events; conflicts resolve async via handlers.
- **Resource Fairness**: High-pri ops "win" without starving system—rejects low-pri early.
- **Hierarchy for Context**: Nested ops inherit pri (e.g., sub-op in structural chain gets high pri).
- **Fits Framework Philosophy**: Entity-native (ops as entities), event-reactive, no queues—preemption via events.

### Limitations
- Adds event overhead for conflicts; hierarchies need traversal (keep shallow). Requires careful event handler design to avoid cycles.

## How the Solutions Complement Each Other
- **Solution 1** provides the foundational detection mechanism (timestamps for conflicts), enabling safe retries.
- **Solution 2** layers prioritization and hierarchy on top, making retries adaptive and fair in mixed workloads.
- Together: Use timestamps for low-level OCC; hierarchy for high-level coordination. Integrate via events for reactivity.

For implementation questions or testing, refer to the framework's examples or reach out to collaborators.
