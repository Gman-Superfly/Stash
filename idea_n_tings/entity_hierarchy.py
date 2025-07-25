from abstractions.ecs.entity import Entity
from pydantic import Field
from typing import Optional
from uuid import UUID

class OperationEntity(Entity):
    """Base entity for all operations, with default priority and retries."""
    op_type: str = Field(default="", description="Type of operation (e.g., 'version_entity')")
    priority: int = Field(default=5, description="Priority level (1-10; higher = more important)")
    target_entity_id: UUID  # The entity this op targets
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=5, description="Max retries before rejection")
    parent_op_id: Optional[UUID] = Field(default=None, description="Parent operation for hierarchy")
    status: str = Field(default="pending")

class StructuralOperation(OperationEntity):
    """High-priority operations that must succeed (e.g., core structural changes)."""
    priority: int = Field(default=10)  # Override to high
    max_retries: int = Field(default=20)  # More retries for persistence

class NormalOperation(OperationEntity):
    """Standard operations with balanced priority."""
    priority: int = Field(default=5)  # Medium
    max_retries: int = Field(default=5)

class LowPriorityOperation(OperationEntity):
    """Low-priority ops that can yield or fail quickly."""
    priority: int = Field(default=2)  # Low
    max_retries: int = Field(default=3)  # Fewer retries to avoid contention

# Usage Examples:
# Import and use in your main code (e.g., where version_entity or other ops are defined)
#
# from entity_hierarchy import StructuralOperation, NormalOperation, LowPriorityOperation  # Import the subclasses you need
#
# # Example 1: Basic usage in version_entity for a high-priority structural op
# async def version_entity(cls, entity: "Entity", force_versioning: bool = False) -> bool:
#     op = StructuralOperation(target_entity_id=entity.ecs_id, op_type="versioning")  # Create op entity with high priority
#     op.promote_to_root()  # Promote to make it traceable in ECS
#     await emit(OperationStartedEvent(subject_id=op.ecs_id))  # Emit start event
#     
#     retries = 0
#     while retries <= op.max_retries:
#         # ... (fetch old_tree, build new_tree, detect conflict via version/last_modified)
#         if conflict_detected:
#             op.retry_count += 1  # Update retry count on the op entity
#             await emit(OperationConflictEvent(op_id=op.ecs_id, details=conflict_details))  # Emit conflict for handlers to resolve
#             if op.retry_count > op.max_retries:
#                 op.status = "rejected"  # Update status
#                 await emit(OperationRejectedEvent(op_id=op.ecs_id, reason="max_retries_exceeded"))  # Soft-fail via event
#                 return False
#             await asyncio.sleep(0.01 * (2 ** op.retry_count))  # Backoff based on retry count
#             continue
#         # Proceed with merge/fork (update_ecs_ids, register tree)
#         op.status = "succeeded"
#         await emit(OperationCompletedEvent(op_id=op.ecs_id))
#         return True
#
# # Example 2: Nested hierarchy for a pipeline (structural parent with normal child ops)
# async def structural_pipeline(entity: "Entity"):
#     parent_op = StructuralOperation(target_entity_id=entity.ecs_id, op_type="pipeline")  # High-pri parent
#     parent_op.promote_to_root()
#     await emit(OperationStartedEvent(subject_id=parent_op.ecs_id))
#     
#     # Sub-op inherits via parent_op_id
#     sub_op = NormalOperation(target_entity_id=entity.ecs_id, op_type="sub_update", parent_op_id=parent_op.ecs_id)
#     sub_op.promote_to_root()
#     # ... execute sub-op, which uses parent's priority if needed in conflict resolution
#
# # Example 3: Low-priority background task that yields easily
# async def low_pri_update(entity: "Entity"):
#     op = LowPriorityOperation(target_entity_id=entity.ecs_id, op_type="minor_update")
#     op.promote_to_root()
#     await emit(OperationStartedEvent(subject_id=op.ecs_id))
#     # ... short retry loop; on conflict, likely rejected by higher-pri ops via event handlers
#
# # Example 4: Conflict resolution handler (event-driven, no locks)
# @on(OperationConflictEvent)
# async def resolve_conflict(event: OperationConflictEvent):
#     op = get(f"@{event.op_id}")  # Fetch the op entity
#     conflicts = get_conflicting_ops(op.target_entity_id)  # Custom func to find ops on same target
#     for conflict_op in conflicts:
#         if conflict_op.priority < op.priority:
#             # Preempt lower-pri: Update status and emit rejection
#             conflict_op.status = "rejected"
#             await emit(OperationRejectedEvent(op_id=conflict_op.ecs_id, reason="preempted_by_higher_priority"))
#         elif conflict_op.priority == op.priority:
#             # Tie: Use last_modified (nanosecond timestamp) to decide winner
#             if op.last_modified < conflict_op.last_modified:
#                 # This op is earlier; preempt the other
#                 conflict_op.status = "rejected"
#                 await emit(OperationRejectedEvent(op_id=conflict_op.ecs_id, reason="tie_broken_by_timestamp"))
#     # If hierarchy, compute effective priority: max(op.priority, parent.priority if parent exists)
