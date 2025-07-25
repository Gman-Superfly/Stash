#!/usr/bin/env python3
"""Base Operational Entity for General Operations

This module provides BaseOperationalEntity as the base class for all
general operational capabilities that are not specific to ND (Neural Dynamics) operations.

BaseOperationalEntity inherits from abstractions.ecs.Entity and provides:
- Standard operational validation
- Performance metrics tracking
- Operation type classification
- General entity lifecycle management

Use this for:
- Infrastructure entities (storage, compute, monitoring)
- Workflow entities (orchestration, pipelines)
- RL entities (policies, trainers, buffers)
- Mathematical entities (computations, analysis)
- System management entities

Do NOT use this for ND-specific operations - use NDEntity instead.
"""

__all__ = ['BaseOperationalEntity']

from abstractions.ecs.entity import Entity
from pydantic import Field, ConfigDict
from typing import Dict, Any, Optional


class BaseOperationalEntity(Entity):
    """Base entity for general operational capabilities (non-ND specific).
    
    This class provides the foundation for all operational entities that perform
    general system functions but are not specific to Neural Dynamics operations.
    
    Inherits from abstractions.ecs.Entity to get:
    - Full ECS framework capabilities
    - Entity tree management and versioning
    - Event system integration
    - Provenance tracking
    - EntityRegistry integration
    
    Adds operational-specific features:
    - Operation type classification
    - Performance metrics tracking
    - Standard validation patterns
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    # Core operational fields
    operation_type: str = Field(..., description="Type of operation this entity performs")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance tracking data")
    created_by: str = Field(default="unknown", description="What created this entity")
    
    # Optional operational metadata
    operation_status: str = Field(default="active", description="Current operational status")
    last_operation_time: Optional[float] = Field(default=None, description="Timestamp of last operation")
    operation_count: int = Field(default=0, description="Number of operations performed")
    
    def validate_operational_state(self) -> bool:
        """Validate operational entity state for business logic and composition safety.
        
        This validation handles business logic that Pydantic field validators cannot:
        - Cross-field relationships and consistency
        - Operational readiness and state verification
        - Entity composition safety (when entities contain other entities)
        
        Note: Renamed from validate() to avoid Pydantic v2 BaseModel.validate(value) conflicts.
        Pydantic handles field validation (types, constraints) via @field_validator decorators.
        
        Returns:
            bool: True if entity is operationally valid, False otherwise
        """
        try:
            # Basic entity validation
            if not self.ecs_id:
                return False
                
            # Operation type validation
            if not self.operation_type or not isinstance(self.operation_type, str):
                return False
                
            # Status validation
            valid_statuses = {"active", "inactive", "error", "maintenance"}
            if self.operation_status not in valid_statuses:
                return False
                
            # Performance metrics validation
            if not isinstance(self.performance_metrics, dict):
                return False
                
            return True
            
        except Exception as e:
            print(f"Operational validation error in {self.__class__.__name__}: {e}")
            return False
    
    def record_performance_metric(self, metric_name: str, value: Any) -> None:
        """Record a performance metric for this operational entity.
        
        Args:
            metric_name: Name of the performance metric
            value: Value to record (can be any type)
        """
        self.performance_metrics[metric_name] = value
        
    def increment_operation_count(self) -> None:
        """Increment the operation counter and update last operation time."""
        import time
        self.operation_count += 1
        self.last_operation_time = time.time()
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of entity performance and status.
        
        Returns:
            Dict containing performance summary
        """
        return {
            "operation_type": self.operation_type,
            "operation_status": self.operation_status,
            "operation_count": self.operation_count,
            "last_operation_time": self.last_operation_time,
            "performance_metrics": self.performance_metrics.copy(),
            "entity_id": str(self.ecs_id)
        }
        
    def set_operation_status(self, status: str) -> None:
        """Set the operational status with validation.
        
        Args:
            status: New operational status
            
        Raises:
            ValueError: If status is not valid
        """
        valid_statuses = {"active", "inactive", "error", "maintenance"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        self.operation_status = status 