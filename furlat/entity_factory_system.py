#!/usr/bin/env python3
"""Entity Factory System - LLM-Friendly Entity Creation with Full Tracking

This module provides a unified system for creating, registering, and managing entities
that can be safely used by LLMs while maintaining all tracking, versioning, and 
system integration capabilities.

Key Features:
- Auto-registration without ID changes
- LLM-safe creation patterns
- Full provenance tracking
- CallableRegistry integration
- Event system coordination
- Intelligent validation (avoids Pydantic v2 conflicts)
- Generic validation tiers (Domain → Operational → Workflow)
"""

__all__ = [
    'EntityFactory', 'LLMEntityCreator', 'EntityRegistrationConfig',
    'create_entity_with_registration', 'create_root_entity', 'register_existing_entity',
    'get_entity_creation_config', 'validate_entity_for_registration',
    'EntityCreationStrategy', 'RegistrationMode'
]

import time
import json
from typing import Dict, List, Optional, Any, Union, Type, Callable, Tuple
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
import logging

# Core imports
from abstractions.ecs.entity import Entity, EntityRegistry
from abstractions.ecs.callable_registry import CallableRegistry
from abstractions.events.events import emit_events, ProcessingEvent, ProcessedEvent
from ..base_ops_entity import BaseOperationalEntity

logger = logging.getLogger(__name__)


class EntityCreationStrategy(str, Enum):
    """Strategies for entity creation and registration."""
    AUTO_ROOT = "auto_root"              # Automatically make entities root entities
    MANUAL_ROOT = "manual_root"          # Require explicit root promotion
    LAZY_REGISTRATION = "lazy_registration"  # Register only when needed
    IMMEDIATE_REGISTRATION = "immediate_registration"  # Register immediately
    LLM_FRIENDLY = "llm_friendly"        # Optimized for LLM creation patterns


class RegistrationMode(str, Enum):
    """Modes for entity registration."""
    STABLE_ID = "stable_id"              # Keep original ecs_id, set root_ecs_id manually
    VERSIONED = "versioned"              # Use promote_to_root() for versioning
    NO_REGISTRATION = "no_registration"   # Create but don't register
    CONDITIONAL = "conditional"          # Register based on entity type/properties


class EntityRegistrationConfig(BaseModel):
    """Configuration for entity registration behavior."""
    creation_strategy: EntityCreationStrategy = EntityCreationStrategy.LLM_FRIENDLY
    registration_mode: RegistrationMode = RegistrationMode.STABLE_ID
    auto_validate: bool = True
    emit_events: bool = True
    track_provenance: bool = True
    llm_creation_context: Optional[Dict[str, Any]] = None
    force_registration: bool = False
    validation_level: str = "standard"  # "minimal", "standard", "strict"


class EntityFactory:
    """Unified entity factory with LLM-friendly creation patterns."""
    
    _default_config: EntityRegistrationConfig = EntityRegistrationConfig()
    _creation_history: List[Dict[str, Any]] = []
    _validation_registry: Dict[Type[Entity], Callable] = {}
    
    @classmethod
    def set_default_config(cls, config: EntityRegistrationConfig) -> None:
        """Set the default configuration for entity creation."""
        cls._default_config = config
        logger.info(f"Default entity creation config updated: {config.creation_strategy}")
    
    @classmethod
    def register_validator(cls, entity_type: Type[Entity], validator: Callable[[Entity], bool]) -> None:
        """Register a custom validator for an entity type."""
        cls._validation_registry[entity_type] = validator
        logger.info(f"Registered validator for {entity_type.__name__}")
    
    @classmethod
    def create_entity(
        cls,
        entity_class: Type[Entity],
        config: Optional[EntityRegistrationConfig] = None,
        **kwargs
    ) -> Entity:
        """Create an entity with full registration and tracking.
        
        Args:
            entity_class: Entity class to instantiate
            config: Registration configuration (uses default if None)
            **kwargs: Arguments for entity creation
            
        Returns:
            Created and optionally registered entity
            
        Raises:
            ValueError: If validation fails or registration requirements not met
        """
        config = config or cls._default_config
        
        # Step 1: Create entity instance
        entity = entity_class(**kwargs)
        
        # Step 2: Validation
        if config.auto_validate:
            cls._validate_entity(entity, config.validation_level)
        
        # Step 3: Provenance tracking
        if config.track_provenance:
            cls._setup_provenance(entity, config)
        
        # Step 4: Registration handling
        if config.registration_mode != RegistrationMode.NO_REGISTRATION:
            cls._handle_registration(entity, config)
        
        # Step 5: Event emission
        if config.emit_events:
            cls._emit_creation_events(entity, config)
        
        # Step 6: Track creation
        cls._record_creation(entity, config)
        
        return entity
    
    @classmethod
    def create_root_entity(
        cls,
        entity_class: Type[Entity],
        register: bool = True,
        **kwargs
    ) -> Entity:
        """Create an entity as a root entity with stable ID.
        
        This is the LLM-friendly method that maintains ID stability.
        """
        config = EntityRegistrationConfig(
            creation_strategy=EntityCreationStrategy.LLM_FRIENDLY,
            registration_mode=RegistrationMode.STABLE_ID if register else RegistrationMode.NO_REGISTRATION,
            auto_validate=True,
            emit_events=True,
            track_provenance=True
        )
        
        return cls.create_entity(entity_class, config, **kwargs)
    
    @classmethod
    def _validate_entity(cls, entity: Entity, validation_level: str) -> None:
        """Validate entity with intelligent method detection to avoid Pydantic conflicts.
        
        Uses generic three-tier validation hierarchy:
        - Tier 1: Domain-specific validation (validate_domain_state)
        - Tier 2: Operational validation (validate_operational_state) 
        - Tier 3: Workflow validation (validate_workflow_state)
        """
        # Standard Pydantic validation happens during creation
        
        # Custom validation
        entity_type = type(entity)
        if entity_type in cls._validation_registry:
            validator = cls._validation_registry[entity_type]
            if not validator(entity):
                raise ValueError(f"Custom validation failed for {entity_type.__name__}")
        
        # Intelligent validation with full traceability using generic tier hierarchy
        validation_results = {}
        
        # TIER 1: Domain-specific validation (highest priority)
        if hasattr(entity, 'validate_domain_state'):
            try:
                validation_results['domain_state'] = entity.validate_domain_state()
                if not validation_results['domain_state']:
                    raise ValueError(f"Domain state validation failed for {entity_type.__name__}")
            except Exception as e:
                raise ValueError(f"Domain validation error for {entity_type.__name__}: {e}")
        
        # TIER 2: Operational validation
        elif hasattr(entity, 'validate_operational_state'):
            try:
                validation_results['operational_state'] = entity.validate_operational_state()
                if not validation_results['operational_state']:
                    raise ValueError(f"Operational state validation failed for {entity_type.__name__}")
            except Exception as e:
                raise ValueError(f"Operational validation error for {entity_type.__name__}: {e}")
        
        # TIER 3: Workflow validation
        elif hasattr(entity, 'validate_workflow_state'):
            try:
                validation_results['workflow_state'] = entity.validate_workflow_state()
                if not validation_results['workflow_state']:
                    raise ValueError(f"Workflow state validation failed for {entity_type.__name__}")
            except Exception as e:
                raise ValueError(f"Workflow validation error for {entity_type.__name__}: {e}")
        
        # Fallback: Try generic validate() with conflict detection
        elif hasattr(entity, 'validate') and callable(getattr(entity, 'validate')):
            try:
                # Check method signature to detect Pydantic vs entity validate methods
                import inspect
                validate_method = getattr(entity, 'validate')
                sig = inspect.signature(validate_method)
                
                # If it's entity's custom validate (no required parameters), call it
                required_params = [p for p in sig.parameters.values() if p.default == p.empty]
                if len(required_params) == 0:  # Entity's validate() method
                    validation_results['generic'] = validate_method()
                    if not validation_results['generic']:
                        raise ValueError(f"Generic validation failed for {entity_type.__name__}")
                else:
                    # This is likely Pydantic's validate method - skip to avoid conflict
                    validation_results['skipped_pydantic_conflict'] = True
                    
            except TypeError as e:
                if "missing 1 required positional argument: 'value'" in str(e):
                    # This is the Pydantic conflict - log and skip
                    validation_results['pydantic_conflict_detected'] = True
                    logger.warning(f"Pydantic validation conflict detected for {entity_type.__name__} - skipping generic validate()")
                else:
                    raise ValueError(f"Validation error for {entity_type.__name__}: {e}")
        else:
            # No specific validation method found
            validation_results['no_validation_method'] = True
        
        # Record validation results for traceability
        if hasattr(entity, 'performance_metrics'):
            if entity.performance_metrics is None:
                entity.performance_metrics = {}
            entity.performance_metrics['factory_validation_results'] = validation_results
            entity.performance_metrics['validation_timestamp'] = time.time()
        
        # Strict validation
        if validation_level == "strict":
            cls._strict_validation(entity)
    
    @classmethod
    def _strict_validation(cls, entity: Entity) -> None:
        """Perform strict validation checks."""
        # Check all required fields are set
        for field_name, field_info in entity.model_fields.items():
            if field_info.is_required():
                value = getattr(entity, field_name, None)
                if value is None:
                    raise ValueError(f"Required field {field_name} is None")
        
        # Check ECS ID integrity
        if not entity.ecs_id:
            raise ValueError("Entity missing ecs_id")
        
        # Check for obvious data corruption
        if hasattr(entity, 'performance_metrics'):
            metrics = entity.performance_metrics
            if metrics and not isinstance(metrics, dict):
                raise ValueError("performance_metrics must be a dictionary")
    
    @classmethod
    def _setup_provenance(cls, entity: Entity, config: EntityRegistrationConfig) -> None:
        """Set up provenance tracking for the entity."""
        # Add creation metadata to performance_metrics if available
        if hasattr(entity, 'performance_metrics'):
            creation_info = {
                'created_by': 'EntityFactory',
                'creation_strategy': config.creation_strategy,
                'creation_timestamp': time.time(),
                'factory_version': '1.0.0'
            }
            
            if config.llm_creation_context:
                creation_info['llm_context'] = config.llm_creation_context
            
            if entity.performance_metrics is None:
                entity.performance_metrics = {}
            entity.performance_metrics['creation_info'] = creation_info
    
    @classmethod
    def _handle_registration(cls, entity: Entity, config: EntityRegistrationConfig) -> None:
        """Handle entity registration based on configuration."""
        if config.registration_mode == RegistrationMode.STABLE_ID:
            # Make root entity without changing ecs_id
            entity.root_ecs_id = entity.ecs_id
            entity.root_live_id = entity.live_id
            
            # Register with EntityRegistry
            try:
                EntityRegistry.register_entity(entity)
                logger.info(f"Registered entity {entity.ecs_id} with stable ID")
            except Exception as e:
                if config.force_registration:
                    raise ValueError(f"Failed to register entity: {e}")
                else:
                    logger.warning(f"Registration failed for {entity.ecs_id}: {e}")
        
        elif config.registration_mode == RegistrationMode.VERSIONED:
            # Use promote_to_root for versioning workflow
            entity.promote_to_root()
            logger.info(f"Promoted entity to root with new ID: {entity.ecs_id}")
        
        elif config.registration_mode == RegistrationMode.CONDITIONAL:
            # Register based on entity characteristics
            should_register = (
                hasattr(entity, 'operation_type') or  # Operational entities
                isinstance(entity, BaseOperationalEntity) or  # Base operational entities
                config.force_registration
            )
            
            if should_register:
                entity.root_ecs_id = entity.ecs_id
                entity.root_live_id = entity.live_id
                EntityRegistry.register_entity(entity)
    
    @classmethod
    def _emit_creation_events(cls, entity: Entity, config: EntityRegistrationConfig) -> None:
        """Emit events for entity creation."""
        # This could be enhanced to emit custom events
        # For now, just log the creation
        logger.info(f"Created entity {entity.__class__.__name__} with ID {entity.ecs_id}")
    
    @classmethod
    def _record_creation(cls, entity: Entity, config: EntityRegistrationConfig) -> None:
        """Record entity creation in history."""
        creation_record = {
            'entity_id': str(entity.ecs_id),
            'entity_type': entity.__class__.__name__,
            'creation_strategy': config.creation_strategy,
            'registration_mode': config.registration_mode,
            'timestamp': time.time(),
            'is_registered': entity.root_ecs_id is not None
        }
        
        cls._creation_history.append(creation_record)
        
        # Keep only last 1000 records
        if len(cls._creation_history) > 1000:
            cls._creation_history = cls._creation_history[-1000:]
    
    @classmethod
    def get_creation_history(cls) -> List[Dict[str, Any]]:
        """Get the entity creation history."""
        return cls._creation_history.copy()
    
    @classmethod
    def get_creation_stats(cls) -> Dict[str, Any]:
        """Get statistics about entity creation."""
        if not cls._creation_history:
            return {"total_created": 0}
        
        total = len(cls._creation_history)
        registered = sum(1 for record in cls._creation_history if record['is_registered'])
        
        entity_types = {}
        strategies = {}
        
        for record in cls._creation_history:
            entity_type = record['entity_type']
            strategy = record['creation_strategy']
            
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        return {
            "total_created": total,
            "registered": registered,
            "unregistered": total - registered,
            "entity_types": entity_types,
            "strategies": strategies,
            "registration_rate": registered / total if total > 0 else 0
        }


# Convenience functions for easy imports and LLM usage

def create_entity_with_registration(
    entity_class: Type[Entity],
    register: bool = True,
    **kwargs
) -> Entity:
    """Simple function to create and optionally register any entity."""
    return EntityFactory.create_root_entity(entity_class, register=register, **kwargs)


def create_root_entity(entity_class: Type[Entity], **kwargs) -> Entity:
    """Create an entity as a root entity with registration."""
    return EntityFactory.create_root_entity(entity_class, register=True, **kwargs)


def register_existing_entity(entity: Entity, force: bool = False) -> bool:
    """Register an existing entity that wasn't registered during creation."""
    try:
        if entity.root_ecs_id is None:
            entity.root_ecs_id = entity.ecs_id
            entity.root_live_id = entity.live_id
        
        EntityRegistry.register_entity(entity)
        return True
    except Exception as e:
        if force:
            raise ValueError(f"Failed to register entity: {e}")
        logger.warning(f"Registration failed for {entity.ecs_id}: {e}")
        return False


def get_entity_creation_config(strategy: EntityCreationStrategy) -> EntityRegistrationConfig:
    """Get a pre-configured entity creation configuration."""
    configs = {
        EntityCreationStrategy.LLM_FRIENDLY: EntityRegistrationConfig(
            creation_strategy=EntityCreationStrategy.LLM_FRIENDLY,
            registration_mode=RegistrationMode.STABLE_ID,
            auto_validate=True,
            emit_events=True,
            track_provenance=True
        ),
        EntityCreationStrategy.AUTO_ROOT: EntityRegistrationConfig(
            creation_strategy=EntityCreationStrategy.AUTO_ROOT,
            registration_mode=RegistrationMode.STABLE_ID,
            auto_validate=True,
            emit_events=False,
            track_provenance=False
        ),
        EntityCreationStrategy.IMMEDIATE_REGISTRATION: EntityRegistrationConfig(
            creation_strategy=EntityCreationStrategy.IMMEDIATE_REGISTRATION,
            registration_mode=RegistrationMode.STABLE_ID,
            auto_validate=True,
            emit_events=True,
            track_provenance=True,
            force_registration=True
        )
    }
    
    return configs.get(strategy, EntityFactory._default_config)


def validate_entity_for_registration(entity: Entity) -> Tuple[bool, List[str]]:
    """Validate whether an entity can be safely registered.
    
    Uses generic validation tier hierarchy for compatibility across domains.
    """
    issues = []
    
    # Check basic requirements
    if not entity.ecs_id:
        issues.append("Entity missing ecs_id")
    
    # Check if already registered
    if entity.root_ecs_id and entity.root_ecs_id == entity.ecs_id:
        if EntityRegistry.get_stored_entity(entity.root_ecs_id, entity.ecs_id):
            issues.append("Entity already registered")
    
    # Check entity-specific validation using generic tier hierarchy
    if hasattr(entity, 'validate_domain_state'):
        if not entity.validate_domain_state():
            issues.append("Entity validate_domain_state() returned False")
    elif hasattr(entity, 'validate_operational_state'):
        if not entity.validate_operational_state():
            issues.append("Entity validate_operational_state() returned False")
    elif hasattr(entity, 'validate_workflow_state'):
        if not entity.validate_workflow_state():
            issues.append("Entity validate_workflow_state() returned False")
    
    return len(issues) == 0, issues


# CallableRegistry integration
@CallableRegistry.register("create_entity_with_factory")
@emit_events(
    creating_factory=lambda entity_type, validation_domain="default", complexity_level=0.7, auto_register=True: ProcessingEvent(
        event_type="processing",
        subject_type=None,
        subject_id=None,
        process_name="create_entity_with_factory",
        metadata={"operation": "create_entity_with_factory", "entity_type": entity_type}
    ),
    created_factory=lambda result, entity_type, validation_domain="default", complexity_level=0.7, auto_register=True: ProcessedEvent(
        event_type="processed",
        subject_type=None,
        subject_id=None,
        process_name="create_entity_with_factory",
        metadata={"operation": "create_entity_with_factory", "result_id": result}
    )
)
def create_entity_with_factory_callable(
    entity_type: str, 
    validation_domain: str = "default",
    complexity_level: float = 0.7,
    auto_register: bool = True
) -> str:
    """CallableRegistry function to create entities via the factory system.
    
    Args:
        entity_type: Type of entity to create ("validator", "array_data", "system_introspector")
        validation_domain: Domain for validation entities (generic approach)
        complexity_level: Complexity level (0.0-1.0)
        auto_register: Whether to auto-register the entity
        
    Returns:
        String ID of the created entity
    """
    # For now, return a simple mock entity ID until LLMEntityCreator is implemented
    from uuid import uuid4
    return str(uuid4()) 