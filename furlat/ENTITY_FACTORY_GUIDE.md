# EntityFactory System: Essential Guide

**A concise guide to implementing an entity factory system that provides LLM-friendly, stable ID entity creation while maintaining full ECS integration.**

**Universal Design Pattern**: This system uses generic validation tiers (Domain → Operational → Workflow) that can be adapted to any domain without changing the core architecture.

---

## 🎯 **Core Problem Solved**

Traditional ECS entity creation uses `promote_to_root()` which can change entity IDs during registration, breaking external references. The EntityFactory solves this through **manual root assignment** with stable IDs.

```python
# Traditional approach - unpredictable IDs
entity = MyEntity(param="value")
entity.promote_to_root()  # May change ID!

# EntityFactory approach - stable IDs
entity = MyEntity(param="value")
entity.root_ecs_id = entity.ecs_id  # Manual assignment preserves ID
EntityRegistry.register_entity(entity)
```

---

## 🏗️ **Core Components**

### **EntityCreationStrategy**
```python
LLM_FRIENDLY = "llm_friendly"        # Optimized for AI systems (recommended)
AUTO_ROOT = "auto_root"              # Traditional ECS patterns
LAZY_REGISTRATION = "lazy_registration"  # Defer registration for performance
IMMEDIATE_REGISTRATION = "immediate_registration"  # Register immediately
MANUAL_ROOT = "manual_root"          # Explicit control over promotion
```

### **RegistrationMode**
```python
STABLE_ID = "stable_id"              # Keep original ID (recommended for Claude)
VERSIONED = "versioned"              # Allow ID changes for versioning
NO_REGISTRATION = "no_registration"   # Create without registering
CONDITIONAL = "conditional"          # Register based on entity properties
```

### **EntityRegistrationConfig**
```python
class EntityRegistrationConfig(BaseModel):
    creation_strategy: EntityCreationStrategy = EntityCreationStrategy.LLM_FRIENDLY
    registration_mode: RegistrationMode = RegistrationMode.STABLE_ID
    auto_validate: bool = True
    emit_events: bool = True
    track_provenance: bool = True
    llm_creation_context: Optional[Dict[str, Any]] = None
    force_registration: bool = False
    validation_level: str = "standard"  # "minimal", "standard", "strict"
```

---

## 🔧 **Core Usage**

### **Primary Method for Claude/LLM Integration**
```python
# Create entity with stable ID (recommended for Claude)
entity = EntityFactory.create_root_entity(
    EntityClass,
    register=True,  # Default
    **kwargs
)
```

### **Full Control Method**
```python
# Create with custom configuration
config = EntityRegistrationConfig(
    creation_strategy=EntityCreationStrategy.LLM_FRIENDLY,
    registration_mode=RegistrationMode.STABLE_ID,
    llm_creation_context={"session_id": "abc-123"}
)

entity = EntityFactory.create_entity(EntityClass, config, **kwargs)
```

### **Convenience Functions**
```python
# Simple creation and registration
entity = create_entity_with_registration(EntityClass, **kwargs)

# Create as root entity
entity = create_root_entity(EntityClass, **kwargs)

# Register existing entity
success = register_existing_entity(entity, force=False)
```

---

## ✅ **Validation System**

### **Generic Three-Tier Validation Hierarchy**
The system uses domain-agnostic validation tiers:

1. **Domain-specific** (`validate_domain_state`) - Business/domain logic
2. **Operational** (`validate_operational_state`) - System operations
3. **Workflow** (`validate_workflow_state`) - Process workflows

### **Domain Mapping Example**
```python
class MathematicalEntity(Entity):
    def validate_mathematical_state(self) -> bool:
        return self.validate_domain_state()  # Maps to Tier 1
    
    def validate_domain_state(self) -> bool:
        return self.value > 0 and not math.isnan(self.result)
```

### **Validation Levels**
- **minimal**: Basic checks only (fastest)
- **standard**: Balanced validation (recommended)
- **strict**: Comprehensive validation (slowest but thorough)

### **Conflict Avoidance**
The system uses method signature introspection to avoid calling framework validation methods (like Pydantic) incorrectly.

---

## 📝 **Registration Modes Explained**

### **STABLE_ID Mode (Recommended)**
```python
# Preserves original entity ID
entity.root_ecs_id = entity.ecs_id  # Manual assignment
EntityRegistry.register_entity(entity)  # Direct registration
# Result: ID never changes - perfect for Claude integration
```

### **VERSIONED Mode**
```python
# Traditional ECS with potential ID changes
entity.promote_to_root()
# Result: ID may change for versioning workflows
```

### **NO_REGISTRATION Mode**
```python
# Create without registering (performance optimization)
# Entity exists but not in registry - register manually later
```

### **CONDITIONAL Mode**
```python
# Intelligent registration based on entity characteristics
if (hasattr(entity, 'operation_type') or 
    isinstance(entity, BaseOperationalEntity)):
    # Register automatically
```

---

## 📊 **Tracking and Analytics**

### **Provenance Tracking**
Every entity gets creation metadata stored in `performance_metrics`:
```python
creation_info = {
    'created_by': 'EntityFactory',
    'creation_strategy': config.creation_strategy,
    'creation_timestamp': time.time(),
    'llm_context': config.llm_creation_context
}
```

### **Analytics Methods**
```python
# Get creation statistics
stats = EntityFactory.get_creation_stats()

# Get creation history
history = EntityFactory.get_creation_history()

# Monitor system health
health = monitor_entity_factory_health()
```

---

## 🔍 **Configuration Examples**

### **Claude Integration (Recommended)**
```python
claude_config = EntityRegistrationConfig(
    creation_strategy=EntityCreationStrategy.LLM_FRIENDLY,
    registration_mode=RegistrationMode.STABLE_ID,
    auto_validate=True,
    track_provenance=True,
    llm_creation_context={"claude_session_id": "session-123"}
)
```

### **High Performance**
```python
performance_config = EntityRegistrationConfig(
    creation_strategy=EntityCreationStrategy.LAZY_REGISTRATION,
    registration_mode=RegistrationMode.NO_REGISTRATION,
    auto_validate=False,
    emit_events=False,
    track_provenance=False,
    validation_level="minimal"
)
```

### **Critical Systems**
```python
critical_config = EntityRegistrationConfig(
    creation_strategy=EntityCreationStrategy.IMMEDIATE_REGISTRATION,
    registration_mode=RegistrationMode.STABLE_ID,
    force_registration=True,
    validation_level="strict"
)
```

---

## 🛠️ **Integration with CallableRegistry**

```python
@CallableRegistry.register("create_entity_with_factory")
def create_entity_with_factory_callable(
    entity_type: str,
    validation_domain: str = "default",
    auto_register: bool = True
) -> str:
    """Create entities via CallableRegistry integration."""
    # Implementation returns entity ID string
```

---

## 🚨 **Error Handling**

### **Validation Errors**
- Domain validation failures raise `ValueError` with specific context
- Framework conflicts are detected and gracefully skipped
- Validation results are stored for debugging

### **Registration Failures**
- `force_registration=True`: Raises exception on failure
- `force_registration=False`: Logs warning and continues
- Fallback mechanisms attempt different registration strategies

---

## 📋 **Best Practices**

### **For Claude/LLM Integration**
1. Always use `EntityFactory.create_root_entity()`
2. Use `STABLE_ID` registration mode
3. Include `llm_creation_context` for debugging
4. Store entity IDs for later retrieval

### **For Performance**
1. Use `NO_REGISTRATION` mode for batch operations
2. Set `validation_level="minimal"` for trusted input
3. Disable events and tracking for high-volume scenarios

### **For Production Systems**
1. Use `validation_level="strict"` for critical entities
2. Enable `force_registration` for required entities
3. Monitor creation patterns with analytics
4. Implement real-time health monitoring

---

## 🔗 **Quick Reference**

### **Entity Creation Pipeline**
1. **Create Instance**: Instantiate entity class
2. **Validate**: Run intelligent validation (if enabled)
3. **Track Provenance**: Add creation metadata (if enabled)
4. **Register**: Handle registration based on mode
5. **Emit Events**: Notify system (if enabled)
6. **Record History**: Add to analytics (always)

### **Key Methods**
- `EntityFactory.create_root_entity()` - Primary method for Claude
- `EntityFactory.create_entity()` - Full control method
- `EntityFactory.get_creation_stats()` - Analytics
- `validate_entity_for_registration()` - Pre-registration validation

### **Configuration Priorities**
1. **Stability**: Use STABLE_ID mode for external integrations
2. **Validation**: Use domain-specific validation methods
3. **Performance**: Disable features for high-volume scenarios
4. **Debugging**: Enable full tracking for development

---

This guide provides the essential information for using the EntityFactory system effectively while maintaining the universal design pattern that works across all domains. 