# EntityFactory System: Actual Implementation Guide

**IMPORTANT CLARIFICATION**: The "3-tier validation system" is actually a **2-tier registration system** with **2 distinct methods**. Method 1 (stable ID registration) is used for 3 different validation scenarios, but they all use the same registration approach. This system is designed for your custom code only.

---

## 🎯 **The Real System: 2 Registration Methods**

### **Method 1: Stable ID Registration** (Your Custom Code)
- **Used for**: `validate_domain_state()`, `validate_operational_state()`, `validate_workflow_state()`
- **Registration**: Manual root assignment + direct EntityRegistry registration
- **Result**: Entity ID never changes - perfect for LLM integration

### **Method 2: Versioning Registration** (Framework Fallback)
- **Used for**: Generic `validate()` method (framework entities)
- **Registration**: Traditional `promote_to_root()` 
- **Result**: Entity ID may change for versioning

---

## 🏗️ **How It Actually Works**

```python
def _validate_entity(cls, entity: Entity, validation_level: str) -> None:
    # METHOD 1: Stable ID Registration (3 validation scenarios)
    
    # Scenario A: Domain validation
    if hasattr(entity, 'validate_domain_state'):
        entity.validate_domain_state()
        # → Uses stable ID registration
    
    # Scenario B: Operational validation  
    elif hasattr(entity, 'validate_operational_state'):
        entity.validate_operational_state()
        # → Uses stable ID registration
    
    # Scenario C: Workflow validation
    elif hasattr(entity, 'validate_workflow_state'):
        entity.validate_workflow_state()
        # → Uses stable ID registration
    
    # METHOD 2: Versioning Registration (framework fallback)
    elif hasattr(entity, 'validate') and callable(getattr(entity, 'validate')):
        # Generic framework validation
        entity.validate()  # → Uses promote_to_root() registration
```

## 📋 **Registration Implementation**

### **Method 1: Stable ID (Your Code)**
```python
# All 3 validation scenarios use this same registration approach
def _handle_registration(entity, config):
    if config.registration_mode == RegistrationMode.STABLE_ID:
        # Manual root assignment preserves original ID
        entity.root_ecs_id = entity.ecs_id
        entity.root_live_id = entity.live_id
        
        # Direct registration without ID changes
        EntityRegistry.register_entity(entity)
        # Result: ID remains stable forever
```

### **Method 2: Versioning (Framework Fallback)**
```python
# Traditional ECS registration for framework entities
def _handle_registration(entity, config):
    if config.registration_mode == RegistrationMode.VERSIONED:
        # Framework-managed promotion may change ID
        entity.promote_to_root()
        # Result: ID may change for versioning purposes
```

---

## 🔧 **Usage Patterns**

### **For Your Custom Entities (Method 1)**
```python
# Domain entities
class BusinessEntity(Entity):
    def validate_domain_state(self) -> bool:
        return self.business_logic_valid()

# Operational entities
class SystemEntity(Entity):
    def validate_operational_state(self) -> bool:
        return self.system_requirements_met()

# Workflow entities  
class ProcessEntity(Entity):
    def validate_workflow_state(self) -> bool:
        return self.process_steps_valid()

# All use stable ID registration automatically
entity = EntityFactory.create_root_entity(BusinessEntity, **kwargs)
# ID never changes - perfect for LLM tracking
```

### **For Framework Entities (Method 2)**
```python
# Generic framework entities
class FrameworkEntity(Entity):
    def validate(self) -> bool:  # Generic validation
        return super().validate()

# Uses versioning registration
entity = EntityFactory.create_root_entity(FrameworkEntity, **kwargs)
# ID may change during framework operations
```

---

## 📊 **The Real Truth: Method 1 Has 3 Entry Points**

| Validation Method | Registration Used | ID Behavior | Purpose |
|------------------|------------------|-------------|---------|
| `validate_domain_state()` | Method 1 (Stable) | Never Changes | Your business logic entities |
| `validate_operational_state()` | Method 1 (Stable) | Never Changes | Your system entities |
| `validate_workflow_state()` | Method 1 (Stable) | Never Changes | Your process entities |
| `validate()` (generic) | Method 2 (Versioning) | May Change | Framework fallback |

## 🎯 **Key Insight**

**It's not really a 3-tier system** - it's a **2-method system**:

1. **Your Code Method**: 3 different validation entry points, all using stable ID registration
2. **Framework Method**: 1 generic validation entry point, using versioning registration

The "3 tiers" are just **3 different ways to call the same stable registration method**.

---

## 💡 **Why This Design?**

### **Method 1 (Stable ID) Benefits**
- **LLM Integration**: AI can reliably store and reuse entity IDs
- **External Systems**: APIs and databases get consistent references
- **User Experience**: Predictable entity behavior
- **Debugging**: Entity traces remain consistent

### **Method 2 (Versioning) Benefits**
- **Framework Compatibility**: Works with existing ECS patterns
- **Change Tracking**: Entity evolution is properly tracked
- **Lineage Preservation**: Historical versions maintained
- **Graceful Fallback**: Handles unknown entity types

---

## 🔧 **Implementation Details**

### **Stable ID Registration Flow**
```python
# Step 1: Create entity
entity = YourEntityClass(**kwargs)

# Step 2: Validate using your custom method
if hasattr(entity, 'validate_domain_state'):
    entity.validate_domain_state()

# Step 3: Stable registration
entity.root_ecs_id = entity.ecs_id  # Manual assignment
entity.root_live_id = entity.live_id
EntityRegistry.register_entity(entity)  # Direct registration

# Result: entity.ecs_id never changes
```

### **Versioning Registration Flow**
```python
# Step 1: Create entity
entity = FrameworkEntityClass(**kwargs)

# Step 2: Generic validation
entity.validate()  # Framework method

# Step 3: Versioning registration
entity.promote_to_root()  # May change entity.ecs_id

# Result: entity.ecs_id may change for versioning
```

---

## 📋 **Quick Reference**

### **For Your Custom Code (Always Stable IDs)**
```python
# Create any of your custom entities
entity = EntityFactory.create_root_entity(YourEntity, **kwargs)

# ID is guaranteed stable
original_id = entity.ecs_id
# ... time passes, operations happen ...
assert entity.ecs_id == original_id  # Always true
```

### **Entity Types and Their Registration**
- **Domain entities** (`validate_domain_state`) → Stable ID
- **Operational entities** (`validate_operational_state`) → Stable ID  
- **Workflow entities** (`validate_workflow_state`) → Stable ID
- **Framework entities** (`validate`) → Versioning

### **Configuration Options**
```python
# Force stable ID for all entities
config = EntityRegistrationConfig(
    registration_mode=RegistrationMode.STABLE_ID  # Your code default
)

# Allow versioning for framework compatibility
config = EntityRegistrationConfig(
    registration_mode=RegistrationMode.VERSIONED  # Framework fallback
)
```

---

## ✅ **Summary**

The EntityFactory implements a **2-method registration system**:

1. **Method 1 (Stable)**: For your custom entities with predictable IDs
   - 3 validation entry points: domain, operational, workflow
   - All use manual root assignment
   - Perfect for LLM integration

2. **Method 2 (Versioning)**: For framework compatibility  
   - 1 generic validation entry point
   - Uses traditional promote_to_root()
   - Maintains ECS compatibility

**Bottom Line**: Your custom code gets stable IDs, framework code gets versioning. The "3-tier" terminology refers to 3 ways to access the same stable registration method. 