# Manual Concept Graph Editor

## Overview

The Manual Concept Graph Editor is a comprehensive service for editing and managing concept graphs in the Doc Explainer system. It provides tools for creating, updating, and validating concepts and relationships, with support for backup/restore and import/export functionality.

## Features

### 1. Concept Management

#### Create Concepts
```python
from src.core.knowledge.services.manual_graph_service import ManualGraphService
from src.core.knowledge.models.graph import ConceptGraph

service = ManualGraphService(ConceptGraph())

# Create a basic concept
concept_id = service.create_concept("Machine Learning")

# Create with aliases and definitions
concept_id = service.create_concept(
    "Machine Learning",
    aliases=["ML", "Learning Systems"],
    definitions=["Field of AI focused on learning from data"],
    attributes={"difficulty": "advanced"}
)
```

#### Update Concepts
```python
# Update concept name
service.update_concept("ML", name="Machine Learning")

# Update aliases
service.update_concept("Machine Learning", aliases=["ML", "Statistical Learning"])

# Update definitions
service.update_concept(
    "Machine Learning",
    definitions=["Field of AI", "Data-driven learning"]
)
```

#### Manage Aliases
```python
# Add an alias
service.add_alias("Machine Learning", "Statistical Learning")

# Remove an alias
service.remove_alias("Machine Learning", "Statistical Learning")
```

#### Delete Concepts
```python
# Delete orphaned concept
service.delete_concept("Unused Concept")

# Delete concept with relationships (force)
service.delete_concept("Core Concept", force=True)
```

### 2. Relationship Management

#### Add Relationships
```python
from src.core.knowledge.models.manual_graph_models import RelationshipType

# Basic relationship
service.add_relationship(
    "Machine Learning",
    "Neural Networks"
)

# With specific type and strength
service.add_relationship(
    "Machine Learning",
    "Neural Networks",
    relationship_type=RelationshipType.HIERARCHICAL.value,
    strength=0.9,
    definition="Neural Networks are a technique in ML"
)
```

#### Relationship Types
- **PREREQUISITE**: Concept must be learned before another
- **SIMILAR**: Concepts are similar
- **RELATED**: Concepts are related but not strongly
- **HIERARCHICAL**: One concept is more specific than another
- **DEPENDS_ON**: Concept depends on another
- **PART_OF**: Concept is part of another

#### Update Relationships
```python
service.update_relationship(
    "Machine Learning",
    "Neural Networks",
    relationship_type=RelationshipType.PREREQUISITE.value,
    strength=1.0
)
```

#### Remove Relationships
```python
service.remove_relationship("Machine Learning", "Neural Networks")
```

### 3. Graph Validation

#### Validate Graph
```python
# Get all validation errors
errors = service.validate_graph()

for error in errors:
    print(f"Error: {error.error_type}")
    print(f"Message: {error.message}")
    print(f"Suggestion: {error.suggestion}")
```

#### Circular Dependency Detection
```python
# Detect cycles in prerequisites
cycles = service.detect_circular_prerequisites()

for cycle in cycles:
    print(f"Cycle found: {' -> '.join(cycle)}")
```

#### Concept Name Validation
```python
is_valid, error_msg = service.validate_concept_name("Machine Learning")

if not is_valid:
    print(f"Invalid name: {error_msg}")
```

#### Auto-fix Suggestions
```python
suggestions = service.get_autofix_suggestions()

for suggestion in suggestions:
    print(f"Type: {suggestion['type']}")
    print(f"Action: {suggestion['action']}")
```

### 4. Export/Import

#### Export Graph
```python
# Export to JSON-compatible dictionary
graph_data = service.export_graph()

# Save to file
import json
with open("graph.json", "w") as f:
    json.dump(graph_data, f, indent=2)

# Export structure includes:
# - nodes: List of concepts with properties
# - edges: List of relationships
# - stats: Graph statistics
```

#### Import Graph
```python
# Load from file
with open("graph.json", "r") as f:
    graph_data = json.load(f)

# Import (replace existing)
service.import_graph(graph_data, merge=False)

# Or merge with existing graph
service.import_graph(graph_data, merge=True)
```

### 5. Backup & Restore

#### Create Backups
```python
# Create backup with description
backup_id = service.create_backup(
    description="Before major refactoring",
    tags=["stable", "v1.0"]
)

print(f"Backup created: {backup_id}")
```

#### List Backups
```python
backups = service.list_backups()

for backup in backups:
    print(f"ID: {backup['backup_id']}")
    print(f"Timestamp: {backup['timestamp']}")
    print(f"Description: {backup['description']}")
```

#### Restore from Backup
```python
service.restore_backup(backup_id)
```

#### Delete Backups
```python
service.delete_backup(backup_id)
```

### 6. Query & Inspection

#### Get Concept Information
```python
info = service.get_concept_info("Machine Learning")

# Returns:
# {
#     "id": "concept_id",
#     "name": "Machine Learning",
#     "aliases": [...],
#     "definitions": [...],
#     "score": 0.0,
#     "frequency": 0,
#     "attributes": {},
#     "incoming_relationships": 5,
#     "outgoing_relationships": 3,
#     "predecessors": [...],
#     "successors": [...]
# }
```

#### Graph Statistics
```python
stats = service.get_graph_stats()

# Returns:
# {
#     "node_count": 150,
#     "edge_count": 300,
#     "density": 0.027,
#     "is_dag": True,
#     "orphaned_count": 2,
#     "cycle_count": 0
# }
```

#### Edit History
```python
history = service.get_edit_history(limit=50)

for edit in history:
    print(f"Operation: {edit['type']}")
    print(f"Timestamp: {edit['timestamp']}")
```

## Data Models

### ConceptEdit
Represents an edit operation on a concept:
```python
@dataclass
class ConceptEdit:
    operation: OperationType
    concept_id: str
    concept_name: Optional[str]
    changes: Dict[str, Any]
    timestamp: Optional[datetime]
    previous_state: Optional[Dict[str, Any]]
```

### RelationshipEdit
Represents an edit operation on a relationship:
```python
@dataclass
class RelationshipEdit:
    operation: OperationType
    from_concept_id: str
    to_concept_id: str
    relationship_type: RelationshipType
    strength: float
    metadata: Dict[str, Any]
    timestamp: Optional[datetime]
```

### GraphSnapshot
Represents a snapshot of the graph:
```python
@dataclass
class GraphSnapshot:
    timestamp: Optional[datetime]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    checksum: Optional[str]
    version: str
    metadata: Dict[str, Any]
```

### ValidationError
Represents a validation error:
```python
@dataclass
class ValidationError:
    error_type: str
    message: str
    affected_concepts: List[str]
    affected_relationships: List[tuple]
    severity: str  # "error", "warning", "info"
    suggestion: Optional[str]
```

## Error Handling

The service raises specific exceptions for different error conditions:

```python
from src.core.knowledge.exceptions import (
    ConceptNotFoundError,
    RelationshipNotFoundError,
    CycleDetectedError,
    GraphError
)

try:
    service.add_relationship("A", "B")
except ConceptNotFoundError:
    print("Concept not found")
except CycleDetectedError:
    print("Would create circular dependency")
except GraphError:
    print("General graph operation error")
```

## Validation Rules

### Concept Names
- Cannot be empty or whitespace-only
- Must be unique in the graph
- Maximum 200 characters
- Must be a string

### Relationship Strength
- Must be between 0.0 and 1.0 (inclusive)
- Represents confidence or importance

### Relationship Types
- Must be one of the valid types
- Prerequisite relationships are checked for cycles
- Non-prerequisite relationships can form cycles

### Graph Consistency
- All relationships must connect existing concepts
- Duplicate aliases are detected and warned
- Orphaned concepts are identified

## Workflow Examples

### Example 1: Build a Learning Path
```python
service = ManualGraphService(ConceptGraph())

# Create concepts
concepts = [
    "Linear Algebra",
    "Calculus",
    "Probability",
    "Statistics",
    "Machine Learning"
]

for concept in concepts:
    service.create_concept(concept)

# Add prerequisites
service.add_relationship(
    "Linear Algebra", "Machine Learning",
    relationship_type=RelationshipType.PREREQUISITE.value
)
service.add_relationship(
    "Calculus", "Machine Learning",
    relationship_type=RelationshipType.PREREQUISITE.value
)
service.add_relationship(
    "Probability", "Statistics",
    relationship_type=RelationshipType.PREREQUISITE.value
)

# Validate
errors = service.validate_graph()
print(f"Found {len(errors)} validation issues")
```

### Example 2: Import Existing Knowledge Base
```python
service = ManualGraphService(ConceptGraph())

# Load existing graph
with open("knowledge_base.json", "r") as f:
    kb_data = json.load(f)

service.import_graph(kb_data)

# Validate imported data
errors = service.validate_graph()

# Create backup before modifications
backup_id = service.create_backup("Imported KB")

# Make improvements
for error in errors:
    if error.error_type == "orphaned_concepts":
        for concept in error.affected_concepts:
            service.delete_concept(concept)

# Export cleaned data
cleaned_data = service.export_graph()
```

### Example 3: Detect and Fix Circular Dependencies
```python
service = ManualGraphService(ConceptGraph())

# ... build graph ...

# Check for cycles
cycles = service.detect_circular_prerequisites()

if cycles:
    print(f"Found {len(cycles)} circular dependencies:")
    
    for cycle in cycles:
        print(f"  Cycle: {' -> '.join(cycle)}")
        
        # Remove last edge to break cycle
        if len(cycle) >= 2:
            service.remove_relationship(cycle[-1], cycle[0])

# Validate
errors = service.validate_graph()
is_valid = all(e.severity != "error" for e in errors)
print(f"Graph is valid: {is_valid}")
```

## Performance Considerations

- **Cycle Detection**: O(V + E) using networkx
- **Export**: O(V + E) to traverse and serialize
- **Import**: O(V + E) to add nodes and edges
- **Validation**: O(V + E) for various checks

For large graphs (>10,000 nodes), consider:
- Batching operations
- Using backups before major changes
- Running validation asynchronously

## Integration with Doc Explainer

The service integrates with existing components:

```python
# Use with graph repository
from src.store.knowledge.graph_repository import KnowledgeRepository

repo = KnowledgeRepository()
graph = repo.get_concept_graph()

service = ManualGraphService(graph)

# Make modifications
service.create_concept("New Concept")

# Save back to repository
repo.update_graph(service.graph)
```

## Best Practices

1. **Always create backups before major changes**
   ```python
   backup_id = service.create_backup("Before refactoring")
   ```

2. **Validate after modifications**
   ```python
   errors = service.validate_graph()
   ```

3. **Use prerequisites carefully**
   ```python
   # Check for cycles before adding prerequisite
   if not service._would_create_cycle(from_concept, to_concept):
       service.add_relationship(from_concept, to_concept,
           relationship_type=RelationshipType.PREREQUISITE.value)
   ```

4. **Export regularly for version control**
   ```python
   export_data = service.export_graph()
   # Save to version control
   ```

5. **Handle exceptions appropriately**
   ```python
   try:
       service.create_concept("ML")
   except ValueError as e:
       print(f"Invalid input: {e}")
   except ConceptNotFoundError as e:
       print(f"Concept not found: {e}")
   ```

## Testing

Comprehensive test suite provided in `test_manual_graph_service.py`:
- CRUD operations for concepts
- Relationship management
- Cycle detection
- Validation checks
- Export/import roundtrips
- Backup/restore functionality
- Query methods

Run tests with:
```bash
pytest src/core/knowledge/tests/test_manual_graph_service.py -v
```
