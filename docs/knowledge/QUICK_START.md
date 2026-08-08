# Manual Graph Editor - Quick Start Guide

## Installation & Setup

```python
from src.core.knowledge.services.manual_graph_service import ManualGraphService
from src.core.knowledge.models.graph import ConceptGraph

# Create a new service
service = ManualGraphService(ConceptGraph())
```

## 5-Minute Tutorial

### 1. Create Concepts

```python
# Create basic concepts
service.create_concept("Python")
service.create_concept("JavaScript")
service.create_concept("Web Development")

# Create with metadata
service.create_concept(
    "Machine Learning",
    aliases=["ML", "Statistical Learning"],
    definitions=["Field of AI that learns from data"]
)
```

### 2. Add Relationships

```python
from src.core.knowledge.models.manual_graph_models import RelationshipType

# Create relationships
service.add_relationship(
    "Python",
    "Machine Learning",
    relationship_type=RelationshipType.PREREQUISITE.value,
    strength=0.9
)

service.add_relationship(
    "Python",
    "Web Development",
    relationship_type=RelationshipType.RELATED.value
)
```

### 3. Query the Graph

```python
# Get concept information
info = service.get_concept_info("Python")
print(f"Concept: {info['name']}")
print(f"Relationships: {info['outgoing_relationships']}")

# Get graph statistics
stats = service.get_graph_stats()
print(f"Nodes: {stats['node_count']}")
print(f"Is DAG: {stats['is_dag']}")
```

### 4. Validate

```python
# Check for issues
errors = service.validate_graph()
print(f"Found {len(errors)} issues")

# Get suggestions
suggestions = service.get_autofix_suggestions()
```

### 5. Backup & Export

```python
# Create backup
backup_id = service.create_backup("Stable version")

# Export to JSON
import json
export_data = service.export_graph()

with open("my_graph.json", "w") as f:
    json.dump(export_data, f)
```

## Common Tasks

### Task: Detect Prerequisites

```python
# Check what prerequisites lead to a concept
def get_prerequisites(service, concept_name):
    info = service.get_concept_info(concept_name)
    return info['predecessors']

prereqs = get_prerequisites(service, "Machine Learning")
print(f"Prerequisites for ML: {prereqs}")
```

### Task: Find Orphaned Concepts

```python
# Concepts with no relationships
orphaned = service._find_orphaned_concepts()
print(f"Orphaned concepts: {orphaned}")

# Remove them
for concept in orphaned:
    service.delete_concept(concept)
```

### Task: Merge Two Graphs

```python
# Import a different graph
with open("other_graph.json", "r") as f:
    other_graph = json.load(f)

service.import_graph(other_graph, merge=True)
```

### Task: Rename Concepts

```python
# Simple rename
service.update_concept("OldName", name="NewName")

# Rename with other updates
service.update_concept(
    "OldName",
    name="NewName",
    aliases=["Old", "OldAlias"],
    definitions=["New definition"]
)
```

### Task: Break Circular Dependencies

```python
# Find cycles
cycles = service.detect_circular_prerequisites()

# Break them
for cycle in cycles:
    # Remove last edge in cycle
    if len(cycle) >= 2:
        service.remove_relationship(cycle[-1], cycle[0])
```

## Relationship Types Quick Reference

| Type | Use Case |
|------|----------|
| **PREREQUISITE** | Concept A must be learned before B |
| **HIERARCHICAL** | Concept A is more specific than B |
| **RELATED** | Concepts are related but not directly dependent |
| **SIMILAR** | Concepts are similar/interchangeable |
| **DEPENDS_ON** | Concept A depends on B |
| **PART_OF** | Concept A is part of concept B |

## Error Handling

```python
from src.core.knowledge.exceptions import (
    ConceptNotFoundError,
    RelationshipNotFoundError,
    CycleDetectedError,
    GraphError
)

try:
    service.add_relationship("A", "B", relationship_type="prerequisite")
except ConceptNotFoundError:
    print("One of the concepts doesn't exist")
except CycleDetectedError:
    print("Would create a circular dependency")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Performance Tips

1. **Use backups before major changes**
   ```python
   backup_id = service.create_backup("Before refactoring")
   # ... make changes ...
   if problem_detected:
       service.restore_backup(backup_id)
   ```

2. **Validate after large imports**
   ```python
   service.import_graph(large_graph)
   errors = service.validate_graph()
   ```

3. **Export regularly**
   ```python
   export_data = service.export_graph()
   # Save to version control
   ```

## Troubleshooting

### Issue: "Concept already exists"
```python
# Check if concept exists first
if service.graph.has_concept("Python"):
    print("Already exists")
else:
    service.create_concept("Python")
```

### Issue: "Would create circular dependency"
```python
# Check before adding prerequisite
if not service._would_create_cycle(from_concept, to_concept):
    service.add_relationship(from_concept, to_concept,
        relationship_type=RelationshipType.PREREQUISITE.value)
```

### Issue: "Cannot delete concept with relationships"
```python
# Use force=True
service.delete_concept("ConceptName", force=True)
```

## Integration with Doc Explainer

```python
from src.store.knowledge.graph_repository import KnowledgeRepository

# Get existing graph
repo = KnowledgeRepository()
graph = repo.get_concept_graph()

# Create service with existing graph
service = ManualGraphService(graph)

# Make modifications
service.create_concept("New Concept")

# Save back
repo.update_graph(service.graph)
```

## Next Steps

- Read the full [Manual Graph Editor documentation](MANUAL_GRAPH_EDITOR.md)
- Check the [API Reference](MANUAL_GRAPH_API.md)
- Review test examples in `test_manual_graph_service.py`
