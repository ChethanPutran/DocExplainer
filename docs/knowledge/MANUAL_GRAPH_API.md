# Manual Graph Service API Reference

## Class: ManualGraphService

### Constructor

```python
def __init__(self, graph: Optional[ConceptGraph] = None)
```

**Parameters:**
- `graph` (Optional[ConceptGraph]): Existing graph to use. If None, creates new graph.

**Example:**
```python
from src.core.knowledge.services.manual_graph_service import ManualGraphService
from src.core.knowledge.models.graph import ConceptGraph

service = ManualGraphService(ConceptGraph())
```

---

## Concept Operations

### create_concept()

```python
def create_concept(
    name: str,
    aliases: Optional[List[str]] = None,
    definitions: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> str
```

**Description:** Create a new concept and add it to the graph.

**Parameters:**
- `name` (str): Concept name (required, unique, non-empty)
- `aliases` (Optional[List[str]]): Alternative names for the concept
- `definitions` (Optional[List[str]]): Definitions of the concept
- `attributes` (Optional[Dict[str, Any]]): Additional attributes

**Returns:** Concept ID (str)

**Raises:**
- `ValueError`: If name is invalid or already exists

**Example:**
```python
concept_id = service.create_concept(
    "Python",
    aliases=["Python 3", "Python Language"],
    definitions=["General-purpose programming language"],
    attributes={"year_created": 1991}
)
```

---

### update_concept()

```python
def update_concept(
    concept_name: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    definitions: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> str
```

**Description:** Update an existing concept.

**Parameters:**
- `concept_name` (str): Current concept name
- `name` (Optional[str]): New name
- `aliases` (Optional[List[str]]): New aliases
- `definitions` (Optional[List[str]]): New definitions
- `attributes` (Optional[Dict[str, Any]]): New attributes

**Returns:** Updated concept ID

**Raises:**
- `ConceptNotFoundError`: If concept not found
- `ValueError`: If new name conflicts with existing concept

**Example:**
```python
service.update_concept(
    "Python",
    aliases=["Python 3", "Python Programming"]
)
```

---

### add_alias()

```python
def add_alias(concept_name: str, alias: str) -> None
```

**Description:** Add an alias to a concept.

**Parameters:**
- `concept_name` (str): Concept name
- `alias` (str): New alias to add

**Raises:**
- `ConceptNotFoundError`: If concept not found
- `ValueError`: If alias is invalid or already exists

**Example:**
```python
service.add_alias("Python", "Python Programming Language")
```

---

### remove_alias()

```python
def remove_alias(concept_name: str, alias: str) -> None
```

**Description:** Remove an alias from a concept.

**Parameters:**
- `concept_name` (str): Concept name
- `alias` (str): Alias to remove

**Raises:**
- `ConceptNotFoundError`: If concept not found
- `ValueError`: If alias not found

**Example:**
```python
service.remove_alias("Python", "Python 3")
```

---

### delete_concept()

```python
def delete_concept(concept_name: str, force: bool = False) -> None
```

**Description:** Delete a concept from the graph.

**Parameters:**
- `concept_name` (str): Concept name
- `force` (bool): If True, remove all relationships; otherwise raise error

**Raises:**
- `ConceptNotFoundError`: If concept not found
- `GraphError`: If concept has relationships and force=False

**Example:**
```python
# Delete orphaned concept
service.delete_concept("UnusedConcept")

# Delete concept with relationships
service.delete_concept("CoreConcept", force=True)
```

---

## Relationship Operations

### add_relationship()

```python
def add_relationship(
    from_concept_name: str,
    to_concept_name: str,
    relationship_type: str = RelationshipType.RELATED.value,
    strength: float = 1.0,
    definition: str = ""
) -> None
```

**Description:** Add a relationship between two concepts.

**Parameters:**
- `from_concept_name` (str): Source concept name
- `to_concept_name` (str): Target concept name
- `relationship_type` (str): Type of relationship (prerequisite, similar, related, hierarchical, depends_on, part_of)
- `strength` (float): Relationship strength (0.0 to 1.0)
- `definition` (str): Definition of the relationship

**Raises:**
- `ConceptNotFoundError`: If either concept not found
- `ValueError`: If relationship type invalid or strength out of range
- `CycleDetectedError`: If would create circular prerequisite

**Example:**
```python
service.add_relationship(
    "Python",
    "Functions",
    relationship_type="prerequisite",
    strength=0.9,
    definition="Functions are a core concept in Python"
)
```

---

### update_relationship()

```python
def update_relationship(
    from_concept_name: str,
    to_concept_name: str,
    relationship_type: Optional[str] = None,
    strength: Optional[float] = None,
    definition: Optional[str] = None
) -> None
```

**Description:** Update a relationship between two concepts.

**Parameters:**
- `from_concept_name` (str): Source concept name
- `to_concept_name` (str): Target concept name
- `relationship_type` (Optional[str]): New relationship type
- `strength` (Optional[float]): New strength
- `definition` (Optional[str]): New definition

**Raises:**
- `RelationshipNotFoundError`: If relationship not found
- `ValueError`: If invalid parameters

**Example:**
```python
service.update_relationship(
    "Python",
    "Functions",
    strength=0.95
)
```

---

### remove_relationship()

```python
def remove_relationship(
    from_concept_name: str,
    to_concept_name: str
) -> None
```

**Description:** Remove a relationship between two concepts.

**Parameters:**
- `from_concept_name` (str): Source concept name
- `to_concept_name` (str): Target concept name

**Raises:**
- `ConceptNotFoundError`: If either concept not found
- `RelationshipNotFoundError`: If relationship not found

**Example:**
```python
service.remove_relationship("Python", "Functions")
```

---

## Validation Methods

### validate_graph()

```python
def validate_graph() -> List[ValidationError]
```

**Description:** Validate the graph for consistency issues.

**Returns:** List of ValidationError objects

**Example:**
```python
errors = service.validate_graph()
for error in errors:
    print(f"{error.error_type}: {error.message}")
    if error.suggestion:
        print(f"Suggestion: {error.suggestion}")
```

---

### detect_circular_prerequisites()

```python
def detect_circular_prerequisites() -> List[List[str]]
```

**Description:** Detect circular prerequisite chains.

**Returns:** List of cycles (each cycle is a list of concept names)

**Example:**
```python
cycles = service.detect_circular_prerequisites()
for cycle in cycles:
    print(f"Cycle: {' -> '.join(cycle)}")
```

---

### validate_concept_name()

```python
def validate_concept_name(name: str) -> Tuple[bool, str]
```

**Description:** Validate a concept name.

**Parameters:**
- `name` (str): Concept name to validate

**Returns:** Tuple of (is_valid: bool, error_message: str)

**Example:**
```python
is_valid, error_msg = service.validate_concept_name("Machine Learning")
if not is_valid:
    print(f"Invalid: {error_msg}")
```

---

### validate_relationship_coherence()

```python
def validate_relationship_coherence() -> List[str]
```

**Description:** Check relationship coherence (relationships between valid concepts).

**Returns:** List of issue messages

**Example:**
```python
issues = service.validate_relationship_coherence()
```

---

### get_autofix_suggestions()

```python
def get_autofix_suggestions() -> List[Dict[str, Any]]
```

**Description:** Get suggestions for auto-fixing validation issues.

**Returns:** List of suggestion dictionaries

**Example:**
```python
suggestions = service.get_autofix_suggestions()
for suggestion in suggestions:
    print(f"Type: {suggestion['type']}")
    print(f"Action: {suggestion['action']}")
```

---

## Export/Import Methods

### export_graph()

```python
def export_graph() -> Dict[str, Any]
```

**Description:** Export graph to JSON-compatible dictionary.

**Returns:** Dictionary with nodes, edges, and stats

**Structure:**
```json
{
    "version": "1.0",
    "exported_at": "2024-05-15T10:30:00.123456",
    "nodes": [
        {
            "id": "concept_id",
            "name": "Concept Name",
            "aliases": ["alias1", "alias2"],
            "definitions": ["definition"],
            "score": 0.5,
            "frequency": 3,
            "attributes": {}
        }
    ],
    "edges": [
        {
            "from": "Concept1",
            "to": "Concept2",
            "type": "prerequisite",
            "strength": 0.9,
            "definition": "relationship definition",
            "attributes": {}
        }
    ],
    "stats": {
        "node_count": 150,
        "edge_count": 300
    }
}
```

**Example:**
```python
export_data = service.export_graph()
with open("graph.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

---

### import_graph()

```python
def import_graph(data: Dict[str, Any], merge: bool = False) -> None
```

**Description:** Import graph from JSON-compatible dictionary.

**Parameters:**
- `data` (Dict[str, Any]): Dictionary with nodes and edges
- `merge` (bool): If True, merge with existing graph; if False, replace

**Raises:**
- `ValueError`: If data format invalid

**Example:**
```python
with open("graph.json", "r") as f:
    graph_data = json.load(f)

service.import_graph(graph_data, merge=False)
```

---

## Backup & Restore Methods

### create_backup()

```python
def create_backup(
    description: str = "",
    tags: Optional[List[str]] = None
) -> str
```

**Description:** Create a backup of the current graph.

**Parameters:**
- `description` (str): Description of the backup
- `tags` (Optional[List[str]]): Tags for organizing backups

**Returns:** Backup ID

**Example:**
```python
backup_id = service.create_backup(
    description="Before major refactoring",
    tags=["stable", "v1.0"]
)
```

---

### restore_backup()

```python
def restore_backup(backup_id: str) -> None
```

**Description:** Restore graph from a backup.

**Parameters:**
- `backup_id` (str): ID of backup to restore

**Raises:**
- `ValueError`: If backup not found

**Example:**
```python
service.restore_backup(backup_id)
```

---

### list_backups()

```python
def list_backups() -> List[Dict[str, Any]]
```

**Description:** Get list of available backups.

**Returns:** List of backup info dictionaries

**Example:**
```python
backups = service.list_backups()
for backup in backups:
    print(f"ID: {backup['backup_id']}")
    print(f"Description: {backup['description']}")
```

---

### delete_backup()

```python
def delete_backup(backup_id: str) -> None
```

**Description:** Delete a backup.

**Parameters:**
- `backup_id` (str): ID of backup to delete

**Raises:**
- `ValueError`: If backup not found

**Example:**
```python
service.delete_backup(backup_id)
```

---

## Query Methods

### get_concept_info()

```python
def get_concept_info(concept_name: str) -> Optional[Dict[str, Any]]
```

**Description:** Get detailed information about a concept.

**Parameters:**
- `concept_name` (str): Concept name

**Returns:** Dictionary with concept info or None if not found

**Structure:**
```python
{
    "id": "concept_id",
    "name": "Concept Name",
    "aliases": ["alias1"],
    "definitions": ["definition"],
    "score": 0.5,
    "frequency": 3,
    "attributes": {},
    "incoming_relationships": 5,
    "outgoing_relationships": 3,
    "predecessors": ["Prereq1", "Prereq2"],
    "successors": ["Succ1"]
}
```

**Example:**
```python
info = service.get_concept_info("Machine Learning")
```

---

### get_graph_stats()

```python
def get_graph_stats() -> Dict[str, Any]
```

**Description:** Get statistics about the graph.

**Returns:** Dictionary with graph statistics

**Structure:**
```python
{
    "node_count": 150,
    "edge_count": 300,
    "density": 0.027,
    "is_dag": True,
    "orphaned_count": 2,
    "cycle_count": 0
}
```

**Example:**
```python
stats = service.get_graph_stats()
print(f"Nodes: {stats['node_count']}")
print(f"Is DAG: {stats['is_dag']}")
```

---

### get_edit_history()

```python
def get_edit_history(limit: int = 100) -> List[Dict[str, Any]]
```

**Description:** Get edit history.

**Parameters:**
- `limit` (int): Maximum number of entries to return

**Returns:** List of edit history entries

**Example:**
```python
history = service.get_edit_history(limit=50)
```

---

## Snapshot Methods

### create_snapshot()

```python
def create_snapshot(description: str = "") -> GraphSnapshot
```

**Description:** Create a snapshot of the graph.

**Parameters:**
- `description` (str): Snapshot description

**Returns:** GraphSnapshot object

**Example:**
```python
snapshot = service.create_snapshot("Stable version")
```

---

## Constants

### RelationshipType (Enum)

```python
class RelationshipType(str, Enum):
    PREREQUISITE = "prerequisite"
    SIMILAR = "similar"
    RELATED = "related"
    HIERARCHICAL = "hierarchical"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
```

### OperationType (Enum)

```python
class OperationType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ADD_ALIAS = "add_alias"
    REMOVE_ALIAS = "remove_alias"
    ADD_DEFINITION = "add_definition"
    REMOVE_DEFINITION = "remove_definition"
```

---

## Exception Classes

### ConceptNotFoundError

Raised when a concept is not found.

```python
try:
    service.update_concept("NonExistent")
except ConceptNotFoundError as e:
    print(f"Concept not found: {e}")
```

### RelationshipNotFoundError

Raised when a relationship is not found.

```python
try:
    service.remove_relationship("A", "B")
except RelationshipNotFoundError as e:
    print(f"Relationship not found: {e}")
```

### CycleDetectedError

Raised when adding a relationship would create a cycle in prerequisites.

```python
try:
    service.add_relationship("A", "B", 
        relationship_type="prerequisite")
except CycleDetectedError as e:
    print(f"Would create cycle: {e}")
```

### GraphError

Raised for general graph operation errors.

```python
try:
    service.delete_concept("CoreConcept")
except GraphError as e:
    print(f"Graph operation failed: {e}")
```

---

## Common Workflows

### Workflow 1: Check and Fix Validation Issues

```python
# Validate graph
errors = service.validate_graph()

# Show issues
for error in errors:
    if error.severity == "error":
        print(f"Error: {error.message}")
    else:
        print(f"Warning: {error.message}")

# Get suggestions
suggestions = service.get_autofix_suggestions()

# Apply fixes
for suggestion in suggestions:
    if suggestion["type"] == "remove_orphaned":
        for concept in suggestion["concepts"]:
            service.delete_concept(concept)
```

### Workflow 2: Safe Bulk Operations

```python
# Create backup first
backup_id = service.create_backup("Before bulk operations")

try:
    # Perform operations
    for concept_data in new_concepts:
        service.create_concept(**concept_data)
    
    # Validate
    errors = service.validate_graph()
    if any(e.severity == "error" for e in errors):
        raise Exception("Validation failed")
    
except Exception as e:
    print(f"Operation failed: {e}")
    service.restore_backup(backup_id)
```

### Workflow 3: Export for Version Control

```python
# Export current state
export_data = service.export_graph()

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"graph_backup_{timestamp}.json"

with open(filename, "w") as f:
    json.dump(export_data, f, indent=2)

# Commit to version control
```
