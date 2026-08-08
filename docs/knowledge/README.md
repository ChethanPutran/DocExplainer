# Doc Explainer Knowledge Management Documentation

This directory contains comprehensive documentation for the Doc Explainer knowledge management system, with a focus on the Manual Concept Graph Editor.

## Documentation Files

### 1. [QUICK_START.md](QUICK_START.md)
**Best for**: Getting started quickly
- 5-minute tutorial
- Common tasks and workflows
- Quick reference tables
- Troubleshooting guide

**Read this first** if you want to start using the service immediately.

### 2. [MANUAL_GRAPH_EDITOR.md](MANUAL_GRAPH_EDITOR.md)
**Best for**: Comprehensive feature documentation
- Feature overview
- Concept and relationship management
- Validation and consistency checking
- Export/import functionality
- Backup and restore operations
- Query and inspection methods
- Workflow examples
- Best practices

**Read this** for in-depth understanding of all features.

### 3. [MANUAL_GRAPH_API.md](MANUAL_GRAPH_API.md)
**Best for**: API reference and integration
- Complete method signatures
- Parameter documentation
- Return value specifications
- Exception classes
- Code examples for each method
- Constant definitions

**Use this** as a reference while coding.

### 4. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
**Best for**: Technical overview and status
- Implementation overview
- Files created and modified
- Features implemented with checkmarks
- Test results and coverage
- Data model specifications
- Performance characteristics
- Integration points

**Read this** to understand the technical implementation.

## Quick Navigation

### By Task

**I want to...**

| Task | Document | Section |
|------|----------|---------|
| Get started quickly | QUICK_START.md | 5-Minute Tutorial |
| Create/edit concepts | MANUAL_GRAPH_EDITOR.md | Manual Concept Creation/Editing |
| Define relationships | MANUAL_GRAPH_EDITOR.md | Relationship Definition |
| Check for errors | MANUAL_GRAPH_EDITOR.md | Graph Validation & Consistency |
| Save/load graphs | MANUAL_GRAPH_EDITOR.md | Export/Import Functionality |
| Backup data | MANUAL_GRAPH_EDITOR.md | Backup & Restore |
| Look up a method | MANUAL_GRAPH_API.md | Class: ManualGraphService |
| Troubleshoot | QUICK_START.md | Troubleshooting |
| Understand the code | IMPLEMENTATION_SUMMARY.md | Overview |

### By User Type

**I am a...**

- **New User**: Start with [QUICK_START.md](QUICK_START.md)
- **Developer**: Read [MANUAL_GRAPH_EDITOR.md](MANUAL_GRAPH_EDITOR.md) then use [MANUAL_GRAPH_API.md](MANUAL_GRAPH_API.md) as reference
- **Architect**: Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Maintainer**: Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) and source code

## Key Concepts

### Concepts
- Fundamental units representing ideas or topics
- Have names, aliases, definitions, and attributes
- Can have relationships with other concepts

### Relationships
- Connect concepts together
- Types: PREREQUISITE, SIMILAR, RELATED, HIERARCHICAL, DEPENDS_ON, PART_OF
- Have strength (0.0-1.0) indicating confidence/importance

### Validation
- Checks for circular dependencies
- Detects orphaned concepts
- Finds duplicate aliases
- Validates concept names
- Checks relationship coherence

### Graph Operations
- Create, read, update, delete concepts and relationships
- Export to JSON for storage/sharing
- Import from JSON for merging/loading
- Create snapshots for versioning
- Backup/restore for recovery

## Code Examples

### Create a Concept
```python
service.create_concept("Python")
```

### Add a Relationship
```python
service.add_relationship("Python", "Machine Learning",
    relationship_type="prerequisite", strength=0.9)
```

### Validate Graph
```python
errors = service.validate_graph()
```

### Export to JSON
```python
data = service.export_graph()
```

### Create Backup
```python
backup_id = service.create_backup("Stable version")
```

See [QUICK_START.md](QUICK_START.md) for more examples.

## Testing

The implementation includes 51 comprehensive tests:

```bash
cd /path/to/doc_explainer
source venv/bin/activate
pytest src/core/knowledge/tests/test_manual_graph_service.py -v
```

**Result**: All 51 tests pass ✓

Test coverage includes:
- CRUD operations
- Relationship management
- Cycle detection
- Validation checks
- Export/import roundtrips
- Backup/restore functionality
- Query methods

## Integration

The Manual Graph Service integrates with:
- `KnowledgeRepository` for persistence
- `ConceptGraph` for graph operations
- Existing `Concept` and `ConceptNode` models
- Doc Explainer exception hierarchy

```python
from src.core.knowledge.services.manual_graph_service import ManualGraphService

service = ManualGraphService(existing_graph)
```

## Support

### Common Issues

**Q: How do I prevent circular dependencies?**
A: The service automatically prevents prerequisite cycles. Use non-prerequisite relationship types if you need cycles.

**Q: Can I rename a concept?**
A: Yes, use `update_concept(current_name, name=new_name)`

**Q: How do I merge two graphs?**
A: Export one graph, then import it with `merge=True`

**Q: What happens when I delete a concept?**
A: By default, deletion fails if the concept has relationships. Use `force=True` to remove all relationships.

### Getting Help

1. Check the **Troubleshooting** section in [QUICK_START.md](QUICK_START.md)
2. Look up specific methods in [MANUAL_GRAPH_API.md](MANUAL_GRAPH_API.md)
3. Review examples in [MANUAL_GRAPH_EDITOR.md](MANUAL_GRAPH_EDITOR.md)
4. Examine test cases in `src/core/knowledge/tests/test_manual_graph_service.py`

## Performance Tips

1. **Use backups** before major changes
2. **Batch operations** when possible
3. **Export regularly** for version control
4. **Validate after imports** to catch issues
5. **Monitor graph size** for very large graphs

## Files Reference

### Source Code
- `src/core/knowledge/services/manual_graph_service.py` - Main service (914 lines)
- `src/core/knowledge/models/manual_graph_models.py` - Data models (156 lines)
- `src/core/knowledge/models/graph.py` - Graph implementation (fixed)

### Tests
- `src/core/knowledge/tests/test_manual_graph_service.py` - 51 tests (623 lines)

### Documentation
- This README
- QUICK_START.md (197 lines)
- MANUAL_GRAPH_EDITOR.md (406 lines)
- MANUAL_GRAPH_API.md (512 lines)
- IMPLEMENTATION_SUMMARY.md (328 lines)

## Version Information

**Manual Graph Editor v1.0**
- Status: Production Ready
- Tests: 51 tests, 100% pass rate
- Documentation: Complete
- Code Quality: High (no linting issues)

## License

Part of the Doc Explainer project.
