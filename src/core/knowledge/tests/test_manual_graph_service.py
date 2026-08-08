"""Tests for manual graph service"""
import pytest
from datetime import datetime
from typing import List

from src.core.knowledge.services.manual_graph_service import ManualGraphService
from src.core.knowledge.models.graph import ConceptGraph
from src.core.knowledge.models.manual_graph_models import RelationshipType
from src.core.knowledge.exceptions import (
    ConceptNotFoundError,
    RelationshipNotFoundError,
    CycleDetectedError,
    GraphError
)


class TestManualGraphServiceBasics:
    """Test basic operations of manual graph service"""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service for each test"""
        return ManualGraphService(ConceptGraph())
    
    def test_create_concept_basic(self, service):
        """Test creating a basic concept"""
        concept_id = service.create_concept("Machine Learning")
        assert concept_id
        assert service.graph.has_concept("Machine Learning")
    
    def test_create_concept_with_aliases(self, service):
        """Test creating concept with aliases"""
        aliases = ["ML", "Machine Learning"]
        concept_id = service.create_concept(
            "Machine Learning",
            aliases=aliases
        )
        
        node = service.graph.get_concept("Machine Learning")
        assert node.primary_concept.aliases == aliases
    
    def test_create_concept_with_definitions(self, service):
        """Test creating concept with definitions"""
        definitions = ["Subset of AI", "Learning from data"]
        concept_id = service.create_concept(
            "Machine Learning",
            definitions=definitions
        )
        
        node = service.graph.get_concept("Machine Learning")
        assert node.primary_concept.definitions == definitions
    
    def test_create_duplicate_concept_fails(self, service):
        """Test that creating duplicate concept raises error"""
        service.create_concept("Machine Learning")
        
        with pytest.raises(ValueError) as exc_info:
            service.create_concept("Machine Learning")
        
        assert "already exists" in str(exc_info.value)
    
    def test_create_concept_empty_name_fails(self, service):
        """Test that empty concept name raises error"""
        with pytest.raises(ValueError):
            service.create_concept("")
    
    def test_create_concept_whitespace_name_fails(self, service):
        """Test that whitespace-only name raises error"""
        with pytest.raises(ValueError):
            service.create_concept("   ")
    
    def test_update_concept_name(self, service):
        """Test updating concept name"""
        service.create_concept("ML")
        service.update_concept("ML", name="Machine Learning")
        
        assert not service.graph.has_concept("ML")
        assert service.graph.has_concept("Machine Learning")
    
    def test_update_concept_aliases(self, service):
        """Test updating concept aliases"""
        service.create_concept("Machine Learning")
        service.update_concept(
            "Machine Learning",
            aliases=["ML", "Learning Systems"]
        )
        
        node = service.graph.get_concept("Machine Learning")
        assert set(node.primary_concept.aliases) == {"ML", "Learning Systems"}
    
    def test_update_nonexistent_concept_fails(self, service):
        """Test updating nonexistent concept raises error"""
        with pytest.raises(ConceptNotFoundError):
            service.update_concept("NonExistent")
    
    def test_add_alias(self, service):
        """Test adding an alias to a concept"""
        service.create_concept("Machine Learning")
        service.add_alias("Machine Learning", "ML")
        
        node = service.graph.get_concept("Machine Learning")
        assert "ML" in node.primary_concept.aliases
    
    def test_add_duplicate_alias_fails(self, service):
        """Test that adding duplicate alias raises error"""
        service.create_concept("Machine Learning", aliases=["ML"])
        
        with pytest.raises(ValueError):
            service.add_alias("Machine Learning", "ML")
    
    def test_remove_alias(self, service):
        """Test removing an alias"""
        service.create_concept("Machine Learning", aliases=["ML", "Learning"])
        service.remove_alias("Machine Learning", "ML")
        
        node = service.graph.get_concept("Machine Learning")
        assert "ML" not in node.primary_concept.aliases
        assert "Learning" in node.primary_concept.aliases
    
    def test_remove_nonexistent_alias_fails(self, service):
        """Test removing nonexistent alias raises error"""
        service.create_concept("Machine Learning")
        
        with pytest.raises(ValueError):
            service.remove_alias("Machine Learning", "ML")
    
    def test_delete_concept_orphan(self, service):
        """Test deleting an orphaned concept"""
        service.create_concept("Machine Learning")
        service.delete_concept("Machine Learning")
        
        assert not service.graph.has_concept("Machine Learning")
    
    def test_delete_concept_with_relationships_fails(self, service):
        """Test that deleting concept with relationships fails without force"""
        service.create_concept("Machine Learning")
        service.create_concept("Neural Networks")
        service.add_relationship("Machine Learning", "Neural Networks")
        
        with pytest.raises(GraphError):
            service.delete_concept("Machine Learning")
    
    def test_delete_concept_with_force(self, service):
        """Test deleting concept with relationships using force"""
        service.create_concept("Machine Learning")
        service.create_concept("Neural Networks")
        service.add_relationship("Machine Learning", "Neural Networks")
        
        service.delete_concept("Machine Learning", force=True)
        
        assert not service.graph.has_concept("Machine Learning")


class TestRelationshipOperations:
    """Test relationship operations"""
    
    @pytest.fixture
    def service_with_concepts(self):
        """Create service with test concepts"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("Machine Learning")
        service.create_concept("Neural Networks")
        service.create_concept("Deep Learning")
        service.create_concept("Linear Regression")
        return service
    
    def test_add_relationship_basic(self, service_with_concepts):
        """Test adding a basic relationship"""
        service_with_concepts.add_relationship(
            "Machine Learning",
            "Neural Networks"
        )
        
        assert service_with_concepts.graph.graph.has_edge(
            "Machine Learning",
            "Neural Networks"
        )
    
    def test_add_relationship_with_type(self, service_with_concepts):
        """Test adding relationship with specific type"""
        service_with_concepts.add_relationship(
            "Machine Learning",
            "Neural Networks",
            relationship_type=RelationshipType.HIERARCHICAL.value
        )
        
        edge_data = service_with_concepts.graph.graph[
            "Machine Learning"]["Neural Networks"]
        assert edge_data['data'].relationship.relation == RelationshipType.HIERARCHICAL.value
    
    def test_add_relationship_invalid_type_fails(self, service_with_concepts):
        """Test that invalid relationship type raises error"""
        with pytest.raises(ValueError):
            service_with_concepts.add_relationship(
                "Machine Learning",
                "Neural Networks",
                relationship_type="invalid_type"
            )
    
    def test_add_relationship_invalid_strength_fails(self, service_with_concepts):
        """Test that invalid strength raises error"""
        with pytest.raises(ValueError):
            service_with_concepts.add_relationship(
                "Machine Learning",
                "Neural Networks",
                strength=1.5
            )
    
    def test_add_relationship_to_nonexistent_concept_fails(self, service_with_concepts):
        """Test that adding relationship to nonexistent concept fails"""
        with pytest.raises(ConceptNotFoundError):
            service_with_concepts.add_relationship(
                "Machine Learning",
                "NonExistent"
            )
    
    def test_update_relationship(self, service_with_concepts):
        """Test updating a relationship"""
        service_with_concepts.add_relationship(
            "Machine Learning",
            "Neural Networks",
            relationship_type=RelationshipType.RELATED.value,
            strength=0.5
        )
        
        service_with_concepts.update_relationship(
            "Machine Learning",
            "Neural Networks",
            relationship_type=RelationshipType.HIERARCHICAL.value,
            strength=0.9
        )
        
        edge_data = service_with_concepts.graph.graph[
            "Machine Learning"]["Neural Networks"]
        rel = edge_data['data'].relationship
        assert rel.relation == RelationshipType.HIERARCHICAL.value
        assert rel.strength == 0.9
    
    def test_remove_relationship(self, service_with_concepts):
        """Test removing a relationship"""
        service_with_concepts.add_relationship(
            "Machine Learning",
            "Neural Networks"
        )
        
        service_with_concepts.remove_relationship(
            "Machine Learning",
            "Neural Networks"
        )
        
        assert not service_with_concepts.graph.graph.has_edge(
            "Machine Learning",
            "Neural Networks"
        )
    
    def test_remove_nonexistent_relationship_fails(self, service_with_concepts):
        """Test removing nonexistent relationship fails"""
        with pytest.raises(RelationshipNotFoundError):
            service_with_concepts.remove_relationship(
                "Machine Learning",
                "Neural Networks"
            )


class TestCycleDetection:
    """Test cycle detection and validation"""
    
    @pytest.fixture
    def service_with_concepts(self):
        """Create service with test concepts"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("A")
        service.create_concept("B")
        service.create_concept("C")
        service.create_concept("D")
        return service
    
    def test_no_cycles_initially(self, service_with_concepts):
        """Test that new graph has no cycles"""
        cycles = service_with_concepts.detect_circular_prerequisites()
        assert len(cycles) == 0
    
    def test_detect_simple_cycle(self, service_with_concepts):
        """Test detecting a simple cycle with non-prerequisite relationships"""
        # Use non-prerequisite relationships to form cycles (allowed)
        service_with_concepts.add_relationship(
            "A", "B",
            relationship_type=RelationshipType.RELATED.value
        )
        service_with_concepts.add_relationship(
            "B", "C",
            relationship_type=RelationshipType.RELATED.value
        )
        service_with_concepts.add_relationship(
            "C", "A",
            relationship_type=RelationshipType.RELATED.value
        )
        
        cycles = service_with_concepts.detect_circular_prerequisites()
        # Since we're using non-prerequisite relationships, should detect the cycle in general graph
        assert len(cycles) == 1
    
    def test_prevent_prerequisite_cycle(self, service_with_concepts):
        """Test that prerequisite relationships prevent cycles"""
        service_with_concepts.add_relationship(
            "A", "B",
            relationship_type=RelationshipType.PREREQUISITE.value
        )
        service_with_concepts.add_relationship(
            "B", "C",
            relationship_type=RelationshipType.PREREQUISITE.value
        )
        
        # This should fail because it would create a cycle
        with pytest.raises(CycleDetectedError):
            service_with_concepts.add_relationship(
                "C", "A",
                relationship_type=RelationshipType.PREREQUISITE.value
            )
    
    def test_non_prerequisite_relationships_allow_cycles(self, service_with_concepts):
        """Test that non-prerequisite relationships can form cycles"""
        # Non-prerequisite relationships should be allowed to form cycles
        service_with_concepts.add_relationship(
            "A", "B",
            relationship_type=RelationshipType.RELATED.value
        )
        service_with_concepts.add_relationship(
            "B", "A",
            relationship_type=RelationshipType.RELATED.value
        )
        
        # Should not raise error
        assert service_with_concepts.graph.graph.has_edge("A", "B")
        assert service_with_concepts.graph.graph.has_edge("B", "A")


class TestValidation:
    """Test validation functionality"""
    
    @pytest.fixture
    def service(self):
        """Create service with test data"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("Python")
        service.create_concept("Java", aliases=["Duplicate", "Language"])
        service.create_concept("JavaScript", aliases=["Duplicate"])
        service.create_concept("Orphan")
        return service
    
    def test_validate_concept_name_valid(self, service):
        """Test validating valid concept name"""
        valid, msg = service.validate_concept_name("Machine Learning")
        assert valid
        assert not msg
    
    def test_validate_concept_name_empty(self, service):
        """Test validating empty concept name"""
        valid, msg = service.validate_concept_name("")
        assert not valid
        assert msg
    
    def test_validate_concept_name_whitespace(self, service):
        """Test validating whitespace concept name"""
        valid, msg = service.validate_concept_name("   ")
        assert not valid
        assert msg
    
    def test_validate_concept_name_too_long(self, service):
        """Test validating too long concept name"""
        long_name = "A" * 201
        valid, msg = service.validate_concept_name(long_name)
        assert not valid
        assert msg
    
    def test_find_orphaned_concepts(self, service):
        """Test finding orphaned concepts"""
        service.create_concept("Connected1")
        service.create_concept("Connected2")
        service.add_relationship("Connected1", "Connected2")
        
        orphaned = service._find_orphaned_concepts()
        
        assert "Orphan" in orphaned
        assert "Connected1" not in orphaned
        assert "Connected2" not in orphaned
    
    def test_find_duplicate_aliases(self, service):
        """Test finding duplicate aliases"""
        duplicates = service._find_duplicate_aliases()
        
        assert "Duplicate" in duplicates
        assert len(duplicates["Duplicate"]) == 2
    
    def test_validate_graph_reports_issues(self, service):
        """Test that validate_graph reports issues"""
        errors = service.validate_graph()
        
        # Should have orphaned concepts and duplicate aliases
        error_types = [e.error_type for e in errors]
        assert "orphaned_concepts" in error_types
        assert "duplicate_aliases" in error_types


class TestExportImport:
    """Test export/import functionality"""
    
    @pytest.fixture
    def service_with_data(self):
        """Create service with test data"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("ML", aliases=["Machine Learning"])
        service.create_concept("NN", definitions=["Neural Network"])
        service.add_relationship("ML", "NN", relationship_type=RelationshipType.HIERARCHICAL.value)
        return service
    
    def test_export_graph_structure(self, service_with_data):
        """Test exporting graph structure"""
        export = service_with_data.export_graph()
        
        assert "version" in export
        assert "nodes" in export
        assert "edges" in export
        assert "stats" in export
        assert len(export["nodes"]) == 2
        assert len(export["edges"]) == 1
    
    def test_export_contains_concept_data(self, service_with_data):
        """Test that export contains concept data"""
        export = service_with_data.export_graph()
        
        ml_node = next(n for n in export["nodes"] if n["name"] == "ML")
        assert ml_node["aliases"] == ["Machine Learning"]
        
        nn_node = next(n for n in export["nodes"] if n["name"] == "NN")
        assert nn_node["definitions"] == ["Neural Network"]
    
    def test_export_contains_relationship_data(self, service_with_data):
        """Test that export contains relationship data"""
        export = service_with_data.export_graph()
        
        edge = export["edges"][0]
        assert edge["from"] == "ML"
        assert edge["to"] == "NN"
        assert edge["type"] == RelationshipType.HIERARCHICAL.value
    
    def test_import_graph_basic(self):
        """Test importing a basic graph"""
        service = ManualGraphService(ConceptGraph())
        
        import_data = {
            "version": "1.0",
            "nodes": [
                {"name": "A", "aliases": [], "definitions": [], "score": 0.0},
                {"name": "B", "aliases": [], "definitions": [], "score": 0.0}
            ],
            "edges": [
                {"from": "A", "to": "B", "type": "related", "strength": 1.0}
            ]
        }
        
        service.import_graph(import_data)
        
        assert service.graph.has_concept("A")
        assert service.graph.has_concept("B")
        assert service.graph.graph.has_edge("A", "B")
    
    def test_import_export_roundtrip(self, service_with_data):
        """Test that export -> import preserves data"""
        export1 = service_with_data.export_graph()
        
        new_service = ManualGraphService(ConceptGraph())
        new_service.import_graph(export1)
        
        export2 = new_service.export_graph()
        
        assert len(export1["nodes"]) == len(export2["nodes"])
        assert len(export1["edges"]) == len(export2["edges"])
    
    def test_import_with_merge(self):
        """Test importing with merge=True"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("Existing")
        
        import_data = {
            "version": "1.0",
            "nodes": [
                {"name": "New", "aliases": [], "definitions": [], "score": 0.0}
            ],
            "edges": []
        }
        
        service.import_graph(import_data, merge=True)
        
        assert service.graph.has_concept("Existing")
        assert service.graph.has_concept("New")
    
    def test_import_invalid_data_fails(self):
        """Test that importing invalid data fails"""
        service = ManualGraphService(ConceptGraph())
        
        with pytest.raises(ValueError):
            service.import_graph({"invalid": "data"})


class TestBackupRestore:
    """Test backup and restore functionality"""
    
    @pytest.fixture
    def service_with_data(self):
        """Create service with test data"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("ML")
        service.create_concept("NN")
        service.add_relationship("ML", "NN")
        return service
    
    def test_create_backup(self, service_with_data):
        """Test creating a backup"""
        backup_id = service_with_data.create_backup("Test backup")
        
        assert backup_id
        assert backup_id in service_with_data.backups
    
    def test_backup_contains_graph_data(self, service_with_data):
        """Test that backup contains graph data"""
        backup_id = service_with_data.create_backup()
        backup = service_with_data.backups[backup_id]
        
        assert backup.graph_data is not None
        assert len(backup.graph_data["nodes"]) == 2
        assert len(backup.graph_data["edges"]) == 1
    
    def test_restore_backup(self, service_with_data):
        """Test restoring from backup"""
        backup_id = service_with_data.create_backup()
        
        # Modify the graph
        service_with_data.delete_concept("ML", force=True)
        
        assert not service_with_data.graph.has_concept("ML")
        
        # Restore
        service_with_data.restore_backup(backup_id)
        
        assert service_with_data.graph.has_concept("ML")
        assert service_with_data.graph.has_concept("NN")
    
    def test_list_backups(self, service_with_data):
        """Test listing backups"""
        service_with_data.create_backup("Backup 1")
        service_with_data.create_backup("Backup 2")
        
        backups = service_with_data.list_backups()
        
        assert len(backups) == 2
    
    def test_delete_backup(self, service_with_data):
        """Test deleting a backup"""
        backup_id = service_with_data.create_backup()
        
        service_with_data.delete_backup(backup_id)
        
        assert backup_id not in service_with_data.backups


class TestEditHistory:
    """Test edit history functionality"""
    
    def test_edit_history_records_operations(self):
        """Test that edit history records operations"""
        service = ManualGraphService(ConceptGraph())
        
        service.create_concept("ML")
        service.create_concept("NN")
        service.add_relationship("ML", "NN")
        
        history = service.get_edit_history()
        
        assert len(history) >= 3
    
    def test_edit_history_has_timestamps(self):
        """Test that edit history has timestamps"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("ML")
        
        history = service.get_edit_history()
        
        assert "timestamp" in history[0]


class TestQueryMethods:
    """Test query methods"""
    
    @pytest.fixture
    def service_with_relationships(self):
        """Create service with relationships"""
        service = ManualGraphService(ConceptGraph())
        service.create_concept("A")
        service.create_concept("B")
        service.create_concept("C")
        service.add_relationship("A", "B")
        service.add_relationship("B", "C")
        return service
    
    def test_get_concept_info(self, service_with_relationships):
        """Test getting concept info"""
        info = service_with_relationships.get_concept_info("B")
        
        assert info["name"] == "B"
        assert info["incoming_relationships"] == 1
        assert info["outgoing_relationships"] == 1
        assert "A" in info["predecessors"]
        assert "C" in info["successors"]
    
    def test_get_graph_stats(self, service_with_relationships):
        """Test getting graph statistics"""
        stats = service_with_relationships.get_graph_stats()
        
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2
        assert stats["is_dag"] == True
        assert "density" in stats
