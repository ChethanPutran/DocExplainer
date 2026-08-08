"""Service for manual concept graph editing and management"""
import hashlib
import json
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime
from collections import deque, defaultdict

from ..models.concept import Concept
from ..models.relationship import ConceptNode, ConceptRelationship, ConceptNodeRelationship
from ..models.graph import ConceptGraph
from ..models.manual_graph_models import (
    ConceptEdit,
    RelationshipEdit,
    GraphSnapshot,
    ValidationError,
    GraphBackup,
    RelationshipType,
    OperationType
)
from ..exceptions import (
    KnowledgeBaseError,
    ConceptNotFoundError,
    RelationshipNotFoundError,
    GraphError,
    CycleDetectedError
)


class ManualGraphService:
    """Service for manual editing and management of concept graphs"""
    
    VALID_RELATIONSHIP_TYPES = {rel.value for rel in RelationshipType}
    
    def __init__(self, graph: Optional[ConceptGraph] = None):
        self.graph = graph or ConceptGraph()
        self.edit_history: List[Dict[str, Any]] = []
        self.backups: Dict[str, GraphBackup] = {}
        self.concept_index: Dict[str, str] = {}  # Maps concept names to IDs
    
    # ==================== Concept CRUD Operations ====================
    
    def create_concept(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        definitions: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new concept and add it to the graph
        
        Args:
            name: Concept name
            aliases: Alternative names for the concept
            definitions: Definitions of the concept
            attributes: Additional attributes
        
        Returns:
            Concept ID
        
        Raises:
            ValueError: If concept name is invalid or already exists
        """
        # Validate
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Concept name cannot be empty")
        
        name = name.strip()
        
        if self.graph.has_concept(name):
            raise ValueError(f"Concept '{name}' already exists in the graph")
        
        # Create concept
        concept = Concept(
            name=name,
            aliases=aliases or [],
            definitions=definitions or [],
            attributes=attributes or {}
        )
        
        # Create node and add to graph
        node = ConceptNode(primary_concept=concept)
        self.graph.add_concept_node(node)
        self.concept_index[name] = concept.id
        
        # Record edit
        self._record_edit({
            "type": "concept_create",
            "concept_id": concept.id,
            "concept_name": name,
            "data": {
                "aliases": aliases or [],
                "definitions": definitions or [],
                "attributes": attributes or {}
            }
        })
        
        return concept.id
    
    def update_concept(
        self,
        concept_name: str,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        definitions: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Update an existing concept
        
        Args:
            concept_name: Current concept name
            name: New name (optional)
            aliases: New aliases (optional)
            definitions: New definitions (optional)
            attributes: New attributes (optional)
        
        Returns:
            Updated concept ID
        
        Raises:
            ConceptNotFoundError: If concept not found
            ValueError: If new name conflicts with existing concept
        """
        node = self.graph.get_concept(concept_name)
        if not node:
            raise ConceptNotFoundError(f"Concept '{concept_name}' not found")
        
        concept = node.primary_concept
        changes = {}
        
        # Update name if provided
        if name and name != concept_name:
            if self.graph.has_concept(name):
                raise ValueError(f"Concept '{name}' already exists")
            changes["name"] = (concept_name, name)
            
            # Rename in graph using networkx relabel_nodes
            import networkx as nx
            mapping = {concept_name: name}
            self.graph.graph = nx.relabel_nodes(self.graph.graph, mapping)
            concept.name = name
            
            # Update index
            if concept_name in self.concept_index:
                del self.concept_index[concept_name]
            self.concept_index[name] = concept.id
        
        # Update aliases if provided
        if aliases is not None:
            changes["aliases"] = (concept.aliases.copy(), aliases)
            concept.aliases = aliases
        
        # Update definitions if provided
        if definitions is not None:
            changes["definitions"] = (concept.definitions.copy(), definitions)
            concept.definitions = definitions
        
        # Update attributes if provided
        if attributes is not None:
            changes["attributes"] = (concept.attributes.copy(), attributes)
            concept.attributes = attributes
        
        # Record edit
        self._record_edit({
            "type": "concept_update",
            "concept_id": concept.id,
            "concept_name": concept_name,
            "changes": changes
        })
        
        return concept.id
    
    def add_alias(self, concept_name: str, alias: str) -> None:
        """
        Add an alias to a concept
        
        Args:
            concept_name: Concept name
            alias: New alias to add
        
        Raises:
            ConceptNotFoundError: If concept not found
            ValueError: If alias is invalid or already exists
        """
        if not alias or not isinstance(alias, str):
            raise ValueError("Alias must be a non-empty string")
        
        node = self.graph.get_concept(concept_name)
        if not node:
            raise ConceptNotFoundError(f"Concept '{concept_name}' not found")
        
        if alias in node.primary_concept.aliases:
            raise ValueError(f"Alias '{alias}' already exists for concept '{concept_name}'")
        
        node.primary_concept.aliases.append(alias)
        
        self._record_edit({
            "type": "alias_add",
            "concept_name": concept_name,
            "alias": alias
        })
    
    def remove_alias(self, concept_name: str, alias: str) -> None:
        """
        Remove an alias from a concept
        
        Args:
            concept_name: Concept name
            alias: Alias to remove
        
        Raises:
            ConceptNotFoundError: If concept not found
            ValueError: If alias not found
        """
        node = self.graph.get_concept(concept_name)
        if not node:
            raise ConceptNotFoundError(f"Concept '{concept_name}' not found")
        
        if alias not in node.primary_concept.aliases:
            raise ValueError(f"Alias '{alias}' not found for concept '{concept_name}'")
        
        node.primary_concept.aliases.remove(alias)
        
        self._record_edit({
            "type": "alias_remove",
            "concept_name": concept_name,
            "alias": alias
        })
    
    def delete_concept(self, concept_name: str, force: bool = False) -> None:
        """
        Delete a concept from the graph
        
        Args:
            concept_name: Concept name
            force: If True, remove all relationships; if False, raise error if concept has relationships
        
        Raises:
            ConceptNotFoundError: If concept not found
            GraphError: If concept has relationships and force=False
        """
        if not self.graph.has_concept(concept_name):
            raise ConceptNotFoundError(f"Concept '{concept_name}' not found")
        
        # Check for relationships
        in_edges = list(self.graph.graph.in_edges(concept_name))
        out_edges = list(self.graph.graph.out_edges(concept_name))
        
        if (in_edges or out_edges) and not force:
            raise GraphError(
                f"Cannot delete concept '{concept_name}' without removing relationships. "
                f"Use force=True to remove all relationships."
            )
        
        # Remove all relationships
        for source, target in in_edges + out_edges:
            self.graph.graph.remove_edge(source, target)
        
        # Remove node
        self.graph.graph.remove_node(concept_name)
        
        if concept_name in self.concept_index:
            del self.concept_index[concept_name]
        
        self._record_edit({
            "type": "concept_delete",
            "concept_name": concept_name,
            "forced": force
        })
    
    # ==================== Relationship Operations ====================
    
    def add_relationship(
        self,
        from_concept_name: str,
        to_concept_name: str,
        relationship_type: str = RelationshipType.RELATED.value,
        strength: float = 1.0,
        definition: str = ""
    ) -> None:
        """
        Add a relationship between two concepts
        
        Args:
            from_concept_name: Source concept name
            to_concept_name: Target concept name
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
            definition: Definition of the relationship
        
        Raises:
            ConceptNotFoundError: If either concept not found
            ValueError: If relationship type invalid or strength out of range
            CycleDetectedError: If relationship would create cycle with prerequisites
        """
        if relationship_type not in self.VALID_RELATIONSHIP_TYPES:
            raise ValueError(f"Invalid relationship type: {relationship_type}")
        
        if not (0.0 <= strength <= 1.0):
            raise ValueError("Strength must be between 0.0 and 1.0")
        
        from_node = self.graph.get_concept(from_concept_name)
        to_node = self.graph.get_concept(to_concept_name)
        
        if not from_node:
            raise ConceptNotFoundError(f"Concept '{from_concept_name}' not found")
        if not to_node:
            raise ConceptNotFoundError(f"Concept '{to_concept_name}' not found")
        
        # Check for prerequisite cycles
        if relationship_type == RelationshipType.PREREQUISITE.value:
            if self._would_create_cycle(from_concept_name, to_concept_name):
                raise CycleDetectedError(
                    f"Adding prerequisite from '{from_concept_name}' to '{to_concept_name}' "
                    f"would create a circular dependency"
                )
        
        # Create relationship
        concept_rel = ConceptRelationship(
            concept1=from_node.primary_concept,
            concept2=to_node.primary_concept,
            relation=relationship_type,
            definition=definition,
            strength=strength
        )
        
        node_rel = ConceptNodeRelationship(from_node, to_node, concept_rel)
        self.graph.add_relationship(from_node, to_node, node_rel)
        
        self._record_edit({
            "type": "relationship_add",
            "from_concept": from_concept_name,
            "to_concept": to_concept_name,
            "relationship_type": relationship_type,
            "strength": strength,
            "definition": definition
        })
    
    def remove_relationship(
        self,
        from_concept_name: str,
        to_concept_name: str
    ) -> None:
        """
        Remove a relationship between two concepts
        
        Args:
            from_concept_name: Source concept name
            to_concept_name: Target concept name
        
        Raises:
            ConceptNotFoundError: If either concept not found
            RelationshipNotFoundError: If relationship not found
        """
        if not self.graph.has_concept(from_concept_name):
            raise ConceptNotFoundError(f"Concept '{from_concept_name}' not found")
        if not self.graph.has_concept(to_concept_name):
            raise ConceptNotFoundError(f"Concept '{to_concept_name}' not found")
        
        if not self.graph.graph.has_edge(from_concept_name, to_concept_name):
            raise RelationshipNotFoundError(
                f"No relationship found from '{from_concept_name}' to '{to_concept_name}'"
            )
        
        self.graph.graph.remove_edge(from_concept_name, to_concept_name)
        
        self._record_edit({
            "type": "relationship_remove",
            "from_concept": from_concept_name,
            "to_concept": to_concept_name
        })
    
    def update_relationship(
        self,
        from_concept_name: str,
        to_concept_name: str,
        relationship_type: Optional[str] = None,
        strength: Optional[float] = None,
        definition: Optional[str] = None
    ) -> None:
        """
        Update a relationship between two concepts
        
        Args:
            from_concept_name: Source concept name
            to_concept_name: Target concept name
            relationship_type: New relationship type (optional)
            strength: New strength (optional)
            definition: New definition (optional)
        
        Raises:
            RelationshipNotFoundError: If relationship not found
            ValueError: If invalid parameters
        """
        if not self.graph.graph.has_edge(from_concept_name, to_concept_name):
            raise RelationshipNotFoundError(
                f"No relationship found from '{from_concept_name}' to '{to_concept_name}'"
            )
        
        edge_data = self.graph.graph[from_concept_name][to_concept_name]
        node_rel: ConceptNodeRelationship = edge_data.get('data')
        
        if not node_rel:
            raise RelationshipNotFoundError("Relationship data not found")
        
        changes = {}
        
        if relationship_type is not None:
            if relationship_type not in self.VALID_RELATIONSHIP_TYPES:
                raise ValueError(f"Invalid relationship type: {relationship_type}")
            changes["relationship_type"] = (node_rel.relationship.relation, relationship_type)
            node_rel.relationship.relation = relationship_type
        
        if strength is not None:
            if not (0.0 <= strength <= 1.0):
                raise ValueError("Strength must be between 0.0 and 1.0")
            changes["strength"] = (node_rel.relationship.strength, strength)
            node_rel.relationship.strength = strength
        
        if definition is not None:
            changes["definition"] = (node_rel.relationship.definition, definition)
            node_rel.relationship.definition = definition
        
        self._record_edit({
            "type": "relationship_update",
            "from_concept": from_concept_name,
            "to_concept": to_concept_name,
            "changes": changes
        })
    
    # ==================== Validation & Consistency Checking ====================
    
    def validate_graph(self) -> List[ValidationError]:
        """
        Validate the graph for consistency issues
        
        Returns:
            List of validation errors found
        """
        errors = []
        
        # Check for orphaned concepts
        orphaned = self._find_orphaned_concepts()
        if orphaned:
            errors.append(ValidationError(
                error_type="orphaned_concepts",
                message=f"Found {len(orphaned)} orphaned concepts",
                affected_concepts=orphaned,
                severity="warning",
                suggestion="Consider adding these concepts to the graph or deleting them"
            ))
        
        # Check for prerequisite cycles
        cycles = self._find_cycles()
        if cycles:
            errors.append(ValidationError(
                error_type="circular_prerequisites",
                message=f"Found {len(cycles)} circular prerequisite chains",
                affected_relationships=cycles,
                severity="error",
                suggestion="Break at least one edge in each cycle"
            ))
        
        # Check for duplicate aliases
        duplicates = self._find_duplicate_aliases()
        if duplicates:
            errors.append(ValidationError(
                error_type="duplicate_aliases",
                message=f"Found {len(duplicates)} duplicate aliases",
                affected_concepts=list(duplicates.keys()),
                severity="warning",
                suggestion="Remove duplicate aliases"
            ))
        
        # Check for invalid concept names
        invalid_names = self._find_invalid_names()
        if invalid_names:
            errors.append(ValidationError(
                error_type="invalid_names",
                message=f"Found {len(invalid_names)} invalid concept names",
                affected_concepts=invalid_names,
                severity="error",
                suggestion="Rename concepts with empty or invalid names"
            ))
        
        return errors
    
    def detect_circular_prerequisites(self) -> List[List[str]]:
        """
        Detect circular prerequisite chains
        
        Returns:
            List of cycles (each cycle is a list of concept names)
        """
        return self._find_cycles()
    
    def validate_concept_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate a concept name
        
        Args:
            name: Concept name to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name or not isinstance(name, str):
            return False, "Name must be a non-empty string"
        
        if not name.strip():
            return False, "Name cannot be only whitespace"
        
        if len(name) > 200:
            return False, "Name must be 200 characters or less"
        
        return True, ""
    
    def validate_relationship_coherence(self) -> List[str]:
        """
        Check relationship coherence (relationships between valid concepts)
        
        Returns:
            List of issues found
        """
        issues = []
        
        for source, target in self.graph.graph.edges():
            if not self.graph.has_concept(source):
                issues.append(f"Relationship source '{source}' does not exist")
            if not self.graph.has_concept(target):
                issues.append(f"Relationship target '{target}' does not exist")
        
        return issues
    
    def get_autofix_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get suggestions for auto-fixing validation issues
        
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        # Suggest removing orphaned concepts
        orphaned = self._find_orphaned_concepts()
        if orphaned:
            suggestions.append({
                "type": "remove_orphaned",
                "concepts": orphaned,
                "action": f"Remove {len(orphaned)} orphaned concepts"
            })
        
        # Suggest breaking cycles
        cycles = self._find_cycles()
        if cycles:
            suggestions.append({
                "type": "break_cycles",
                "cycles": cycles,
                "action": f"Break {len(cycles)} prerequisite cycles"
            })
        
        # Suggest fixing duplicate aliases
        duplicates = self._find_duplicate_aliases()
        if duplicates:
            suggestions.append({
                "type": "remove_duplicate_aliases",
                "duplicates": duplicates,
                "action": "Remove duplicate aliases"
            })
        
        return suggestions
    
    # ==================== Export/Import ====================
    
    def export_graph(self) -> Dict[str, Any]:
        """
        Export graph to JSON-compatible dictionary
        
        Returns:
            Dictionary with nodes and edges
        """
        nodes = []
        edges = []
        
        # Export nodes
        for node_name, node_data in self.graph.graph.nodes(data=True):
            node: ConceptNode = node_data.get('data')
            if node:
                concept = node.primary_concept
                nodes.append({
                    "id": concept.id,
                    "name": concept.name,
                    "aliases": concept.aliases,
                    "definitions": concept.definitions,
                    "score": concept.score,
                    "frequency": concept.frequency,
                    "attributes": concept.attributes
                })
        
        # Export edges
        for source, target, edge_data in self.graph.graph.edges(data=True):
            node_rel: ConceptNodeRelationship = edge_data.get('data')
            if node_rel:
                rel = node_rel.relationship
                edges.append({
                    "from": source,
                    "to": target,
                    "type": rel.relation,
                    "strength": rel.strength,
                    "definition": rel.definition,
                    "attributes": rel.attributes
                })
        
        return {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }
    
    def import_graph(self, data: Dict[str, Any], merge: bool = False) -> None:
        """
        Import graph from JSON-compatible dictionary
        
        Args:
            data: Dictionary with nodes and edges
            merge: If True, merge with existing graph; if False, replace
        
        Raises:
            ValueError: If data format invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Import data must be a dictionary")
        
        if "nodes" not in data or "edges" not in data:
            raise ValueError("Import data must contain 'nodes' and 'edges' keys")
        
        if not merge:
            self.graph = ConceptGraph()
            self.concept_index = {}
        
        # Import nodes
        for node_data in data["nodes"]:
            concept_name = node_data.get("name")
            if not concept_name:
                continue
            
            if self.graph.has_concept(concept_name):
                continue
            
            concept = Concept(
                name=concept_name,
                aliases=node_data.get("aliases", []),
                definitions=node_data.get("definitions", []),
                score=node_data.get("score", 0.0),
                frequency=node_data.get("frequency", 0),
                attributes=node_data.get("attributes", {})
            )
            
            node = ConceptNode(primary_concept=concept)
            self.graph.add_concept_node(node)
            self.concept_index[concept_name] = concept.id
        
        # Import edges
        for edge_data in data["edges"]:
            from_name = edge_data.get("from")
            to_name = edge_data.get("to")
            
            if not from_name or not to_name:
                continue
            
            if not self.graph.has_concept(from_name) or not self.graph.has_concept(to_name):
                continue
            
            try:
                self.add_relationship(
                    from_name,
                    to_name,
                    relationship_type=edge_data.get("type", RelationshipType.RELATED.value),
                    strength=edge_data.get("strength", 1.0),
                    definition=edge_data.get("definition", "")
                )
            except (CycleDetectedError, ValueError):
                # Skip relationships that would create cycles or are invalid
                continue
        
        self._record_edit({
            "type": "graph_import",
            "merge": merge,
            "node_count": len(data.get("nodes", [])),
            "edge_count": len(data.get("edges", []))
        })
    
    # ==================== Backup & Restore ====================
    
    def create_backup(self, description: str = "", tags: Optional[List[str]] = None) -> str:
        """
        Create a backup of the current graph
        
        Args:
            description: Description of the backup
            tags: Tags for organizing backups
        
        Returns:
            Backup ID
        """
        # Use timestamp with microseconds to ensure uniqueness
        from datetime import datetime as dt
        import time
        timestamp = dt.utcnow()
        backup_id = f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}_{str(int(time.time() * 1000000) % 1000000).zfill(6)}"
        
        backup = GraphBackup(
            backup_id=backup_id,
            graph_data=self.export_graph(),
            description=description,
            tags=tags or []
        )
        
        self.backups[backup_id] = backup
        
        self._record_edit({
            "type": "backup_create",
            "backup_id": backup_id,
            "description": description
        })
        
        return backup_id
    
    def restore_backup(self, backup_id: str) -> None:
        """
        Restore graph from a backup
        
        Args:
            backup_id: ID of backup to restore
        
        Raises:
            ValueError: If backup not found
        """
        if backup_id not in self.backups:
            raise ValueError(f"Backup '{backup_id}' not found")
        
        backup = self.backups[backup_id]
        if not backup.graph_data:
            raise ValueError(f"Backup '{backup_id}' has no graph data")
        
        self.import_graph(backup.graph_data, merge=False)
        
        self._record_edit({
            "type": "backup_restore",
            "backup_id": backup_id
        })
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Get list of available backups"""
        return [
            {
                "backup_id": backup.backup_id,
                "timestamp": backup.timestamp.isoformat() if backup.timestamp else None,
                "description": backup.description,
                "tags": backup.tags
            }
            for backup in self.backups.values()
        ]
    
    def delete_backup(self, backup_id: str) -> None:
        """Delete a backup"""
        if backup_id not in self.backups:
            raise ValueError(f"Backup '{backup_id}' not found")
        
        del self.backups[backup_id]
        
        self._record_edit({
            "type": "backup_delete",
            "backup_id": backup_id
        })
    
    # ==================== History & Snapshots ====================
    
    def get_edit_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get edit history"""
        return self.edit_history[-limit:] if self.edit_history else []
    
    def create_snapshot(self, description: str = "") -> GraphSnapshot:
        """
        Create a snapshot of the graph
        
        Args:
            description: Snapshot description
        
        Returns:
            GraphSnapshot object
        """
        export_data = self.export_graph()
        
        snapshot = GraphSnapshot(
            nodes=export_data.get("nodes", []),
            edges=export_data.get("edges", []),
            checksum=self._calculate_checksum(export_data),
            metadata={"description": description}
        )
        
        return snapshot
    
    # ==================== Helper Methods ====================
    
    def _record_edit(self, edit: Dict[str, Any]) -> None:
        """Record an edit in history"""
        edit["timestamp"] = datetime.utcnow().isoformat()
        self.edit_history.append(edit)
    
    def _would_create_cycle(self, from_concept: str, to_concept: str) -> bool:
        """Check if adding edge would create a cycle in prerequisite relationships"""
        try:
            import networkx as nx
            # If there's already a path from to_concept to from_concept, adding an edge
            # from from_concept to to_concept would create a cycle
            return nx.has_path(self.graph.graph, to_concept, from_concept)
        except Exception:
            return False
    
    def _find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph"""
        cycles = []
        try:
            import networkx as nx
            cycles_gen = nx.simple_cycles(self.graph.graph)
            cycles = [list(cycle) for cycle in cycles_gen]
        except Exception:
            pass
        return cycles
    
    def _find_orphaned_concepts(self) -> List[str]:
        """Find concepts with no relationships"""
        orphaned = []
        for node_name in self.graph.graph.nodes():
            in_degree = self.graph.graph.in_degree(node_name)
            out_degree = self.graph.graph.out_degree(node_name)
            if in_degree == 0 and out_degree == 0:
                orphaned.append(node_name)
        return orphaned
    
    def _find_duplicate_aliases(self) -> Dict[str, List[str]]:
        """Find concepts that share aliases"""
        alias_to_concepts = defaultdict(list)
        
        for node_name, node_data in self.graph.graph.nodes(data=True):
            node: ConceptNode = node_data.get('data')
            if node:
                for alias in node.primary_concept.aliases:
                    alias_to_concepts[alias].append(node_name)
        
        duplicates = {alias: concepts for alias, concepts in alias_to_concepts.items()
                     if len(concepts) > 1}
        return duplicates
    
    def _find_invalid_names(self) -> List[str]:
        """Find concepts with invalid names"""
        invalid = []
        for node_name, node_data in self.graph.graph.nodes(data=True):
            valid, _ = self.validate_concept_name(node_name)
            if not valid:
                invalid.append(node_name)
        return invalid
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for graph data"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    # ==================== Query Methods ====================
    
    def get_concept_info(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a concept"""
        node = self.graph.get_concept(concept_name)
        if not node:
            return None
        
        concept = node.primary_concept
        
        in_edges = list(self.graph.graph.in_edges(concept_name))
        out_edges = list(self.graph.graph.out_edges(concept_name))
        
        return {
            "id": concept.id,
            "name": concept.name,
            "aliases": concept.aliases,
            "definitions": concept.definitions,
            "score": concept.score,
            "frequency": concept.frequency,
            "attributes": concept.attributes,
            "incoming_relationships": len(in_edges),
            "outgoing_relationships": len(out_edges),
            "predecessors": [source for source, _ in in_edges],
            "successors": [target for _, target in out_edges]
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph"""
        import networkx as nx
        
        return {
            "node_count": len(self.graph.graph),
            "edge_count": len(self.graph.graph.edges()),
            "density": nx.density(self.graph.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph.graph),
            "orphaned_count": len(self._find_orphaned_concepts()),
            "cycle_count": len(self._find_cycles())
        }
