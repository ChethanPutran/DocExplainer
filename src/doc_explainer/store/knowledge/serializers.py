from typing import Dict, Any, List
import numpy as np
from ...core.knowledge.models.concept import Concept
from ...core.knowledge.models.relationship import ConceptRelationship, ConceptNode, ConceptNodeRelationship
from ...core.knowledge.models.graph import ConceptGraph


class ConceptSerializer:
    """Serializer for Concept objects"""
    
    @staticmethod
    def serialize_concept(concept: Concept) -> Dict[str, Any]:
        """Serialize concept to dictionary"""
        data = {
            'id': concept.id,
            'name': concept.name,
            'aliases': concept.aliases,
            'definitions': concept.definitions,
            'score': concept.score,
            'frequency': concept.frequency,
            'first_position': concept.first_position,
            'attributes': concept.attributes,
            'occurrences': concept.occurrences
        }
        
        # Handle embedding (might be numpy array)
        if concept.embedding is not None:
            if isinstance(concept.embedding, np.ndarray):
                data['embedding'] = concept.embedding.tolist()
            else:
                data['embedding'] = concept.embedding
        
        return data
    
    @staticmethod
    def deserialize_concept(data: Dict[str, Any]) -> Concept:
        """Deserialize concept from dictionary"""
        concept = Concept(
            name=data['name'],
            aliases=data.get('aliases', []),
            definitions=data.get('definitions', []),
            score=data.get('score', 0.0),
            frequency=data.get('frequency', 0),
            first_position=data.get('first_position', -1),
            attributes=data.get('attributes', {}),
            occurrences=data.get('occurrences', [])
        )
        concept.id = data.get('id', concept.id)
        
        # Restore embedding
        embedding = data.get('embedding')
        if embedding is not None:
            if isinstance(embedding, list):
                concept.embedding = np.array(embedding)
            else:
                concept.embedding = embedding
        
        return concept
    
    @staticmethod
    def serialize_node(node: ConceptNode) -> Dict[str, Any]:
        """Serialize concept node to dictionary"""
        return {
            'primary_concept': ConceptSerializer.serialize_concept(node.primary_concept),
            'has_embedding': node.embedding is not None
        }
    
    @staticmethod
    def deserialize_node(data: Dict[str, Any]) -> ConceptNode:
        """Deserialize concept node from dictionary"""
        concept = ConceptSerializer.deserialize_concept(data['primary_concept'])
        return ConceptNode(primary_concept=concept)


class RelationshipSerializer:
    """Serializer for relationship objects"""
    
    @staticmethod
    def serialize_relationship(rel: ConceptRelationship) -> Dict[str, Any]:
        """Serialize relationship to dictionary"""
        return {
            'concept1': rel.concept1.name,
            'concept2': rel.concept2.name,
            'relation': rel.relation,
            'definition': rel.definition,
            'strength': rel.strength,
            'attributes': rel.attributes
        }
    
    @staticmethod
    def deserialize_relationship(data: Dict[str, Any], 
                                concept1: Concept, 
                                concept2: Concept) -> ConceptRelationship:
        """Deserialize relationship from dictionary"""
        return ConceptRelationship(
            concept1=concept1,
            concept2=concept2,
            relation=data.get('relation', 'related_to'),
            definition=data.get('definition', ''),
            strength=data.get('strength', 1.0),
            attributes=data.get('attributes', {})
        )
    
    @staticmethod
    def serialize_node_relationship(node_rel: ConceptNodeRelationship) -> Dict[str, Any]:
        """Serialize node relationship to dictionary"""
        return {
            'concept1': node_rel.concept1.primary_concept.name,
            'concept2': node_rel.concept2.primary_concept.name,
            'relationship': RelationshipSerializer.serialize_relationship(node_rel.relationship)
        }


class GraphSerializer:
    """Serializer for ConceptGraph objects"""
    
    @staticmethod
    def serialize_graph(graph: ConceptGraph) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        nodes = []
        for node_name, node_data in graph.graph.nodes(data=True):
            node_obj = node_data.get('data')
            if node_obj:
                nodes.append({
                    'name': node_name,
                    'data': ConceptSerializer.serialize_node(node_obj)
                })
        
        edges = []
        for u, v, data in graph.graph.edges(data=True):
            rel_wrapper = data.get('relationship')
            if rel_wrapper:
                edges.append({
                    'source': u,
                    'target': v,
                    'relationship': RelationshipSerializer.serialize_relationship(rel_wrapper.relationship)
                })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    @staticmethod
    def deserialize_graph(data: Dict[str, Any]) -> ConceptGraph:
        """Deserialize graph from dictionary"""
        graph = ConceptGraph()
        
        # First pass: create all nodes
        nodes = {}
        for node_data in data.get('nodes', []):
            node = ConceptSerializer.deserialize_node(node_data['data'])
            nodes[node_data['name']] = node
            graph.add_concept_node(node)
        
        # Second pass: add edges
        for edge_data in data.get('edges', []):
            source_node = nodes.get(edge_data['source'])
            target_node = nodes.get(edge_data['target'])
            
            if source_node and target_node:
                rel_data = edge_data['relationship']
                relationship = ConceptRelationship(
                    concept1=source_node.primary_concept,
                    concept2=target_node.primary_concept,
                    relation=rel_data.get('relation', 'related_to'),
                    definition=rel_data.get('definition', ''),
                    strength=rel_data.get('strength', 1.0),
                    attributes=rel_data.get('attributes', {})
                )
                
                from src.core.knowledge.models.relationship import ConceptNodeRelationship
                node_rel = ConceptNodeRelationship(source_node, target_node, relationship)
                graph.add_relationship(source_node, target_node, node_rel)
        
        return graph