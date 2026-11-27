"""Pointing a reader at what will help: first, the parts of THIS document that teach what they are missing."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..common.enums import ExplanationLevel, ResourceType
from ..common.explanation import Resource, ResourceSuggestion
from ..common.knowledge import Concept
from ..common.structure import DocumentStructure


def in_document_resources(doc_id: str, structure: DocumentStructure, gaps: Sequence[Concept],
                          prerequisites: Dict[str, List[Concept]], current_section_id: str, limit: int = 4
                          ) -> Tuple[List[ResourceSuggestion], List[Resource]]:
    """Earlier sections that introduce the concepts the reader lacks, or what those concepts rest on.

    Only sections BEFORE the reader's current one are suggested: a pointer forward would spoil the document."""
    current = structure.node(current_section_id)
    by_section: Dict[str, List[Concept]] = {}
    for gap in gaps:
        for concept in [gap, *prerequisites.get(gap.id, [])]:
            if not concept.first_node_id or concept.first_node_id not in structure:
                continue
            section = structure.section_of(concept.first_node_id)
            if section is None or section.ordinal >= current.ordinal:
                continue
            names = by_section.setdefault(section.id, [])
            if concept not in names:
                names.append(concept)
    suggestions: List[ResourceSuggestion] = []
    resources: List[Resource] = []
    for section_id, concepts in sorted(by_section.items(), key=lambda kv: (-len(kv[1]), structure.node(kv[0]).ordinal))[:limit]:
        section = structure.node(section_id)
        path = " > ".join(structure.section_path(section_id)) or section.title
        names = ", ".join(c.name for c in concepts[:3])
        resources.append(Resource(title=path, url=f"doc://{doc_id}#{section_id}", type=ResourceType.SECTION,
                                  description=f"Introduces {names} (page {section.page_start})", difficulty=ExplanationLevel.BEGINNER,
                                  source="This document"))
        suggestions.extend(ResourceSuggestion(concept=c.name, resource_type=ResourceType.SECTION, difficulty=ExplanationLevel.BEGINNER)
                           for c in concepts[:3])
    return suggestions, resources
