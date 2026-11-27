"""Shared knowledge-graph data types (data only; the logic lives in ``core.knowledge``).

The graph describes ONE document, independent of any reader, and is built along the document's
reading order. Everything is stamped with the ordinal at which it first appeared, so the graph "as
of" any point is just a filter, and nothing can depend on something the reader has not reached yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class RelationType(str, Enum):
    PREREQUISITE_OF = "prerequisite_of"   # src must be understood before dst (directed, only ever earlier -> later)
    PART_OF = "part_of"                   # src is a part/component of dst (directed)
    EXAMPLE_OF = "example_of"             # src is an example of dst (directed)
    CONTRASTS_WITH = "contrasts_with"     # symmetric
    RELATED_TO = "related_to"             # symmetric

    @property
    def symmetric(self) -> bool:
        return self in (RelationType.CONTRASTS_WITH, RelationType.RELATED_TO)


class MentionRole(str, Enum):
    DEFINES = "defines"
    USES = "uses"
    EXAMPLE = "example"


@dataclass
class Concept:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    introduced_step: int = 0
    introduced_ordinal: int = 0           # reading-order position where it first appears
    first_node_id: str = ""
    definition: str = ""                  # the first definition given in the text, if any
    definition_node_id: str = ""
    defined_ordinal: Optional[int] = None
    importance: int = 0                   # accumulated weight: how much the text dwells on it
    mention_count: int = 0


@dataclass(frozen=True)
class Mention:
    concept_id: str
    node_id: str
    ordinal: int
    step: int
    role: MentionRole


@dataclass
class Relation:
    src: str
    dst: str
    type: RelationType
    first_ordinal: int
    first_step: int
    strength: int = 1                     # how many times the text supports it
    evidence: List[str] = field(default_factory=list)   # node ids

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.src, self.dst, self.type.value)


@dataclass
class KnowledgeDelta:
    """What one step of reading added to the graph. The deltas, in order, ARE the graph's history."""
    step: int
    section_id: str
    ordinal_start: int
    ordinal_end: int
    new_concepts: List[Concept] = field(default_factory=list)
    updated_aliases: List[Tuple[str, List[str]]] = field(default_factory=list)       # (concept id, new aliases)
    definitions: List[Tuple[str, int, str, str]] = field(default_factory=list)      # (concept id, ordinal, text, node id)
    mentions: List[Mention] = field(default_factory=list)
    new_relations: List[Relation] = field(default_factory=list)
    strengthened: List[Tuple[str, str, str, str]] = field(default_factory=list)     # (src, dst, type, evidence node)
    rejected: List[str] = field(default_factory=list)                                # why something was dropped
    importance: List[Tuple[str, int]] = field(default_factory=list)                 # (concept id, weight added)
    words: int = 0


@dataclass
class StepRecord:
    step: int
    section_id: str
    ordinal_start: int
    ordinal_end: int
    words: int
    status: str                           # done | failed
    new_concepts: int = 0
    new_relations: int = 0
    mentions: int = 0
    known_mentions: int = 0               # mentions of concepts introduced earlier (reuse)
    error: str = ""
