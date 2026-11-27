"""Asks the LLM what concepts and relations a section contains, given what the document has already covered.

Real model output is messy, so everything it returns is validated and clamped here: unknown
fields are ignored, wrong types dropped, paragraph numbers bounds-checked, sizes capped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..common.knowledge import Concept, MentionRole, RelationType
from ..common.ports import JsonLLM
from .steps import Step

MAX_PARAGRAPH_CHARS = 1800
_STOP_NAMES = {"this", "that", "these", "those", "figure", "table", "section", "chapter", "page", "example",
               "introduction", "summary", "conclusion", "text", "document", "thing", "way", "one", "it"}

SYSTEM = """You build a knowledge graph of a document, one section at a time, in reading order.
You are given the concepts already introduced earlier in the document and the text of the current section.
Reply with ONE JSON object and nothing else.

Rules:
- A concept is a technical term, theory, method, quantity, named entity or idea the text actually discusses or relies on. Never generic words ("example", "figure", "this").
- "name": short canonical form, singular, lowercase unless it is a proper noun.
- If the text discusses a KNOWN concept, set "known_id" to its id. Never invent ids. Use null for new concepts.
- "role": "defines" if the text explains what it is, "uses" if it relies on it, "example" if it is an instance used to illustrate.
- "definition": one sentence taken or condensed from the text when role is "defines", else null.
- "paragraphs": the numbers of the paragraphs where it appears.
- "importance": 1 (passing mention), 2 (discussed), 3 (central to this section).
- Relations only when the text shows them. "prerequisite_of": the source must be understood BEFORE the target. Others: "part_of", "example_of", "contrasts_with", "related_to".
- At most {max_concepts} concepts."""

USER = """DOCUMENT: {title}
SECTION: {path}

KNOWN CONCEPTS (introduced earlier):
{known}

TEXT OF THIS SECTION (numbered paragraphs):
{paragraphs}

Return JSON of this shape:
{{"concepts": [{{"name": "...", "known_id": null, "aliases": [], "role": "defines", "definition": "...", "paragraphs": [1], "importance": 2}}],
  "relations": [{{"source": "name", "target": "name", "type": "prerequisite_of", "paragraph": 1}}]}}"""


@dataclass
class RawConcept:
    name: str
    role: MentionRole = MentionRole.USES
    aliases: List[str] = field(default_factory=list)
    definition: str = ""
    paragraphs: List[int] = field(default_factory=list)      # 1-based, validated
    importance: int = 1
    known_id: Optional[str] = None


@dataclass
class RawRelation:
    source: str
    target: str
    type: RelationType
    paragraph: Optional[int] = None


@dataclass
class Extraction:
    concepts: List[RawConcept] = field(default_factory=list)
    relations: List[RawRelation] = field(default_factory=list)


_SNAKE_CASE = re.compile(r"[A-Za-z]{3,}(?:_[A-Za-z]{3,})+")


def _clean_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    name = re.sub(r"\s+", " ", value).strip().strip("\"'`.,;:")
    if _SNAKE_CASE.fullmatch(name):                    # "fourier_neural_operator" -> words; maths like "w_ij" is left alone
        name = name.replace("_", " ")
    return name[:80]


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        return int(m.group()) if m else None
    return None


class ConceptExtractor:
    def __init__(self, llm: JsonLLM, max_concepts: int = 25, max_known: int = 120):
        self.llm, self.max_concepts, self.max_known = llm, max_concepts, max_known

    # ------------------------------------------------------------------ prompt
    def build_prompt(self, step: Step, known: Sequence[Concept]) -> tuple:
        shown = sorted(known, key=lambda c: (-c.importance, -c.introduced_ordinal))[: self.max_known]
        lines = [f"{c.id} | {c.name} | {(c.definition or '').strip()[:90]}" for c in sorted(shown, key=lambda c: c.introduced_ordinal)]
        paragraphs = "\n".join(f"[{i}] {text[:MAX_PARAGRAPH_CHARS]}" for i, (_, _, text) in enumerate(step.paragraphs, 1))
        system = SYSTEM.format(max_concepts=self.max_concepts)
        user = USER.format(title=step.path.split(" > ")[0] if step.path else "", path=step.path or step.title,
                           known="\n".join(lines) or "(none yet: this is the beginning of the document)",
                           paragraphs=paragraphs)
        return system, user

    # ------------------------------------------------------------------ call + validate
    def extract(self, step: Step, known: Sequence[Concept]) -> Extraction:
        system, user = self.build_prompt(step, known)
        raw = self.llm.ask_json(system, user)     # an empty/cut-off reply is retried with a bigger budget by the client
        return self.parse(raw, len(step.paragraphs))

    def parse(self, raw: Any, n_paragraphs: int) -> Extraction:
        if not isinstance(raw, dict):
            return Extraction()
        by_name: Dict[str, RawConcept] = {}
        for item in raw.get("concepts") or []:
            if not isinstance(item, dict):
                continue
            name = _clean_name(item.get("name"))
            if len(name) < 2 or name.lower() in _STOP_NAMES or name.replace(".", "").isdigit():
                continue
            definition = item.get("definition")
            definition = re.sub(r"\s+", " ", definition).strip()[:300] if isinstance(definition, str) else ""
            role_text = str(item.get("role") or "").lower()
            role = MentionRole(role_text) if role_text in MentionRole._value2member_map_ else (
                MentionRole.DEFINES if definition else MentionRole.USES)
            paragraphs = sorted({p for p in (_as_int(x) for x in (item.get("paragraphs") or []) if x is not None)
                                 if p is not None and 1 <= p <= n_paragraphs})
            known_id = item.get("known_id")
            rc = RawConcept(
                name=name, role=role, definition=definition,
                aliases=[a for a in (_clean_name(x) for x in (item.get("aliases") or [])[:5]) if len(a) >= 2],
                paragraphs=paragraphs or ([1] if n_paragraphs else []),
                importance=min(max(_as_int(item.get("importance")) or 1, 1), 3),
                known_id=known_id.strip() if isinstance(known_id, str) and known_id.strip() else None)
            key = name.lower()
            if key in by_name:                                    # the model listed the same concept twice: merge
                prev = by_name[key]
                prev.paragraphs = sorted(set(prev.paragraphs) | set(rc.paragraphs))
                prev.aliases = list(dict.fromkeys(prev.aliases + rc.aliases))
                prev.importance = max(prev.importance, rc.importance)
                prev.known_id = prev.known_id or rc.known_id
                if rc.role is MentionRole.DEFINES and rc.definition:
                    prev.role, prev.definition = rc.role, prev.definition or rc.definition
            else:
                by_name[key] = rc
        concepts = sorted(by_name.values(), key=lambda c: -c.importance)[: self.max_concepts]
        relations: List[RawRelation] = []
        for item in raw.get("relations") or []:
            if not isinstance(item, dict):
                continue
            src, dst = _clean_name(item.get("source")), _clean_name(item.get("target"))
            kind = re.sub(r"[\s-]+", "_", str(item.get("type") or "").strip().lower())
            if not src or not dst or kind not in RelationType._value2member_map_:
                continue
            p = _as_int(item.get("paragraph"))
            relations.append(RawRelation(src, dst, RelationType(kind), p if p and 1 <= p <= n_paragraphs else None))
        return Extraction(concepts, relations[: self.max_concepts * 3])
