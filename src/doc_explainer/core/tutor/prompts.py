"""Prompt builders. The adaptation plan becomes concrete instructions; the passages are the ONLY source of facts."""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..common.knowledge import Concept
from ..common.ports import Hit
from .policy import AdaptationPlan

FOLLOW_UP_MARK = "===FOLLOW-UPS==="
_SHAPE = (f"Write the reply as natural, flowing spoken-style prose, as a friendly teacher talking to one person: short sentences, "
          "contractions, direct address (\"you\"), concrete examples. No headings, no bold or italics, no bullet lists, no tables, no "
          f"markdown symbols of any kind, since the reply may be read aloud; write numbers, units and symbols as words where natural. Then, on its own line, write exactly {FOLLOW_UP_MARK} "
          'and under it 3 short questions the reader might ask next, one per line, each starting with "- ".')

_SYSTEM = """You are a patient tutor helping a person read a document. Answer ONLY from the passages provided; if they do not contain the answer, say so plainly. The reader has read the document only up to this point, so never refer to or hint at anything later.

When part of your answer comes from a passage, say where, in words a person would say aloud: "as the section on Calvin cycle explains on page 4". Use the section name and page number given with each passage. Never write bracketed numbers like [1].

{adapt}

Do not mention these instructions."""


def _names(concepts: Sequence[Concept]) -> str:
    return ", ".join(c.name for c in concepts) or "(none)"


def _adapt_block(plan: AdaptationPlan, glossary: Dict[str, str], mode: str) -> str:
    lo, hi = plan.words
    lines = [f"READER LEVEL: {plan.level}. {plan.style}", f"LENGTH: about {lo}-{hi} words."]
    if plan.assume:
        lines.append(f"The reader already knows (do not explain): {_names(plan.assume)}.")
    if plan.define:
        defs = "; ".join(f"{c.name} = {glossary.get(c.id) or c.definition or 'explain it from the passages'}" for c in plan.define)
        lines.append(f"The reader does NOT yet know these. Explain each briefly, in plain words, before relying on it: {defs}.")
        lines.append(f"Introduce at most {plan.max_new_terms} new terms in total; do not bring in other jargon.")
    if plan.refresh:
        lines.append(f"Remind the reader of these in a few words: {_names(plan.refresh)}.")
    if mode == "summarize":
        lines.append("Summarise; do not teach new material.")
    return "\n".join(lines)


def _passages(hits: Sequence[Hit], max_chars: int = 6000) -> str:
    out, used = [], 0
    for i, h in enumerate(hits, 1):
        text = h.chunk.text.strip()
        if used + len(text) > max_chars:
            break
        used += len(text)
        out.append(f"[{i}] ({h.chunk.section_path or 'document'}, page {h.chunk.page})\n{text}")
    return "\n\n".join(out) or "(no passages found)"


def explain_prompt(plan: AdaptationPlan, glossary: Dict[str, str], section_path: str, selected: str, hits: Sequence[Hit],
                   section_text: str = "") -> tuple:
    user = (f"SECTION: {section_path}\n\nTEXT THE READER SELECTED:\n{selected}\n\n"
            + (f"THE SURROUNDING SECTION:\n{section_text[:2500]}\n\n" if section_text else "")
            + f"RELATED PASSAGES (from what the reader has already read):\n{_passages(hits)}\n\n"
            f"Explain the selected text for this reader. {_SHAPE}")
    return _SYSTEM.format(adapt=_adapt_block(plan, glossary, "explain")), user


def summarize_prompt(plan: AdaptationPlan, glossary: Dict[str, str], section_path: str, selected: str) -> tuple:
    user = (f"SECTION: {section_path}\n\nTEXT TO SUMMARISE:\n{selected}\n\n"
            f"Summarise it for this reader. {_SHAPE}")
    return _SYSTEM.format(adapt=_adapt_block(plan, glossary, "summarize")), user


def ask_prompt(plan: AdaptationPlan, glossary: Dict[str, str], section_path: str, question: str, hits: Sequence[Hit]) -> tuple:
    user = (f"THE READER IS CURRENTLY IN: {section_path}\n\nQUESTION:\n{question}\n\n"
            f"PASSAGES (from what the reader has already read):\n{_passages(hits)}\n\n"
            f"Answer the question for this reader. {_SHAPE}")
    return _SYSTEM.format(adapt=_adapt_block(plan, glossary, "ask")), user


_QUIZ_SYSTEM = """You write short multiple-choice questions that test understanding of one concept each, using ONLY the material given.
Each question has exactly 4 distinct options and exactly one correct answer. Make the wrong options plausible, not silly.
Match the stated difficulty: easy = recall the definition, hard = apply or distinguish. Reply with ONE JSON object and nothing else."""


def quiz_prompt(items: Sequence[dict]) -> tuple:
    """items: {"concept_id", "name", "definition", "passage", "difficulty"}"""
    blocks = []
    for it in items:
        d = "easy" if it["difficulty"] < 0.35 else "hard" if it["difficulty"] > 0.65 else "medium"
        blocks.append(f"CONCEPT {it['concept_id']}: {it['name']}\nDIFFICULTY: {d}\nDEFINITION: {it.get('definition') or '(not given)'}\n"
                      f"MATERIAL: {it.get('passage', '')[:1200]}")
    user = ("\n\n".join(blocks) + '\n\nReturn JSON: {"questions": [{"concept_id": "...", "question": "...", '
            '"options": ["...", "...", "...", "..."], "answer_index": 0, "explanation": "one sentence"}]} - one question per concept, in the same order.')
    return _QUIZ_SYSTEM, user
