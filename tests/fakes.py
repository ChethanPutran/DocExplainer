"""Deterministic stand-ins for the expensive parts, shared by the tests."""
import hashlib
import math
import re
from typing import List, Sequence

from doc_explainer.core.common.ports import LLMError


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector. Texts sharing words are close, so retrieval
    tests are meaningful, with no model to download. Test-only: the product never uses fake vectors."""
    name = "hashing-test"

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls += 1
        out = []
        for text in texts:
            v = [0.0] * self.dimension
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class ScriptedLLM:
    """A JsonLLM that answers from a script keyed by the section being read, and records every prompt."""

    def __init__(self, script):
        self.script = script          # section title -> dict (or an Exception to raise, or a callable)
        self.prompts = []
        self.fail = {}                # section title -> exception to raise instead (cleared by the test)

    def ask_json(self, system, user):
        self.prompts.append(user)
        title = next((l[len("SECTION: "):].split(" > ")[-1] for l in user.splitlines() if l.startswith("SECTION: ")), "")
        if title in self.fail:
            raise self.fail[title]
        answer = self.script.get(title, {"concepts": [], "relations": []})
        if isinstance(answer, Exception):
            raise answer
        return answer(user) if callable(answer) else answer


# ----------------------------------------------------------------------------- the sample book and what the 'model' says about it
BOOK = """# Photosynthesis

Photosynthesis converts light energy into chemical energy in plants.

It relies on chlorophyll, a green pigment.

## Light reactions

Chlorophyll absorbs light and splits water. The light reactions produce ATP.

## Calvin cycle

The Calvin cycle uses ATP from the light reactions to fix carbon dioxide into sugar.

## Cellular respiration

Cellular respiration releases the energy stored in sugar. It contrasts with photosynthesis.
"""


def c(name, role="uses", paragraphs=(1,), importance=2, known_id=None, definition=None, aliases=()):
    return {"name": name, "role": role, "paragraphs": list(paragraphs), "importance": importance,
            "known_id": known_id, "definition": definition, "aliases": list(aliases)}


def rel(src, dst, kind, paragraph=1):
    return {"source": src, "target": dst, "type": kind, "paragraph": paragraph}


SCRIPT = {
    "Photosynthesis": {"concepts": [c("photosynthesis", "defines", (1,), 3, definition="Making chemical energy from light."),
                                    c("chlorophyll", "defines", (2,), 2, definition="A green pigment.")],
                       "relations": [rel("chlorophyll", "photosynthesis", "part_of", 2),
                                     rel("chlorophyll", "photosynthesis", "prerequisite_of", 2)]},   # forward: must be rejected
    "Light reactions": {"concepts": [c("chlorophyll", known_id="c2"), c("light reactions", "defines", definition="The light-dependent stage."),
                                     c("ATP", "defines", definition="The cell's energy currency.")],
                        "relations": [rel("chlorophyll", "light reactions", "prerequisite_of"), rel("ATP", "light reactions", "part_of")]},
    "Calvin cycle": {"concepts": [c("Calvin cycle", "defines", definition="Sugar-making stage."), c("atp"), c("Light Reactions"),
                                  c("carbon dioxide"), c("sugar")],
                     "relations": [rel("light reactions", "calvin cycle", "prerequisite_of"),
                                   rel("calvin cycle", "light reactions", "prerequisite_of"),          # cycle: must be rejected
                                   rel("carbon dioxide", "sugar", "prerequisite_of")]},
    "Cellular respiration": {"concepts": [c("cellular respiration", "defines", definition="Releasing energy from sugar."),
                                          c("sugars"), c("photosynthesis", "defines", known_id="c1",
                                                         definition="The opposite of respiration, building sugar from light.")],
                             "relations": [rel("photosynthesis", "cellular respiration", "contrasts_with"),
                                           rel("sugar", "cellular respiration", "prerequisite_of"),
                                           rel("ghost", "sugar", "related_to")]},                      # unknown concept
}




class Brain:
    """Answers the three kinds of request the system makes, from what the prompt says."""

    def __init__(self, script=SCRIPT):
        self.script, self.calls, self.gate, self.fail_knowledge = script, [], None, False
        self.stream_gate, self.fail_reply, self.reply_suffix = None, False, ""

    def ask_json(self, system, user):
        self.calls.append((system, user))
        if "build a knowledge graph" in system:
            if self.fail_knowledge:
                raise LLMError("401 invalid api key")
            if self.gate is not None:
                self.gate.wait(10)
            title = next((l[len("SECTION: "):].split(" > ")[-1] for l in user.splitlines() if l.startswith("SECTION: ")), "")
            return self.script.get(title, {"concepts": [], "relations": []})
        if "multiple-choice" in system:
            ids = re.findall(r"^CONCEPT (\S+):", user, re.M)
            return {"questions": [{"concept_id": i.rstrip(":"), "question": f"About {i}?", "options": ["right", "w1", "w2", "w3"],
                                   "answer_index": 0, "explanation": "because"} for i in ids]}
        raise AssertionError("tutor prompts are streamed text now: use stream_text")

    def stream_text(self, system, user):
        """The tutor's reply, in small pieces, with the follow-up block after the marker."""
        self.calls.append((system, user))
        if self.stream_gate is not None:
            self.stream_gate.wait(10)
        if self.fail_reply:
            raise LLMError("the model is down")
        if "study notes" in system:            # ChatService.summarize
            yield "- gist of the conversation\n- " + user.count("READER:").__str__() + " question(s) asked"
            return
        level = re.search(r"READER LEVEL: (\w+)", system).group(1)
        define = re.search(r"Explain each briefly, in plain words, before relying on it: (.*?)\.\n", system + "\n")
        text = f"LEVEL={level}; DEFINE={define.group(1) if define else 'nothing'}" + self.reply_suffix + "\n===FOLLOW-UPS===\n- Why?\n"
        for i in range(0, len(text), 7):
            yield text[i:i + 7]
