"""Where to learn a concept on the internet: one strategy per kind of resource.

Each strategy builds a SEARCH link (YouTube, Wikipedia, Khan Academy, Google ...). A link to a search page always
works; a link the model "remembers" to one particular video may not exist. No network call is made here.
New kinds are added by writing a strategy and appending it to ``DEFAULT_STRATEGIES``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence
from urllib.parse import quote_plus

from ..common.enums import ExplanationLevel, ResourceType
from ..common.explanation import Resource

_LEVEL_WORDS = {ExplanationLevel.BEGINNER: "for beginners", ExplanationLevel.INTERMEDIATE: "", ExplanationLevel.ADVANCED: "advanced", ExplanationLevel.EXPERT: "advanced"}


class ResourceStrategy(ABC):
    kind: ResourceType
    source: str

    @abstractmethod
    def build(self, concept: str, level: ExplanationLevel, context: str = "") -> Resource: ...

    def _query(self, concept: str, level: ExplanationLevel, *extra: str, context: str = "") -> str:
        return " ".join(p for p in (concept, context, *extra, _LEVEL_WORDS.get(level, "")) if p).strip()


class VideoStrategy(ResourceStrategy):
    kind, source = ResourceType.VIDEO, "YouTube"

    def build(self, concept, level, context=""):
        return Resource(title=f"Video lectures: {concept}", type=self.kind, source=self.source, difficulty=level,
                        url="https://www.youtube.com/results?search_query=" + quote_plus(self._query(concept, level, "lecture explained", context=context)),
                        description=f"Lectures and explainers on {concept}.")


class ReferenceStrategy(ResourceStrategy):
    kind, source = ResourceType.DOCUMENTATION, "Wikipedia"

    def build(self, concept, level, context=""):
        return Resource(title=f"Encyclopedia: {concept}", type=self.kind, source=self.source, difficulty=level,
                        url="https://en.wikipedia.org/w/index.php?search=" + quote_plus(concept),
                        description=f"A neutral overview of {concept}.")


class ArticleStrategy(ResourceStrategy):
    """Papers for readers who want depth, plain tutorials for everyone else."""
    kind = ResourceType.ARTICLE

    def __init__(self):
        self.source = "Google"

    def build(self, concept, level, context=""):
        if level in (ExplanationLevel.ADVANCED, ExplanationLevel.EXPERT):
            return Resource(title=f"Papers: {concept}", type=self.kind, source="Google Scholar", difficulty=level,
                            url="https://scholar.google.com/scholar?q=" + quote_plus(self._query(concept, level, context=context)),
                            description=f"Research papers about {concept}.")
        return Resource(title=f"Articles: {concept}", type=self.kind, source="Google", difficulty=level,
                        url="https://www.google.com/search?q=" + quote_plus(self._query(concept, level, "tutorial explained", context=context)),
                        description=f"Tutorials and explanations of {concept}.")


class CourseStrategy(ResourceStrategy):
    kind, source = ResourceType.COURSE, "Khan Academy"

    def build(self, concept, level, context=""):
        return Resource(title=f"Courses: {concept}", type=self.kind, source=self.source, difficulty=level,
                        url="https://www.khanacademy.org/search?search_again=1&page_search_query=" + quote_plus(concept),
                        description=f"Structured lessons on {concept}.")


class ExerciseStrategy(ResourceStrategy):
    kind, source = ResourceType.EXERCISE, "Google"

    def build(self, concept, level, context=""):
        return Resource(title=f"Practice: {concept}", type=self.kind, source=self.source, difficulty=level,
                        url="https://www.google.com/search?q=" + quote_plus(self._query(concept, level, "practice problems exercises quiz", context=context)),
                        description=f"Exercises to test your understanding of {concept}.")


DEFAULT_STRATEGIES: List[ResourceStrategy] = [VideoStrategy(), ReferenceStrategy(), ArticleStrategy(), CourseStrategy(), ExerciseStrategy()]


def web_resources(concepts: Sequence[str], level: ExplanationLevel, kinds: Optional[Sequence[ResourceType]] = None,
                  context: str = "", per_concept: int = 3, strategies: Optional[Sequence[ResourceStrategy]] = None) -> List[Resource]:
    """Links for each concept (video first, then a reference, then the rest), at most ``per_concept`` each."""
    chosen = [s for s in (strategies or DEFAULT_STRATEGIES) if kinds is None or s.kind in kinds]
    out: List[Resource] = []
    for name in concepts:
        name = name.strip()
        if name:
            out.extend(s.build(name, level, context) for s in chosen[:per_concept])
    return out
