
from typing import Optional, Protocol

from doc_explainer.core.document.models.base import ProcessedSection


class GraphStore(Protocol):

    def add_section(
        self,
        section: ProcessedSection
    ) -> None:
        ...

    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str
    ) -> None:
        ...

    def get_section(
        self,
        section_id: str
    ) -> Optional[ProcessedSection]:
        ...