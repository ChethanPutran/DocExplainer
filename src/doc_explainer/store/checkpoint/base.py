from typing import Optional, Protocol


class CheckpointStore(Protocol):

    def mark_started(
        self,
        document_id: str,
        section_id: str
    ) -> None:
        ...

    def mark_completed(
        self,
        document_id: str,
        section_id: str
    ) -> None:
        ...

    def is_completed(
        self,
        document_id: str,
        section_id: str
    ) -> bool:
        ...

    def get_last_completed(
        self,
        document_id: str
    ) -> Optional[str]:
        ...