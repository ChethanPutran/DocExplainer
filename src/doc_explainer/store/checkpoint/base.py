from typing import Optional, Protocol


class CheckpointStore(Protocol):

    # ============================================================
    # Section lifecycle
    # ============================================================

    def mark_started(
        self,
        namespace: str,
        section_id: str,
    ) -> None:
        ...

    def mark_completed(
        self,
        namespace: str,
        section_id: str,
    ) -> None:
        ...

    def mark_section_failed(
        self,
        namespace: str,
        section_id: str,
        error: str,
    ) -> None:
        ...

    def is_completed(
        self,
        namespace: str,
        section_id: str,
    ) -> bool:
        ...

    def get_last_completed(
        self,
        namespace: str,
    ) -> Optional[str]:
        ...

    def is_run_complete(self, namespace: str) -> bool:
        ...

    def mark_registration_complete(self, namespace: str) -> None:
        ...

    def is_registration_complete(self, namespace: str) -> bool:
        ...

    # ============================================================
    # Document / namespace lifecycle
    # ============================================================

    def start(
        self,
        namespace: str,
        file_path: str,
    ) -> None:
        ...

    def complete(
        self,
        namespace: str,
    ) -> None:
        ...