from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Callable


class ProgressStatus(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProgressEvent:
    job_id: str
    document_id: Optional[str]
    step: str
    status: ProgressStatus
    progress: float
    message: str
    error: Optional[str] = None


class ProgressReporter(Protocol):

    def report(self, event: ProgressEvent) -> None:
        ...


class CallbackProgressReporter:

    def __init__(
        self,
        callback: Callable[[ProgressEvent], None],
    ):
        self.callback = callback

    def report(self, event: ProgressEvent) -> None:
        self.callback(event)