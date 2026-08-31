from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class LocalExecutor:
    """Executes a step function in-process."""
    def execute(self, step_fn, inputs: Dict[str, Any], **kwargs) -> Any:
        # inputs are already resolved (actual values, not Nodes)
        return step_fn(**inputs)