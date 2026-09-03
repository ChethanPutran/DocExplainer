from typing import Callable, Optional
from .step import Step, get_current_context, push_context, pop_context
from .dag import DAG
from .node import Node
import logging


class Pipeline:
    """Represents a pipeline definition."""
    def __init__(self, fn: Callable, name: Optional[str] = None,logger: Optional[logging.Logger] = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def compile(self) -> DAG:
        """Execute the pipeline function to build the DAG."""
        from .context import PipelineContext
        ctx = PipelineContext()
        push_context(ctx)
        try:
            self.fn()
        finally:
            pop_context()
        nodes = list(ctx.nodes.values())
        return DAG(nodes)

    def run(self, **run_kwargs):
        """Run the pipeline using the default execution engine."""
        from .execution.engine import ExecutionEngine
        from .execution.local import LocalExecutor
        from .artifacts.local import LocalArtifactStore
        from .metadata.sqlite import SQLiteMetadataStore

        dag = self.compile()

        self.logger.info(f"Running pipeline '{self.name}' with {dag.nodes}")
        artifact_store = LocalArtifactStore(base_dir="./db/artifacts")
        metadata_store = SQLiteMetadataStore(db_path="./db/metadata/metadata.db")
        engine = ExecutionEngine(
            executor=LocalExecutor(),
            artifact_store=artifact_store,
            metadata_store=metadata_store
        )
        return engine.run(dag, pipeline_name=self.name)


    def get_result(self, run_id: str):
        from .metadata.sqlite import SQLiteMetadataStore
        from .artifacts.local import LocalArtifactStore
        from .metadata.models import StepRun
        import json

        metadata_store = SQLiteMetadataStore(db_path="./db/metadata/metadata.db")
        artifact_store = LocalArtifactStore(base_dir="./db/artifacts")

        session = metadata_store.Session()
        last_step = session.query(StepRun).filter_by(pipeline_run_id=run_id).order_by(StepRun.started_at.desc()).first()
        session.close()

        if last_step and last_step.artifact_ref_json:
            ref_dict = json.loads(last_step.artifact_ref_json)
            from .artifacts.artifact import ArtifactRef
            ref = ArtifactRef(**ref_dict)
            return artifact_store.load(ref)
        return None

    

def pipeline(fn: Callable = None, *, name: Optional[str] = None):
    """Decorator to convert a function into a Pipeline."""
    def decorator(func):
        return Pipeline(func, name=name)
    if fn is None:
        return decorator
    return decorator(fn)