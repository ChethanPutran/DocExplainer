import threading
import time
import logging
from typing import Dict, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from doc_explainer.store.checkpoint.base import CheckpointStore
from ..base.dag import DAG
from ..base.node import Node
from ..artifacts.store import ArtifactStore
from ..metadata.sqlite import SQLiteMetadataStore
from ..metadata.models import RunStatus, StepStatus
from ..utils.hash import compute_hash
import pickle

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(
        self,
        executor,
        artifact_store: ArtifactStore,
        metadata_store: SQLiteMetadataStore,
        checkpoint_store: Optional[CheckpointStore] = None,
        max_workers: int = 4
    ):
        self.executor = executor
        self.artifact_store = artifact_store
        self.metadata_store = metadata_store
        self.checkpoint_store = checkpoint_store
        self.max_workers = max_workers
        self._results: Dict[str, Any] = {}  # node_id -> output value (or ArtifactRef)
        self._lock = threading.Lock()

    def run(self, dag: DAG, pipeline_name: str) -> str:
        run_id = self.metadata_store.create_pipeline_run(pipeline_name)
        sorted_nodes = dag.topological_sort()

        # Register step runs in metadata
        for node in sorted_nodes:
            self.metadata_store.create_step_run(run_id, node.id, node.step_name)

        pending: Set[str] = {n.id for n in sorted_nodes}
        ready: Set[str] = set()
        futures = {}
        node_map = {n.id: n for n in sorted_nodes}
        self._results: Dict[str, Any] = {}   # shared across threads
        self._lock = threading.Lock()        # protects _results

        # Initial roots
        for root in dag.get_source_nodes():
            ready.add(root.id)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            pipeline_failed = False

            while pending and not pipeline_failed:
                # Submit all ready nodes not yet submitted
                to_submit = ready - set(futures.keys()) # Set operation to find nodes that are ready but not yet submitted
                for node_id in to_submit:
                    node = node_map[node_id]
                    logger.info(f"Submitting node {node_id} ({node_map[node_id].step_name})")
                    future = pool.submit(self._execute_node, node, run_id)
                    futures[node_id] = future

                # Remove submitted nodes from ready to avoid resubmission
                ready.difference_update(to_submit)

                # Process completed futures
                completed = [fid for fid, fut in futures.items() if fut.done()]
                for node_id in completed:
                    logger.info(f"Node {node_id} completed successfully")
                    future = futures.pop(node_id)
                    try:
                        result = future.result()
                        with self._lock:
                            self._results[node_id] = result
                        pending.remove(node_id)
                        self.metadata_store.update_step_run(
                            node_id, status=StepStatus.SUCCESS
                        )

                        # Add dependents if all their deps are done
                        for dep_node in dag.get_dependents(node_id):
                            dep_id = dep_node.id
                            if dep_id in pending:
                                deps = dep_node.dependencies
                                with self._lock:
                                    # Check if all dependencies are completed
                                    if all(d in self._results for d in deps):
                                        ready.add(dep_id)

                    except Exception as e:
                        logger.error(f"Node {node_id} failed: {e}")
                        self.metadata_store.update_step_run(
                            node_id, status=StepStatus.FAILED, error=str(e)
                        )
                        # Remove from pending only if still present
                        if node_id in pending:
                            pending.remove(node_id)
                        logger.error(f"Pipeline failed due to node {node_id} failure")
                        pipeline_failed = True
                        break   # stop further execution

                if pipeline_failed:
                    break

                # Avoid busy loop when nothing is happening
                if not futures and pending and not ready:
                    logger.error("Deadlock detected – no nodes ready but pending remain")
                    pipeline_failed = True
                    break

                time.sleep(0.1)   # small yield

        # Final status
        if pipeline_failed:
            self.metadata_store.update_pipeline_run(run_id, status=RunStatus.FAILED)
        else:
            self.metadata_store.update_pipeline_run(run_id, status=RunStatus.SUCCESS)

        return run_id
    
    def _execute_node(self, node: Node, run_id: str):
        logger.info(f"Executing node {node.id} ({node.step_name})")
        from ..base.step import Step
        step = getattr(node, 'step', None)
        if step is None:
            raise ValueError(f"Node {node.id} missing step object")

        # Check cache – returns ArtifactRef or None
        cache_key = self._compute_cache_key(node)
        cached_ref = self.metadata_store.get_cached_artifact(cache_key)
        if cached_ref:
            # Load the actual value from the artifact store
            value = self.artifact_store.load(cached_ref)
            self.metadata_store.update_step_run(
                node.id, status=StepStatus.CACHED, artifact_id=cached_ref.id
            )
            return value  

        # Resolve inputs – they are now actual values because previous nodes returned them
        resolved_inputs = {}
        for arg_name, value in node.inputs.items():
            if isinstance(value, Node):
                # Retrieve from _results (which stores actual values)
                result = self._results.get(value.id)
                if result is None:
                    raise RuntimeError(f"Dependency {value.id}:{value.step_name} not executed yet")
                resolved_inputs[arg_name] = result
            else:
                resolved_inputs[arg_name] = value

        # Execute the step function
        result = self.executor.execute(step.fn, resolved_inputs)

        # Store artifact
        artifact_ref = self.artifact_store.save(
            result, step_name=step.name, run_id=run_id
        )
        # Update metadata
        self.metadata_store.update_step_run(
            node.id, status=StepStatus.SUCCESS, artifact_id=artifact_ref.id
        )
        # Save cache entry with the full ArtifactRef
        self.metadata_store.save_cache_entry(cache_key, artifact_ref)

        
        return result  

    def _compute_cache_key(self, node: Node) -> str:
        # Simple hash based on step name, inputs (as values), and code hash.
        # We'll compute a hash of the step function's source code (or module+name)
        import inspect
        try:
            source = inspect.getsource(node.step.fn)
        except:
            source = node.step.fn.__name__  # fallback
        inputs_hash = compute_hash(node.inputs)
        code_hash = compute_hash(source)
        return f"{node.step_name}:{code_hash}:{inputs_hash}"