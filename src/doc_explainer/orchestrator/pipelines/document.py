from doc_explainer.orchestrator import pipeline, registry
from doc_explainer.orchestrator.steps.build_tree import build_tree_step
from doc_explainer.orchestrator.steps.index import index_step
from doc_explainer.orchestrator.steps.parse import parse_document_step


@pipeline
def doc_ingestion_pipeline(file_path: str, target_query: str = None, force_reprocess: bool = False):
    # Check if already processed
    if not force_reprocess:
        existing = registry.get_processed(file_path)
        if existing:
            doc_id, run_id = existing
            # We could return early, but we want to keep the DAG structure.
            # We'll return the existing info from the first step.
            # We'll just pass the doc_id to subsequent steps, which will skip if already done.
            # But we need to fetch doc_id; we can use a special step that returns existing data.
            # For simplicity, we'll still run the steps; they will detect existing artifacts.
            pass

    # Step 1
    parse_result = parse_document_step(file_path)
    doc_id = parse_result["doc_id"]

    # Step 2
    tree_result = build_tree_step(doc_id, target_query)

    # Step 3
    index_result = index_step(tree_result["tree_id"])

    # After successful run, mark in registry
    # We can't easily mark here because we need run_id, which is not available in pipeline.
    # We'll mark in the calling code.

    return {
        "doc_id": doc_id,
        "tree_id": tree_result["tree_id"],
        "full_db_path": index_result["full_db_path"],
        "tree_db_path": index_result["tree_db_path"]
    }
