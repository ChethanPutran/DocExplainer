import os
from ...common.step import step

@step(annotations={"stage": "index", "retries": 2})
def index_step(registry,tree_id: str, persist_dir: str = "./db/vectors") -> dict:
    """Create vector databases from the tree."""
    tree = registry.artifact_store.load(tree_id)
    if not tree:
        # Try loading from repository
        doc_id = tree_id.replace("tree_", "")
        tree = registry.repository.get_tree(doc_id)
        if not tree:
            raise ValueError(f"Tree {tree_id} not found")

    # Check if vector DBs already exist
    full_db_path = f"{persist_dir}/{tree_id}/full_db"
    tree_db_path = f"{persist_dir}/{tree_id}/tree_db"
    if os.path.exists(full_db_path) and os.path.exists(tree_db_path):
        return {
            "full_db_path": full_db_path,
            "tree_db_path": tree_db_path,
            "existing": True
        }

    # Create vector DBs using processor
    processor = registry.processor
    if not processor.langchain_embeddings:
        raise ValueError("Embedding model required for indexing")

    # For full DB we need the original document. We can get it from repository.
    doc_id = tree_id.replace("tree_", "")
    document = registry.repository.get_document(doc_id)
    if not document:
        raise ValueError(f"Document {doc_id} not found")

    full_db = processor.create_full_vector_db(document, persist_directory=full_db_path)
    tree_db = processor.create_tree_aware_db(tree, persist_directory=tree_db_path)

    # Save paths as artifacts
    registry.artifact_store.save(full_db_path, key=f"{tree_id}_full_db_path")
    registry.artifact_store.save(tree_db_path, key=f"{tree_id}_tree_db_path")

    return {
        "full_db_path": full_db_path,
        "tree_db_path": tree_db_path,
        "existing": False
    }