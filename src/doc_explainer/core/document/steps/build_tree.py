from ...common.step import step

@step(annotations={"stage": "build_tree", "retries": 2})
def build_tree_step(registry, doc_id: str, target_query: str = None) -> dict:
    """Build document tree with summaries, processing sections incrementally."""
    # Check if tree already built
    tree = registry.repository.get_tree(doc_id)
    if tree:
        return {
            "doc_id": doc_id,
            "tree_id": f"tree_{doc_id}",
            "num_chunks": tree.total_chunks,
            "existing": True
        }

    # Get document
    document = registry.repository.get_document(doc_id)
    if not document:
        raise ValueError(f"Document {doc_id} not found")

    # Build tree using engine (which internally processes sections)
    # The engine's ingest_and_map uses the processor, which builds the tree.
    # To avoid memory, we can modify the processor to build the tree incrementally.
    # For now, we call the existing method; it may load everything.
    tree = registry.engine.ingest_and_map(document, target_query)

    # Save tree to repository and artifact store
    registry.repository.save_tree(tree, doc_id)
    registry.artifact_store.save(tree, key=f"tree_{doc_id}")

    return {
        "doc_id": doc_id,
        "tree_id": f"tree_{doc_id}",
        "num_chunks": tree.total_chunks,
        "existing": False
    }