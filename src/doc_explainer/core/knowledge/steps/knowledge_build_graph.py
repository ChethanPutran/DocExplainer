from ...common.step import step

@step(annotations={"stage": "build_graph", "retries": 2})
def build_knowledge_graph_step(registry, doc_id: str) -> dict:
    """Build the full concept graph for a document."""
    tree = registry.repository.get_tree(doc_id)
    if not tree:
        raise ValueError(f"Tree not found for doc {doc_id}")

    graph_builder = registry.concept_graph_builder
    graph = graph_builder.build_from_document(tree)

    # Store graph in repository
    registry.knowledge_store.save_graph(graph, doc_id)
    registry.artifact_store.save(graph, key=f"graph_{doc_id}")

    return {"doc_id": doc_id, "node_count": len(graph.graph.nodes)}