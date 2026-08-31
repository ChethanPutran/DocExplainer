from doc_explainer.orchestrator import step
from doc_explainer.orchestrator.registry import registry

@step(annotations={"stage": "knowledge_extract", "retries": 2})
def extract_concepts_step(tree_id: str, section_id: str) -> dict:
    """Extract concepts from a section and store in the document tree."""
    # Load document tree from artifact store
    tree = registry.artifact_store.load(tree_id)
    if not tree:
        raise ValueError(f"Tree {tree_id} not found")

    # Get section
    section_node = tree.root.children.get(section_id)
    if not section_node:
        raise ValueError(f"Section {section_id} not found")

    # Use the graph builder to extract concepts for this section
    graph_builder = registry.concept_graph_builder  # must be registered
    graph_builder.build_from_document(tree)  # or just extract from section

    # Save updated tree back to artifact store and repository
    registry.repository.save_tree(tree, tree_id.replace("tree_", ""))
    registry.artifact_store.save(tree, key=tree_id)

    return {"section_id": section_id, "concept_count": len(section_node.concepts)}