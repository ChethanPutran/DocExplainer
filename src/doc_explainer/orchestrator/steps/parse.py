import uuid
from doc_explainer.orchestrator import step
from doc_explainer.orchestrator.registry import registry
from doc_explainer.core.document.models.base import DocumentInfo

@step(annotations={"stage": "parse", "retries": 2})
def parse_document_step(file_path: str) -> dict:
    """Parse document and store sections incrementally."""
    # 1. Check if already parsed (via checkpoint)
    doc_id = registry.manager.get_document_id_by_path(file_path)  # you need to implement this
    if doc_id:
        # Already parsed, return existing doc info
        doc = registry.repository.get_document(doc_id)
        return {
            "doc_id": doc.id,
            "title": doc.title,
            "num_sections": len(doc.sections),
            "existing": True
        }

    # 2. Parse the document
    parser = registry.parser_factory.create_parser(file_path)
    if not parser:
        raise ValueError(f"No parser available for {file_path}")

    # Parse metadata first
    doc_info = parser.parse_metadata(file_path)
    doc_id = doc_info.document_id or str(uuid.uuid4())  # generate if missing

    # 3. Iterate sections and store incrementally
    # We'll use the engine's ingest method which already does checkpointing.
    # But we need to ensure the engine uses the checkpoint store.
    engine = registry.engine
    engine.parser = parser  # inject parser if not already set

    # The ingest method processes each section and saves to vector/graph stores.
    # We'll call it and let it handle incremental storage.
    # But ingest expects a file_path and returns doc_id.
    returned_doc_id = engine.ingest(file_path)
    if returned_doc_id != doc_id:
        doc_id = returned_doc_id

    # 4. Also store the full Document structure in the repository for later use.
    # The ingest method may have stored the document; if not, we save it.
    document = parser.parse(file_path)  # This loads the full doc; but we can use it only if needed.
    # To avoid memory, we could skip full parsing and only store sections, but we need the structure.
    # For now, we'll save the full doc (it's okay if we have memory).
    registry.repository.save_document(document, doc_id)

    return {
        "doc_id": doc_id,
        "title": document.title,
        "num_sections": len(document.sections),
        "existing": False
    }