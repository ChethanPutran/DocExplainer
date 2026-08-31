# src/doc_explainer/orchestrator/knowledge_pipeline.py
from .. import pipeline
from ..steps.knowledge_extract import extract_concepts_step
from ..steps.knowledge_build_graph import build_knowledge_graph_step

@pipeline
def knowledge_processing_pipeline(doc_id: str):
    # Step 1: Extract concepts for all sections (we can parallelise per section)
    # For simplicity, we'll just build the graph (which internally extracts)
    graph_result = build_knowledge_graph_step(doc_id)
    return graph_result