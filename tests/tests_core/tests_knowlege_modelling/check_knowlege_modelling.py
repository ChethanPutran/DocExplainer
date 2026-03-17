import argparse
from pathlib import Path
import sys

# Allow direct execution: `python src/core/knowlege_modelling/check_knowlege_modelling.py ...`
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.document.manager import DocumentManager
from core.document.builder.processor import HierarchicalDocumentProcessor
from src.core.knowlege.graph.hierarchy import build_document_hierarchy
from src.core.knowlege.graph.state_manager import GraphStateManager
from core.knowlege.user.knowledge_tracing import BayesianKnowledgeTracer
from src.models.text import TextModels,NERRegex


class RegexOnlyTextModels:
    """Lightweight text model provider that avoids loading heavy NLP backends."""

    def __init__(self):
        self._regex = NERRegex()

    def get_ner_model(self):
        return self._regex

    def get_ner_regex(self):
        return self._regex

    def get_ner_llm(self):
        return None


def extract_text_and_tree(document_path: Path):
    suffix = document_path.suffix.lower()

    if suffix == ".pdf":
        doc_manager = DocumentManager()
        doc_id = doc_manager.load_document(str(document_path))
        document = doc_manager.get_document(doc_id)
        processor = HierarchicalDocumentProcessor(doc_manager)
        doc_tree = processor.process_document(doc_id)
        text = document.raw_text
        return text, doc_tree

    # Plain text / markdown fallback
    text = document_path.read_text(encoding="utf-8")
    doc_tree = build_document_hierarchy(text)
    return text, doc_tree


def main():
    parser = argparse.ArgumentParser(
        description="Check pipeline for knowledge modelling: document -> tree -> graph -> visualize."
    )
    parser.add_argument("document", help="Path to source document (.pdf/.txt/.md)")
    parser.add_argument(
        "--concepts-per-paragraph",
        type=int,
        default=8,
        help="Maximum concepts to keep per paragraph (default: 8)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip graph visualization (useful for headless environments)",
    )

    args = parser.parse_args()
    doc_path = Path(args.document)

    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    print(f"[1/4] Extracting text from: {doc_path}")
    text, doc_tree = extract_text_and_tree(doc_path)
    print(f"      Text length: {len(text)} characters")

    print("[2/4] Document tree ready")
    print(f"      Sections detected: {len(doc_tree.root.children)}")

    print("[3/4] Building concept graph")
    text_models = RegexOnlyTextModels()
    tracer = BayesianKnowledgeTracer()
    graph_manager = GraphStateManager(text_models, tracer)
    graph_manager.build_chain(doc_tree, concepts_per_para=args.concepts_per_paragraph)
    concept_graph = graph_manager.get_concept_graph_upto(-1)

    print("[4/4] Graph generated")
    print(f"      Nodes: {concept_graph.graph.number_of_nodes()}")
    print(f"      Edges: {concept_graph.graph.number_of_edges()}")

    if args.no_visualize:
        return

    print("Visualizing graph...")
    concept_graph.visualize()


if __name__ == "__main__":
    main()
