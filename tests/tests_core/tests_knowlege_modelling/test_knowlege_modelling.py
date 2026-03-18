from core.knowlege.graph.base import (
    Concept,
    ConceptGraph,
    ConceptNode,
    ConceptNodeRelationship,
    ConceptRelationship,
    GraphDelta,
)
from src.core.knowledge.graph.hierarchy import build_document_hierarchy
from src.core.knowledge.graph.state_manager import GraphStateManager
from src.core.knowledge.graph.updater import GraphUpdater
from core.knowlege.user.knowledge_tracing import BayesianKnowledgeTracer
from src.core.knowledge.knowlege_graph import (
    ConceptBuilder,
    DocumentChain,
    GraphStateManager as CompatGraphStateManager,
    GraphUpdater as CompatGraphUpdater,
    build_document_hierarchy as compat_build_document_hierarchy,
)
from core.knowlege.extraction.extractor import LLMRelationshipExtractor
from core.knowlege.user.model import UserKnowledgeState, UserState


class _ExtractorStub:
    def __init__(self, concepts):
        self._concepts = concepts

    def extract_concepts(self, _texts):
        concepts = [Concept('transformer', aliases=['transformer'], definitions=['A neural network architecture that uses self-attention to process and generate sequential data.']),
                    Concept('code generation', aliases=['code generation'], definitions=[
                            'The automated creation of programming source code by an artificial intelligence model.']),
                    Concept('diffusion', aliases=['diffusion model', 'diffusion'], definitions=[
                            'A class of generative models that create data by iteratively removing noise from a random starting state.']),
                    Concept('sequence', aliases=['sequence'], definitions=[
                        'An ordered list of data points, such as words in a sentence or frames in a video.']),
                    Concept('self - attention mechanism', aliases=['self-attention mechanism', 'self - attention mechanism'], definitions=[
                        'A process that allows a model to weigh the importance of different parts of the input data relative to each other.']),
                    Concept('gpt-5', aliases=['gpt-5'], definitions=[
                        'A specific iteration of the Generative Pre-trained Transformer large language model.']),
                    Concept('gemini-3', aliases=['gemini-3'], definitions=[
                        'A specific multimodal artificial intelligence model developed by Google    .']),
                    Concept('claude-4.5', aliases=['claude-4.5'], definitions=[
                        'A specific large language model developed by Anthropic.']),
                    Concept('video generation', aliases=['video generation'], definitions=[
                        'The process of using AI to create moving visual content from textual or other prompts.']),
                    Concept('random noise', aliases=['random noise'], definitions=['Unstructured data used as the initial input for diffusion-based generative processes.'])]
        return concepts


class _TextModelsStub:
    def __init__(self):
        self._ner = _ExtractorStub(["alpha", "beta", "gamma"])
        self._regex = _ExtractorStub(["alpha"])

    def get_ner_model(self):
        return self._ner

    def get_ner_regex(self):
        return self._regex

    def get_ner_llm(self):
        return None


class _NERLLMStub:
    def __init__(self, llm):
        self.llm = llm

    def extract_concepts(self, _texts):
        return ["alpha", "beta"]


class _TextModelsWithLLMStub:
    def __init__(self):
        self._ner = _ExtractorStub(["alpha", "beta"])
        self._regex = _ExtractorStub(["alpha", "beta"])
        self.llm_client = _LLMStub()
        self._ner_llm = _NERLLMStub(self.llm_client)

    def get_ner_model(self):
        return self._ner

    def get_ner_regex(self):
        return self._regex

    def get_ner_llm(self):
        return self._ner_llm


class _BKTStub:
    def get_user_knowledge_state(self):
        return UserKnowledgeState()


class _LLMStub:
    def generate(self, _prompt):
        return (
            '[{"src":"alpha","tgt":"beta","type":"depends_on",'
            '"why":"alpha is needed before beta"}]'
        )


def test_concept_graph_merges_duplicate_edge_strength():
    graph = ConceptGraph()
    alpha = ConceptNode(Concept("alpha"))
    beta = ConceptNode(Concept("beta"))

    rel_1 = ConceptRelationship(
        alpha.primary_concept, beta.primary_concept, strength=0.4)
    edge_1 = ConceptNodeRelationship(alpha, beta, rel_1)
    graph.add_relationship(alpha, beta, edge_1)

    rel_2 = ConceptRelationship(
        alpha.primary_concept, beta.primary_concept, strength=0.6)
    edge_2 = ConceptNodeRelationship(alpha, beta, rel_2)
    graph.add_relationship(alpha, beta, edge_2)

    stored = graph.graph["alpha"]["beta"]["relationship"]
    assert stored.relationship.strength == 1.0


def test_graph_delta_create_collects_nodes_and_edges():
    graph = ConceptGraph()
    alpha = Concept("alpha")
    beta = Concept("beta")
    rel = ConceptRelationship(alpha, beta, strength=0.7)

    delta = GraphDelta(section_id=1, data=type(
        "Chunk", (), {"text": "alpha beta"})())
    delta.create(graph, [(alpha, [(beta, rel)])])

    assert "alpha" in delta.new_concepts
    assert "beta" in delta.new_concepts
    assert len(delta.new_edges) == 1
    assert delta.new_edges[0].relationship.strength == 0.7


def test_graph_updater_applies_delta_and_adds_subjective_weight():
    graph = ConceptGraph()
    user = UserState()
    user.confidence["alpha"] = 0.6
    user.confidence["beta"] = 0.2
    updater = GraphUpdater(graph, user)

    alpha = Concept("alpha")
    beta = Concept("beta")
    rel = ConceptRelationship(
        alpha, beta, strength=0.5, attributes={"weight": 0.5})
    delta = GraphDelta(section_id=2, data=type(
        "Chunk", (), {"text": "alpha beta"})())
    delta.create(graph, [(alpha, [(beta, rel)])])

    updater.apply_delta(delta)
    edge = graph.graph["alpha"]["beta"]["relationship"]

    assert graph.has_concept("alpha")
    assert graph.has_concept("beta")
    assert edge.relationship.attributes["subjective_weight"] == 0.2


def test_state_manager_build_chain_and_context():
    document_text = "Section A\nAlpha connects beta.\n\nSection B\nBeta enables gamma."
    tree = build_document_hierarchy(document_text)
    manager = GraphStateManager(_TextModelsStub(), _BKTStub())

    manager.build_chain(tree, concepts_per_para=3)
    concept_graph = manager.get_concept_graph_upto(-1)
    context = manager.get_document_context(0)

    assert concept_graph.graph.number_of_nodes() > 0
    assert "text" in context
    assert "Section A" in context["text"]


def test_relationship_extractor_parses_llm_json():
    concepts = [Concept("alpha"), Concept("beta")]
    extractor = LLMRelationshipExtractor(_LLMStub())
    semantic_map = extractor.extract(concepts, "alpha needs beta")

    key = tuple(sorted(["alpha", "beta"]))
    assert key in semantic_map
    assert semantic_map[key]["relation"] == "depends_on"


def test_concept_builder_uses_llm_relationship_extractor():
    builder = ConceptBuilder(_TextModelsWithLLMStub())
    alpha = Concept("alpha")
    beta = Concept("beta")
    text = "Alpha is needed before beta."

    merged = builder._extract_relationships([alpha, beta], text)

    relations_for_alpha = dict((c.name, rel) for c, rel in merged[0][1])
    assert "beta" in relations_for_alpha
    assert relations_for_alpha["beta"].relation == "depends_on"


def test_compatibility_exports_still_available():
    assert CompatGraphStateManager is GraphStateManager
    assert CompatGraphUpdater is GraphUpdater
    assert ConceptBuilder is not None
    assert DocumentChain is not None
    assert compat_build_document_hierarchy is build_document_hierarchy


def test_bayesian_knowledge_tracer_update_knowledge_creates_state():
    tracer = BayesianKnowledgeTracer()
    tracer.update_knowledge({"concept": "graph neural network"})
    state = tracer.get_user_knowledge_state()

    assert len(state.knowledge_states) == 1
    concept = next(iter(state.knowledge_states.keys()))
    assert concept.name == "graph neural network"
