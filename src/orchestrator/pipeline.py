from typing import Dict
from src.core.document.document import Document
from src.orchestrator.document_manager import DocumentManager
from src.core.document.document_processing import HierarchicalDocumentProcessor as DocumentProcessor
from src.models.llm_wrapper import LLMWrapper
from src.core.explanation_engine.adaptive_explainer import AdaptiveExplainer, Explanation, ExplanationMetadata
from src.core.knowlege_modelling.user_model import BayesianKnowledgeTracer as UserModel
from src.core.knowlege_modelling.knowledge_tracing import ConceptGraphBuilder 


class DocExplainerPipeline:
    def __init__(self):
        self.doc_manager = DocumentManager()
        self.processor = DocumentProcessor()
        self.concept_graph_builder = ConceptGraphBuilder()
        self.user = UserModel()
        self.llm = LLMWrapper()
        self.explainer = AdaptiveExplainer(self.llm)

    def register_document(self, path: str) -> str:
        return self.doc_manager.load_document(path)

    def summarize(self, doc_id: str, selected_text: str):
        if self.processor.is_document_set():
            raise ValueError("No document set in processor. Please provide a document.")
        doc = self.doc_manager.get_document(doc_id)
        # response = self.explainer.summarize(doc_id=doc_id, text=selected_text)

        return Explanation("This is summarization",{},{},[],[],[],ExplanationMetadata(1,True,True))

    def answer_question(self, doc_id: str, selected_text: str):
        if self.processor.is_document_set():
            raise ValueError("No document set in processor. Please provide a document.")
        doc = self.doc_manager.get_document(doc_id)

        self.on_user_feedback({
            "doc_id": doc_id,
            "selected_text": selected_text
        })
        # response = self.explainer.answer_question(doc_id=doc_id, text=selected_text)
        return Explanation("Hello World",{},{},[],[],[],ExplanationMetadata(1,True,True))

    def explain(self, doc_id: str, selected_text: str) -> Explanation:
        """
        Generate an explanation for the given text within the context of the document.
        If a document is provided, it sets it in the processor; otherwise, it uses the
        already set document.
        Args:
            text (str): The text segment to explain.
            doc (fitz.Document | None): Optional document to set in the processor.
        Returns:
            Explanation: The generated explanation object.
        Raises:
            ValueError: If no document is set in the processor and no document is provided.
        """
        if self.processor.is_document_set():
            raise ValueError("No document set in processor. Please provide a document.")

        doc = self.doc_manager.get_document(doc_id)

        return self._explain_with_context(doc, selected_text)

    def _explain_with_context(self, doc: Document, text: str) -> Explanation:
        """
        Generate an explanation for the given text within the context of the document.
        """
        self.processor.set_document(doc)
        concepts = self.processor.process_document(text)
        concept_graph = self.concept_graph_builder.extract_concepts_from_document(concepts)
        
        explanation = self.explainer.explain(
            text=text,
            context={},
            concept_graph=concept_graph,
            user_state=self.user.get_user_knowledge_state()
        )
        return explanation
    
    def on_user_feedback(self, user_response:Dict):
        """
        Update the user model based on feedback about a specific concept.
        Args:
            feedback (Dict): Feedback information containing concept_id and understood status.
        """
        self.user.update_knowledge(user_response)
