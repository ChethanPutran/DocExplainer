from typing import Dict
from core.document.document_manager import DocumentManager
from src.core.document.document_processing import HierarchicalDocumentProcessor as DocumentProcessor
from src.models.llm_wrapper import LLMWrapper
from src.models.text import TextModels
from src.core.explanation_engine.adaptive_explainer import AdaptiveExplainer, Explanation, ExplanationMetadata
from src.core.knowlege_modelling.user_model import BayesianKnowledgeTracer as UserModel
from src.core.knowlege_modelling.knowledge_tracing import  GraphStateManager
from src.core.memory.long_term_memory import LongTermMemory
from src.core.memory.session import Context, Context, SessionChain

class DocExplainerPipeline:
    def __init__(self):
        self.doc_manager = DocumentManager()
        self.processor = DocumentProcessor(self.doc_manager)
        self.llm = LLMWrapper()
        self.text_models = TextModels(self.llm)
        self.long_term_memory = LongTermMemory()
        self.session_chain = SessionChain()
        self.explainer = AdaptiveExplainer(self.llm)
        self.user = self.long_term_memory.integrate_with_user_model(UserModel())
        self.graph_state_manager = GraphStateManager(self.text_models)

    def register_document(self, path: str) -> int:
        doc_id = self.doc_manager.load_document(path) # Load and register document
        self.processor.set_document(doc_id) # Process document
        document_tree = self.processor.get_document_tree(doc_id) # Get document tree
        self.graph_state_manager.build_chain(document_tree) # Build document chain
        return doc_id

    def summarize(self, doc_id: int, selected_text: str, section_id: int = 0) -> Explanation:
        self._check_document_set(doc_id)
        self.session_chain.add_interaction("summarize",selected_text)
        context = self._get_context(section_id=section_id)
        # response = self.explainer.summarize(text=selected_text, context=context)
        response = Explanation("This is summarization",{},{},[],[],[],ExplanationMetadata(1,True,True))
        self.session_chain.add_interaction("summarization_response",response.explanation)
        self.long_term_memory.store_summarization(selected_text, response)
        return response
    
    def answer_question(self, doc_id: int, question: str,section_id: int = 0) -> Explanation:
        self._check_document_set(doc_id)
        self.session_chain.add_interaction("answer_question",question)
        context = self._get_context(section_id=section_id)
        response = self.explainer.ask(text=question, context=context)
        response = Explanation("Hello World",{},{},[],[],[],ExplanationMetadata(1,True,True)) # Generate answer
        self._update_user_model(
            {
                "question": question,
                "answer": response.explanation
            }
        )  # Update user model based on the question
        self.long_term_memory.store_question_answer(question,response) # Store Q&A in long-term memory
        self.session_chain.add_interaction("answer_response",response.explanation)
        return response

    def explain(self, doc_id: int, selected_text: str, section_id: int = 0) -> Explanation:
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
        self._check_document_set(doc_id)
        self.session_chain.add_interaction("explain", selected_text)
        context = self._get_context(section_id=section_id)
        response = self.explainer.explain(text=selected_text, context=context)
        response = Explanation("Hello World",{},{},[],[],[],ExplanationMetadata(1,True,True)) # Generate answer
        self._update_user_model({
            "text": selected_text,
            "explanation": response.explanation
        })  # Update user model based on the question
        self.long_term_memory.store_question_answer(selected_text,response) # Store Q&A in long-term memory
        self.session_chain.add_interaction("explain_response",response.explanation)
        return response

    def _check_document_set(self, doc_id: int):
        if not self.processor.is_document_set(doc_id):
            raise ValueError("No document set in processor. Please provide a document.")
    
    def _update_user_model(self, user_response:Dict):
        """
        Update the user model based on feedback about a specific concept.
        Args:
            feedback (Dict): Feedback information containing concept_id and understood status.
        """
        self.user.update_knowledge(user_response)

    def _get_context(self, section_id: int) -> Context:
        
        user_knowledge = self.user.get_user_knowledge_state()
        session_context = self.session_chain.get_session_context()
        document_context = self.graph_state_manager.get_document_context(section_id)
        concept_graph = self.graph_state_manager.get_concept_graph_upto(section_id)
        context = Context(
            user_knowledge=user_knowledge,
            session_context=session_context,
            document_context=document_context,
            concept_graph=concept_graph
        )
        return context
    