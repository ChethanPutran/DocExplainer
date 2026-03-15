from typing import Dict
from src.core.document.document_manager import DocumentManager
from src.core.explanation_engine.adaptive_explainer import AdaptiveExplainer, Explanation
from src.core.knowlege_modelling.graph import  GraphStateManager
from src.core.memory.memory_manager import MemoryManager
from src.core.memory.session import Context, Context, SessionManager
from src.core.knowlege_modelling.user import UserManager


class DocExplainerPipeline:
    def __init__(self):
        self.doc_manager = DocumentManager()
        self.memory_manager = MemoryManager()
        self.session_manager = SessionManager()
        self.explainer = AdaptiveExplainer()
        self.graph_state_manager = GraphStateManager()
        self.user_manager = UserManager() 
        # self.user = self.long_term_memory.integrate_with_user_model(self.user_model_engine)
        

    # --- Core Pipeline Actions ---

    def summarize(self, doc_id: int, selected_text: str, section_id: int = 0) -> Explanation:
        self._check_document_set(doc_id)
        self.session_manager.handle_interaction("summarize", selected_text)
        context = self._get_context(section_id=section_id)
        response = self.explainer.summarize(text=selected_text, context=context)
        
        # BKT Update: Summaries provide moderate familiarity
        for concept in response.unknown_concepts_explained:
            self.user_model_engine.update_knowledge({"concept": concept, "correct": True})

        self.session_manager.handle_interaction("summarization_response", response.explanation)
        self.memory_manager.handle_event("summarization", {"text": selected_text, "summary": response.explanation})
        return response
    
    def answer_question(self, doc_id: int, question: str, section_id: int = 0) -> Explanation:
        self._check_document_set(doc_id)
        self.session_manager.handle_interaction("answer_question", question)
        context = self._get_context(section_id=section_id)
        
        response = self.explainer.ask(question, context=context)

        # BKT Update: Update knowledge for concepts explained in the answer
        for concept in response.unknown_concepts_explained:
            self.user_model_engine.update_knowledge({"concept": concept, "correct": True})
            
        self._update_user_model({"question": question, "answer": response.explanation})
        self.memory_manager.handle_event("question_answer", {"question": question, "answer": response.explanation})
        self.session_manager.handle_interaction("answer_response", response.explanation)
        return response

    def explain(self, doc_id: int, selected_text: str, section_id: int = 0) -> Explanation:
        print("Explain text : ", selected_text, doc_id)
        self._check_document_set(doc_id)
        self.session_manager.handle_interaction("explain", selected_text)
        context = self._get_context(section_id=section_id)
        
        response = self.explainer.explain(text=selected_text, context=context)

        # BKT Update: Deep explanations are primary learning opportunities
        for concept in response.unknown_concepts_explained:
            self.user_model_engine.update_knowledge({"concept": concept, "correct": True})

        self._update_user_model({"text": selected_text, "explanation": response.explanation})
        self.memory_manager.handle_event("explanation", {"text": selected_text, "explanation": response.explanation})
        self.session_manager.handle_interaction("explain_response", response.explanation)
        return response
    

    def get_section_id_at(self, doc_id: int, page: int, position: int) -> int:
        doc = self.get_document(doc_id)
        if not doc:
            return -1

        for section in doc.sections:
            for para in section.paragraphs:
                # If position is 0 (typical for our PDF emit), match by page only
                if position == 0:
                    if para.page == page:
                        return section.sec_id
                # Otherwise match by both (for DocumentViewer/Text)
                elif para.page == page and (para.start <= position <= para.end):
                    return section.sec_id
        return -1

    def get_document(self, doc_id: int):
        return self.doc_manager.get_document(doc_id)

    def register_document(self, path: str) -> int:
        print("Parsing the documenty...")
        doc_id = self.doc_manager.load_document(
            path)  # Load and register document
        print("Done")
        print("Building documnet tree...")
        document_tree = self.processor.process_document(
            doc_id)  # Get document tree
        print("Done.")
        print("Building documnet chain...")
        self.graph_state_manager.build_chain(
            document_tree)  # Build document chain
        print("Done.")
        return doc_id


    def _check_document_set(self, doc_id: int):
        if not self.doc_manager.has_document(doc_id):
            raise ValueError(
                "No document set in processor. Please provide a document.")

    def _update_user_model(self, user_response: Dict):
        """
        Update the user model based on feedback about a specific concept.
        Args:
            feedback (Dict): Feedback information containing concept_id and understood status.
        """
        self.user.update_knowledge(user_response)

    def _get_context(self, section_id: int) -> Context:
        print(section_id)

        # User knowlege state
        user_knowledge = self.user.get_user_knowledge_state()

        # Session state
        session_context = self.session_manager.get_session_context()

        # Documents state till the time
        document_context = self.graph_state_manager.get_document_context(
            section_id)
        
        # Concept of the doc
        concept_graph = self.graph_state_manager.get_concept_graph_upto(section_id)
        
        context = Context(
            user_knowledge=user_knowledge,
            session_context=session_context,
            document_context=document_context,
            concept_graph=concept_graph
        )
        return context


if __name__ == "__main__":
    pipeline = DocExplainerPipeline()

    DOCUMNET_PATH = "data/report.pdf"
    doc_id = pipeline.register_document(DOCUMNET_PATH)
