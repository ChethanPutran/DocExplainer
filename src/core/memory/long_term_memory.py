from typing import Dict


class LongTermMemory:
    """Stores and retrieves long-term user knowledge"""
    
    def save_concept_memory(self, concept: str, memory_trace: Dict):
        """Save concept to long-term memory"""
        pass
    
    def retrieve_related_concepts(self, concept: str):
        """Retrieve related concepts from memory"""
        pass
    
    def calculate_forgetting_curve(self, concept: str):
        """Calculate forgetting curve for spaced repetition"""
        pass
    def store_question_answer(self, question, explanation):
        """Store the explanation in long-term memory"""
        pass
    def store_summarization(self, selected_text, explanation):
        """Store the explanation in long-term memory"""
        pass
    def retrieve_summarization(self, concept: str):
        """Retrieve summarization related to a concept from long-term memory"""
        pass
    def store_explanation(self, text, explanation):
        """Store the explanation in long-term memory"""
        pass
    def retrieve_explanation(self, concept: str):
        """Retrieve explanation related to a concept from long-term memory"""
        pass
    def update_memory_trace(self, concept: str, feedback: Dict):
        """Update memory trace based on user feedback"""
        pass
    def schedule_review(self, concept: str):
        """Schedule review sessions based on forgetting curve"""
        pass
    def integrate_with_user_model(self, user_model):
        """Integrate long-term memory with user knowledge model"""
        return user_model