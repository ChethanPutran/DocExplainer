from typing import Dict, Optional
from ..base.interfaces import LongTermMemoryInterface
from ..storage.long_term_memory import LongTermMemory
from ..models.memory_trace import ConceptMemoryTrace
import uuid


class MemoryManager:
    """Manages long-term memory operations"""
    
    def __init__(self, memory_storage: Optional[LongTermMemoryInterface] = None):
        self.memory = memory_storage or LongTermMemory()
    
    def handle_event(self, event_type: str, data: Dict) -> bool:
        """Handle an event by storing relevant information in memory"""
        print(f"Handling event: {event_type}")
        
        try:
            if event_type == "summarization":
                return self.memory.store_summarization(
                    data.get("text", ""),
                    data.get("summary", "")
                )
            elif event_type == "question_answer":
                return self.memory.store_question_answer(
                    data.get("question", ""),
                    data.get("answer", "")
                )
            elif event_type == "explanation":
                return self.memory.store_explanation(
                    data.get("text", ""),
                    data.get("explanation", "")
                )
            elif event_type == "concept_encounter":
                return self._handle_concept_encounter(data)
            else:
                print(f"Unknown event type: {event_type}")
                return False
        except Exception as e:
            print(f"Error handling event: {e}")
            return False
    
    def _handle_concept_encounter(self, data: Dict) -> bool:
        """Handle concept encounter event"""
        concept = data.get("concept", "")
        if not concept:
            return False
        
        # Get or create memory trace
        trace = self.memory.get_concept_memory_trace(concept)
        if not trace:
            trace = ConceptMemoryTrace(
                id=str(uuid.uuid4()),
                concept=concept
            )
        
        # Update trace
        trace.add_interaction("encounter", data)
        trace.update_understanding(data.get("understanding", trace.understanding_level))
        
        return self.memory.save_concept_memory_trace(trace)
    
    def store_user_profile(self, user_knowledge_state) -> bool:
        """Store a snapshot of the user knowledge state"""
        return self.memory.store_user_profile(user_knowledge_state)
    
    def retrieve_user_profile(self) -> Optional[Dict]:
        """Retrieve the stored user profile"""
        return self.memory.retrieve_user_profile()
    
    def get_concept_memory(self, concept: str) -> Optional[ConceptMemoryTrace]:
        """Get memory trace for a concept"""
        return self.memory.get_concept_memory_trace(concept)
    
    def retrieve_related_info(self, query: str) -> Dict:
        """Retrieve all related information for a query"""
        return {
            "explanation": self.memory.retrieve_explanation(query),
            "summarization": self.memory.retrieve_summarization(query),
            "qa": self.memory.retrieve_question_answer(query),
            "concept": self.memory.retrieve_concept_memory(query)
        }
    
    def calculate_forgetting_curve(self, concept: str) -> Dict:
        """Calculate forgetting curve for a concept"""
        return self.memory.calculate_forgetting_curve(concept)
    
    def schedule_review(self, concept: str) -> Dict:
        """Schedule review for a concept"""
        return self.memory.schedule_review(concept)
    
    def update_memory_trace(self, concept: str, feedback: Dict) -> bool:
        """Update memory trace with feedback"""
        return self.memory.update_memory_trace(concept, feedback)
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return self.memory.get_statistics()