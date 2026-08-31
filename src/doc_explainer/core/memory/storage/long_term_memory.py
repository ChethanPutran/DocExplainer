import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from doc_explainer.core.memory.models.context import SessionContext
from .base import MemoryStorage
from ..base.interfaces import LongTermMemoryInterface
from ..base.exceptions import StorageError, RetrievalError
from ..models.memory_trace import ConceptMemoryTrace
from .serializers import MemorySerializer


class LongTermMemory(MemoryStorage, LongTermMemoryInterface):
    """Stores and retrieves long-term user knowledge with JSON persistence"""
    
    def __init__(self, file_path: str = "data/memory/user_memory.json"):
        super().__init__()
        self.file_path = file_path
        self._ensure_directory()
        self._storage = self._load_from_disk()
    
    def _ensure_directory(self):
        """Ensure the directory for the file exists"""
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def _load_from_disk(self) -> Dict:
        """Load memory from JSON file"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {self.file_path}, starting with fresh memory.")
            except IOError as e:
                print(f"IOError reading {self.file_path}: {e}")
        
        # Default structure
        return {
            "user_profiles": {},
            "concepts": {},
            "interactions": {
                "summaries": [],
                "qa": [],
                "explanations": []
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
    
    def _save_to_disk(self) -> bool:
        """Write memory to JSON file"""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self._storage, f, indent=4, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"Failed to save memory: {e}")
            return False
    
    # User Profile Methods
    
    def store_user_profile(self, user_knowledge_state) -> bool:
        """Store a serialized snapshot of the user knowledge state"""
        try:
            if hasattr(user_knowledge_state, 'to_dict'):
                profile_data = user_knowledge_state.to_dict()
            else:
                profile_data = user_knowledge_state
            
            self._storage["user_profiles"]["latest"] = {
                "data": profile_data,
                "timestamp": datetime.now().isoformat()
            }
            return self._save_to_disk()
        except Exception as e:
            raise StorageError(f"Failed to store user profile: {e}")
    
    def retrieve_user_profile(self) -> Optional[Dict]:
        """Retrieve the stored user profile"""
        return self._storage.get("user_profiles", {}).get("latest")
    
    # Concept Memory Methods
    
    def store_concept_memory(self, concept: str, memory_trace: Dict) -> bool:
        """Store concept memory trace"""
        try:
            if concept not in self._storage["concepts"]:
                self._storage["concepts"][concept] = []
            
            trace_with_timestamp = {
                **memory_trace,
                "timestamp": datetime.now().isoformat()
            }
            self._storage["concepts"][concept].append(trace_with_timestamp)
            return self._save_to_disk()
        except Exception as e:
            raise StorageError(f"Failed to store concept memory: {e}")

    def store_session_context(self, session_context: SessionContext) -> bool:
        """Store session context"""
        try:
            self._storage["session_context"] = {
                **session_context.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            return self._save_to_disk()
        except Exception as e:
            raise StorageError(f"Failed to store session context: {e}")
        
    def retrieve_concept_memory(self, concept: str) -> Optional[Dict]:
        """Retrieve concept memory trace"""
        traces = self._storage["concepts"].get(concept, [])
        return traces[-1] if traces else None
    
    def retrieve_all_concept_memories(self, concept: str) -> List[Dict]:
        """Retrieve all memory traces for a concept"""
        return self._storage["concepts"].get(concept, [])
    
    def save_concept_memory_trace(self, trace: ConceptMemoryTrace) -> bool:
        """Save a ConceptMemoryTrace object"""
        return self.store_concept_memory(trace.concept, trace.to_dict())

    def update_concept_memory_trace(self, concept: str, trace: ConceptMemoryTrace) -> bool:
        """Update a ConceptMemoryTrace object"""
        try:
            traces = self._storage["concepts"].get(concept, [])
            if traces:
                traces[-1] = trace.to_dict()
                return self._save_to_disk()
            return False
        except Exception as e:
            raise StorageError(f"Failed to update concept memory trace: {e}")

    def delete_concept_memory_trace(self, concept: str) -> bool:
        """Delete all memory traces for a concept"""
        try:
            if concept in self._storage["concepts"]:
                del self._storage["concepts"][concept]
                return self._save_to_disk()
            return False
        except Exception as e:
            raise StorageError(f"Failed to delete concept memory trace: {e}")
    
    def get_concept_memory_trace(self, concept: str) -> Optional[ConceptMemoryTrace]:
        """Get ConceptMemoryTrace object"""
        trace_data = self.retrieve_concept_memory(concept)
        if trace_data:
            return ConceptMemoryTrace.from_dict(trace_data)
        return None
    
    # Interaction Methods
    
    def store_interaction(self, interaction_type: str, data: Dict) -> bool:
        """Store a user interaction"""
        try:
            if interaction_type in self._storage["interactions"]:
                interaction_with_timestamp = {
                    **data,
                    "timestamp": datetime.now().isoformat()
                }
                self._storage["interactions"][interaction_type].append(interaction_with_timestamp)
                return self._save_to_disk()
            return False
        except Exception as e:
            raise StorageError(f"Failed to store interaction: {e}")
    
    def store_question_answer(self, question: str, answer: str) -> bool:
        """Store a question-answer pair"""
        return self.store_interaction("qa", {
            "question": question,
            "answer": answer
        })
    
    def store_summarization(self, text: str, summary: str) -> bool:
        """Store a summarization"""
        return self.store_interaction("summaries", {
            "text": text,
            "summary": summary
        })
    
    def store_explanation(self, text: str, explanation: str) -> bool:
        """Store an explanation"""
        return self.store_interaction("explanations", {
            "text": text,
            "explanation": explanation
        })
    
    # Retrieval Methods
    
    def retrieve_related_concepts(self, concept: str) -> Optional[Dict]:
        """Retrieve concepts related to the given concept"""
        return self.retrieve_concept_memory(concept)
    
    def retrieve_summarization(self, text: str) -> str | None:
        """Retrieve a summarization related to the query"""
        for item in reversed(self._storage["interactions"]["summaries"]):
            if text.lower() in item.get("text", "").lower():
                return item
        return None
    
    def retrieve_explanation(self, text: str) -> str | None:
        """Retrieve an explanation related to the query"""
        for item in reversed(self._storage["interactions"]["explanations"]):
            if text.lower() in item.get("text", "").lower():
                return item
        return None
    
    def retrieve_question_answer(self, question: str) -> str | None:
        """Retrieve a question-answer pair related to the query"""
        for item in reversed(self._storage["interactions"]["qa"]):
            if question.lower() in item.get("question", "").lower():
                return item
        return None
    
    # Memory Trace Management
    
    def update_memory_trace(self, concept: str, feedback: Dict) -> bool:
        """Update memory trace based on user feedback"""
        try:
            traces = self._storage["concepts"].get(concept, [])
            if traces:
                latest = traces[-1]
                latest.update(feedback)
                latest["updated_at"] = datetime.now().isoformat()
                return self._save_to_disk()
            return False
        except Exception as e:
            raise StorageError(f"Failed to update memory trace: {e}")
    
    def calculate_forgetting_curve(self, concept: str) -> Dict:
        """Calculate forgetting curve for spaced repetition"""
        trace = self.retrieve_concept_memory(concept)
        if not trace:
            return {"concept": concept, "retention": 1.0}
        
        # Simple forgetting curve calculation
        from datetime import datetime
        last_access = datetime.fromisoformat(trace.get("timestamp", datetime.now().isoformat()))
        hours_since = (datetime.now() - last_access).total_seconds() / 3600
        
        # Ebbinghaus forgetting curve approximation
        retention = 1.0 / (1.0 + 0.1 * hours_since)
        
        return {
            "concept": concept,
            "retention": retention,
            "hours_since": hours_since,
            "last_access": last_access.isoformat()
        }
    
    def schedule_review(self, concept: str) -> Dict:
        """Schedule review sessions based on forgetting curve"""
        forgetting = self.calculate_forgetting_curve(concept)
        retention = forgetting["retention"]
        
        # Schedule next review based on retention
        if retention > 0.8:
            next_review = "in 7 days"
        elif retention > 0.6:
            next_review = "in 3 days"
        elif retention > 0.4:
            next_review = "tomorrow"
        else:
            next_review = "today"
        
        return {
            "concept": concept,
            "next_review": next_review,
            "retention": retention
        }
    
    # Integration Methods
    
    def integrate_with_user_model(self, user_model):
        """Integrate long-term memory with user knowledge model"""
        # This would update the user model with long-term memory data
        profile = self.retrieve_user_profile()
        if profile and hasattr(user_model, 'load_from_dict'):
            user_model.load_from_dict(profile.get("data", {}))
        return user_model
    
    # Utility Methods
    
    def clear_all(self):
        """Clear all memory"""
        self._storage = {
            "user_profiles": {},
            "concepts": {},
            "interactions": {
                "summaries": [],
                "qa": [],
                "explanations": []
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "cleared_at": datetime.now().isoformat()
            }
        }
        self._save_to_disk()
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            "total_concepts": len(self._storage["concepts"]),
            "total_summaries": len(self._storage["interactions"]["summaries"]),
            "total_qa": len(self._storage["interactions"]["qa"]),
            "total_explanations": len(self._storage["interactions"]["explanations"]),
            "has_user_profile": "latest" in self._storage["user_profiles"],
            "file_path": self.file_path
        }