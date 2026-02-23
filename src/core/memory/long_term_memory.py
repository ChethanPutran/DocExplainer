from typing import Dict
import json
import os


class LongTermMemory:
    """Stores and retrieves long-term user knowledge with JSON persistence"""

    def __init__(self, file_path="user_memory.json"):
        self.file_path = file_path
        self.storage = self._load_from_disk()

    def store_user_profile(self, user_knowledge_state):
        """Stores a serialized snapshot of the BKT model in the memory storage."""
        # Use your existing to_dict() method for JSON persistence
        self.storage["user_profile_snapshot"] = user_knowledge_state.to_dict()
        self._save_to_disk()

    def _load_from_disk(self) -> Dict:
        """Loads memory from a JSON file if it exists, otherwise returns empty structure."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(
                    f"Error reading {self.file_path}, starting with fresh memory.")

        # Default Structure
        return {
            "concepts": {},
            "summaries": [],
            "qa": [],
            "explanations": [],
        }

    def _save_to_disk(self):
        """Writes the current state of memory to the JSON file."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.storage, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Failed to save memory: {e}")

    # --- Persisted Storage Methods ---

    def store_question_answer(self, question, explanation):
        self.storage["qa"].append({
            "question": question,
            "answer": explanation.explanation
        })
        self._save_to_disk()

    def store_summarization(self, selected_text, explanation):
        self.storage["summaries"].append({
            "text": selected_text,
            "summary": explanation.explanation
        })
        self._save_to_disk()

    def store_explanation(self, text, explanation):
        self.storage["explanations"].append({
            "text": text,
            "explanation": explanation.explanation
        })
        self._save_to_disk()

    def save_concept_memory(self, concept: str, memory_trace: Dict):
        self.storage["concepts"][concept] = memory_trace
        self._save_to_disk()

    # --- Retrieval Methods (No changes needed) ---
    def retrieve_related_concepts(self, concept: str):
        return self.storage["concepts"].get(concept)

    def retrieve_summarization(self, concept: str):
        for item in reversed(self.storage["summaries"]):
            if concept.lower() in item["text"].lower():
                return item
        return None

    def calculate_forgetting_curve(self, concept: str):
        """Calculate forgetting curve for spaced repetition"""
        return {"concept": concept, "retention": 1.0}

    def retrieve_explanation(self, concept: str):
        """Retrieve explanation related to a concept from long-term memory"""
        for item in reversed(self.storage["explanations"]):
            if concept.lower() in item["text"].lower():
                return item
        return None

    def update_memory_trace(self, concept: str, feedback: Dict):
        """Update memory trace based on user feedback"""
        existing = self.storage["concepts"].get(concept, {})
        existing.update(feedback)
        self.storage["concepts"][concept] = existing

    def schedule_review(self, concept: str):
        """Schedule review sessions based on forgetting curve"""
        return {"concept": concept, "next_review": "soon"}

    def integrate_with_user_model(self, user_model):
        """Integrate long-term memory with user knowledge model"""
        return user_model
