from typing import Dict

from core.memory.long_term_memory import LongTermMemory

class MemoryManager:
    """Base class for memory management. Defines the interface for storing and retrieving user knowledge."""
    def __init__(self) -> None:
        self.memory = LongTermMemory()
    def handle_event(self, event_type: str, data: Dict):
        """Handles an event by storing relevant information in memory."""
        # This is where you would implement logic to store the event data in a structured way.
        # For example, you could have different storage strategies based on event type.
        print(f"Handling event: {event_type} with data: {data}")
        if event_type == "summarization":
            self.memory.store_summarization(data["text"], data["summary"])
        elif event_type == "question_answer":
            self.memory.store_question_answer(data["question"], data["answer"])
        elif event_type == "explanation":
            self.memory.store_explanation(data["text"], data["explanation"])
        else:
            print(f"Unknown event type: {event_type}. Data not stored.")

    def store_user_profile(self, user_knowledge_state):
        """Stores a snapshot of the BKT model in the memory storage."""
        raise NotImplementedError("Must implement store_user_profile method")

    def retrieve_user_profile(self):
        """Retrieves the stored user profile from memory."""
        raise NotImplementedError("Must implement retrieve_user_profile method")
    