from typing import Optional, Dict, Any
from ..storage.long_term_memory import LongTermMemory
from ..storage.session_memory import SessionMemory
from ..managers.memory_manager import MemoryManager
from ..managers.session_manager import SessionManager
from ..managers.context_manager import ContextManager
from ..chains.session_chain import SessionChain
from ..strategies.forgetting_curve import EbbinghausForgettingCurve, PowerLawForgettingCurve
from ..strategies.review_scheduler import SpacedRepetitionScheduler, LeitnerSystemScheduler


class MemoryFactory:
    """Factory for creating memory-related objects"""
    
    @staticmethod
    def create_long_term_memory(file_path: str = "data/memory/user_memory.json") -> LongTermMemory:
        """Create long-term memory storage"""
        return LongTermMemory(file_path=file_path)
    
    @staticmethod
    def create_session_memory() -> SessionMemory:
        """Create session memory"""
        return SessionMemory()
    
    @staticmethod
    def create_memory_manager(memory_storage: Optional[LongTermMemory] = None) -> MemoryManager:
        """Create memory manager"""
        if memory_storage is None:
            memory_storage = MemoryFactory.create_long_term_memory()
        return MemoryManager(memory_storage)
    
    @staticmethod
    def create_session_manager() -> SessionManager:
        """Create session manager"""
        return SessionManager()
    
    @staticmethod
    def create_session_chain() -> SessionChain:
        """Create session chain"""
        return SessionChain()
    
    @staticmethod
    def create_context_manager(user_knowledge=None, session_context=None,
                              document_context=None, concept_graph=None) -> ContextManager:
        """Create context manager"""
        return ContextManager(
            user_knowledge=user_knowledge,
            session_context=session_context,
            document_context=document_context,
            concept_graph=concept_graph
        )
    
    @staticmethod
    def create_forgetting_curve(strategy: str = "ebbinghaus", **kwargs) -> Any:
        """Create forgetting curve strategy"""
        if strategy == "ebbinghaus":
            return EbbinghausForgettingCurve(**kwargs)
        elif strategy == "power_law":
            return PowerLawForgettingCurve(**kwargs)
        else:
            raise ValueError(f"Unknown forgetting curve strategy: {strategy}")
    
    @staticmethod
    def create_review_scheduler(strategy: str = "spaced_repetition", **kwargs) -> Any:
        """Create review scheduler strategy"""
        if strategy == "spaced_repetition":
            return SpacedRepetitionScheduler(**kwargs)
        elif strategy == "leitner":
            return LeitnerSystemScheduler(**kwargs)
        else:
            raise ValueError(f"Unknown review scheduler strategy: {strategy}")
    
    @staticmethod
    def create_default_services() -> Dict[str, Any]:
        """Create default memory services"""
        long_term_memory = MemoryFactory.create_long_term_memory()
        
        return {
            'long_term_memory': long_term_memory,
            'session_memory': MemoryFactory.create_session_memory(),
            'memory_manager': MemoryFactory.create_memory_manager(long_term_memory),
            'session_manager': MemoryFactory.create_session_manager(),
            'session_chain': MemoryFactory.create_session_chain(),
            'context_manager': MemoryFactory.create_context_manager(),
            'forgetting_curve': MemoryFactory.create_forgetting_curve(),
            'review_scheduler': MemoryFactory.create_review_scheduler()
        }