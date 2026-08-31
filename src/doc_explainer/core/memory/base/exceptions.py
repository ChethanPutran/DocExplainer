class MemoryError(Exception):
    """Base exception for memory module"""
    pass


class StorageError(MemoryError):
    """Raised when storage operations fail"""
    pass


class RetrievalError(MemoryError):
    """Raised when retrieval operations fail"""
    pass


class SerializationError(MemoryError):
    """Raised when serialization/deserialization fails"""
    pass


class ContextNotFoundError(MemoryError):
    """Raised when context is not found"""
    pass


class ChainError(MemoryError):
    """Raised when chain operations fail"""
    pass