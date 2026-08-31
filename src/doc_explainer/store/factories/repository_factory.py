from typing import Optional, Dict, Any
from ...store.knowledge import ( ConceptRepository,
                                  RelationshipRepository,  
                                  BaseKnowledgeRepository, 
                                  KnowledgeRepository,
                                  InvertedIndexRepository)
from ...store.user import UserRepository
from ...store.document import DocumentRepository


class RepositoryFactory:
    """Factory for creating repository instances"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._instances = {}
    
    def create_concept_repository(self, **kwargs) -> ConceptRepository:
        """Create concept repository"""
        storage_path = kwargs.get('storage_path', 
                                  self.config.get('concept_storage_path', 'data/knowledge/concepts/'))
        
        if 'concept_repo' not in self._instances:
            self._instances['concept_repo'] = ConceptRepository(storage_path=storage_path)
        
        return self._instances['concept_repo']
    
    def create_relationship_repository(self, **kwargs) -> RelationshipRepository:
        """Create relationship repository"""
        storage_path = kwargs.get('storage_path',
                                  self.config.get('relationship_storage_path', 'data/knowledge/relationships/'))
        
        concept_repo = kwargs.get('concept_repo', self.create_concept_repository())
        
        if 'relationship_repo' not in self._instances:
            self._instances['relationship_repo'] = RelationshipRepository(
                storage_path=storage_path,
                concept_repository=concept_repo
            )
        
        return self._instances['relationship_repo']
    
    def create_graph_repository(self, **kwargs) -> BaseKnowledgeRepository:
        """Create graph repository"""
        storage_path = kwargs.get('storage_path',
                                  self.config.get('graph_storage_path', 'data/knowledge/graphs/'))
        
        concept_repo = kwargs.get('concept_repo', self.create_concept_repository())
        relationship_repo = kwargs.get('relationship_repo', self.create_relationship_repository())
        
        if 'graph_repo' not in self._instances:
            self._instances['graph_repo'] = KnowledgeRepository(
                storage_path=storage_path,
                concept_repo=concept_repo,
                relationship_repo=relationship_repo
            )
        
        return self._instances['graph_repo']
    
    def create_index_repository(self, **kwargs) -> InvertedIndexRepository:
        """Create inverted index repository"""
        storage_path = kwargs.get('storage_path',
                                  self.config.get('index_storage_path', 'data/knowledge/index/'))
        
        if 'index_repo' not in self._instances:
            self._instances['index_repo'] = InvertedIndexRepository(storage_path=storage_path)
        
        return self._instances['index_repo']
    
    def create_user_repository(self, **kwargs) -> UserRepository:
        """Create user repository"""
        storage_path = kwargs.get('storage_path',
                                  self.config.get('user_storage_path', 'data/users/'))
        
        if 'user_repo' not in self._instances:
            self._instances['user_repo'] = UserRepository(storage_path=storage_path)
        
        return self._instances['user_repo']
    
    def create_document_repository(self, **kwargs) -> DocumentRepository:
        """Create document repository"""
        storage_path = kwargs.get('storage_path',
                                  self.config.get('document_storage_path', 'data/documents/'))
        
        if 'document_repo' not in self._instances:
            self._instances['document_repo'] = DocumentRepository(storage_path=storage_path)
        
        return self._instances['document_repo']
    
    def create_all_repositories(self) -> Dict[str, Any]:
        """Create all repositories"""
        return {
            'concept_repository': self.create_concept_repository(),
            'relationship_repository': self.create_relationship_repository(),
            'graph_repository': self.create_graph_repository(),
            'index_repository': self.create_index_repository(),
            'user_repository': self.create_user_repository(),
            'document_repository': self.create_document_repository()
        }
    
    def reset(self):
        """Reset factory instances"""
        self._instances = {}