from core.knowlege_modelling.user.knowledge_tracing import BayesianKnowledgeTracer
from src.core.knowlege_modelling.user.base import UserKnowledgeState, User 


class UserManager:
    def __init__(self):
        self.users = {}
        self.bkt = BayesianKnowledgeTracer()
    
    def get_user(self, user_id: str):
        if user_id not in self.users:
            self.users[user_id] = User(user_id)
        return self.users[user_id]