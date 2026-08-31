# src/store/user_store.py 

import json 
from typing import Dict
from ...core.user.models.user import User

class BaseUserStore:
    def get_user(self, user_id: str) -> User:
        raise NotImplementedError
    
    def save_user_data(self, user: User):
        raise NotImplementedError
    
class UserStore(BaseUserStore):
    def __init__(self):
        self.users = {}
    
    def get_user(self, user_id: str):
        if user_id not in self.users:
            self.users[user_id] = self.get_user_data(user_id)
        return self.users[user_id]

    def get_user_data(self, user_id: str)->User:
        try: 
            with open(f"user_data/{user_id}.json", "r") as f:
                return User.from_dict(json.load(f))
            
        except FileNotFoundError:
            raise ValueError("User data not found")

    def save_user_data(self, user: User):
        with open(f"user_data/{user.user_id}.json", "w") as f:
            json.dump(user.to_dict(), f)
