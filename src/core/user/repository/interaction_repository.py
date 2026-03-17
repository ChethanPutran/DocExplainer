import json
from typing import List, Optional
from datetime import datetime
from src.core.user.models.interaction import UserInteraction

class InteractionRepository:
    """Repository for user interaction data"""
    
    def __init__(self, storage_path: str = "data/interactions/"):
        self.storage_path = storage_path
    
    def save_interaction(self, user_id: str, interaction: UserInteraction) -> bool:
        """Save a user interaction"""
        import os
        try:
            os.makedirs(f"{self.storage_path}{user_id}/", exist_ok=True)
            
            filename = f"{interaction.last_seen.strftime('%Y%m%d_%H%M%S')}.json"
            with open(f"{self.storage_path}{user_id}/{filename}", "w") as f:
                json.dump(interaction.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving interaction: {e}")
            return False
    
    def get_interactions(self, user_id: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[UserInteraction]:
        """Get interactions for a user within date range"""
        import os
        import glob
        
        interactions = []
        try:
            files = glob.glob(f"{self.storage_path}{user_id}/*.json")
            
            for filepath in files:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    interaction = UserInteraction.from_dict(data)
                    
                    # Filter by date range if provided
                    if start_date and interaction.last_seen < start_date:
                        continue
                    if end_date and interaction.last_seen > end_date:
                        continue
                    
                    interactions.append(interaction)
            
            # Sort by date
            interactions.sort(key=lambda x: x.last_seen, reverse=True)
            
        except Exception as e:
            print(f"Error loading interactions: {e}")
        
        return interactions
    
    def get_recent_interactions(self, user_id: str, limit: int = 10) -> List[UserInteraction]:
        """Get most recent interactions"""
        interactions = self.get_interactions(user_id)
        return interactions[:limit]
    
    def delete_interactions(self, user_id: str, before_date: datetime) -> bool:
        """Delete interactions older than date"""
        import os
        import glob
        
        try:
            files = glob.glob(f"{self.storage_path}{user_id}/*.json")
            
            for filepath in files:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    last_seen = datetime.fromisoformat(data.get('last_seen'))
                    
                    if last_seen < before_date:
                        os.remove(filepath)
            
            return True
        except Exception as e:
            print(f"Error deleting interactions: {e}")
            return False