"""
User Profile Repository - Handles persistence of user profiles to disk.

Provides CRUD operations and export/import functionality for user profiles.
"""

import json
import os
from typing import Optional, Dict, List
from datetime import datetime
from abc import ABC, abstractmethod

from ..models.user_profile import UserProfile


class BaseUserProfileRepository(ABC):
    """Abstract base class for user profile persistence."""
    
    @abstractmethod
    def save_profile(self, profile: UserProfile) -> UserProfile:
        """Save a user profile."""
        pass
    
    @abstractmethod
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID."""
        pass
    
    @abstractmethod
    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        pass
    
    @abstractmethod
    def profile_exists(self, user_id: str) -> bool:
        """Check if a user profile exists."""
        pass
    
    @abstractmethod
    def list_user_ids(self) -> List[str]:
        """List all user IDs."""
        pass
    
    @abstractmethod
    def export_profile(self, user_id: str, export_path: str) -> bool:
        """Export a profile to a file."""
        pass
    
    @abstractmethod
    def import_profile(self, import_path: str) -> Optional[UserProfile]:
        """Import a profile from a file."""
        pass
    
    @abstractmethod
    def clear_cache(self):
        """Clear in-memory cache if applicable."""
        pass


class UserProfileRepository(BaseUserProfileRepository):
    """File-based repository for user profile persistence."""
    
    def __init__(self, storage_path: str = "data/user_profiles/"):
        """
        Initialize the repository.
        
        Args:
            storage_path: Base path for storing profile files
        """
        self.storage_path = storage_path
        self.cache: Dict[str, UserProfile] = {}
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists."""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "backups"), exist_ok=True)
    
    def _get_profile_path(self, user_id: str) -> str:
        """Get file path for a profile."""
        return os.path.join(self.storage_path, f"{user_id}_profile.json")
    
    def _get_backup_path(self, user_id: str, timestamp: datetime) -> str:
        """Get backup file path for a profile."""
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.storage_path, "backups", f"{user_id}_profile_{ts_str}.json")
    
    def save_profile(self, profile: UserProfile) -> UserProfile:
        """
        Save a user profile to disk.
        
        Args:
            profile: UserProfile instance to save
        
        Returns:
            UserProfile: The saved profile
        """
        profile.updated_at = datetime.now()
        
        filepath = self._get_profile_path(profile.user_id)
        
        # Create backup if file exists
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    backup_data = json.load(f)
                backup_path = self._get_backup_path(
                    profile.user_id,
                    datetime.fromisoformat(backup_data.get('updated_at', datetime.now().isoformat()))
                )
                with open(backup_path, 'w') as f:
                    json.dump(backup_data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not create backup for {profile.user_id}: {e}")
        
        # Save profile
        try:
            with open(filepath, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            # Update cache
            self.cache[profile.user_id] = profile
            return profile
        except Exception as e:
            raise IOError(f"Failed to save profile {profile.user_id}: {e}")
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile from disk or cache.
        
        Args:
            user_id: User ID to retrieve
        
        Returns:
            UserProfile or None if not found
        """
        try:
            # Check cache first
            if user_id in self.cache:
                return self.cache[user_id]
            
            # Load from file
            filepath = self._get_profile_path(user_id)
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                profile = UserProfile.from_dict(data)
                self.cache[user_id] = profile
                return profile
        except Exception as e:
            print(f"Error loading profile {user_id}: {e}")
            return None
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Delete a user profile.
        
        Args:
            user_id: User ID to delete
        
        Returns:
            bool: True if deleted, False if not found
        """
        filepath = self._get_profile_path(user_id)
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                # Remove from cache
                if user_id in self.cache:
                    del self.cache[user_id]
                return True
            except Exception as e:
                print(f"Error deleting profile {user_id}: {e}")
                return False
        
        return False
    
    def profile_exists(self, user_id: str) -> bool:
        """
        Check if a user profile exists.
        
        Args:
            user_id: User ID to check
        
        Returns:
            bool: True if profile exists
        """
        return os.path.exists(self._get_profile_path(user_id))
    
    def list_user_ids(self) -> List[str]:
        """
        List all user IDs with profiles.
        
        Returns:
            List of user IDs
        """
        user_ids = []
        
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('_profile.json'):
                    user_id = filename.replace('_profile.json', '')
                    user_ids.append(user_id)
        except Exception as e:
            print(f"Error listing user profiles: {e}")
        
        return user_ids
    
    def export_profile(self, user_id: str, export_path: str) -> bool:
        """
        Export a user profile to a file.
        
        Args:
            user_id: User ID to export
            export_path: Path to export to
        
        Returns:
            bool: True if successful
        """
        profile = self.get_profile(user_id)
        if not profile:
            return False
        
        try:
            # Create export directory if needed
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            export_data = {
                'profile': profile.to_dict(),
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error exporting profile {user_id}: {e}")
            return False
    
    def import_profile(self, import_path: str) -> Optional[UserProfile]:
        """
        Import a user profile from a file.
        
        Args:
            import_path: Path to import from
        
        Returns:
            UserProfile or None if import failed
        """
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            # Handle both direct profile data and wrapped export format
            if 'profile' in data:
                profile_data = data['profile']
            else:
                profile_data = data
            
            profile = UserProfile.from_dict(profile_data)
            
            # Save to repository
            self.save_profile(profile)
            
            return profile
        except Exception as e:
            print(f"Error importing profile: {e}")
            return None
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()
    
    def reload_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Force reload a profile from disk, bypassing cache.
        
        Args:
            user_id: User ID to reload
        
        Returns:
            UserProfile or None if not found
        """
        if user_id in self.cache:
            del self.cache[user_id]
        return self.get_profile(user_id)
    
    def get_profile_statistics(self, user_id: str) -> Dict:
        """
        Get statistics for a specific user profile.
        
        Args:
            user_id: User ID
        
        Returns:
            Dict with profile statistics
        """
        profile = self.get_profile(user_id)
        if not profile:
            return {}
        
        return profile.get_profile_summary()
    
    def get_all_statistics(self) -> Dict:
        """
        Get statistics for all user profiles.
        
        Returns:
            Dict with statistics for all profiles
        """
        user_ids = self.list_user_ids()
        profiles = []
        
        for user_id in user_ids:
            stats = self.get_profile_statistics(user_id)
            if stats:
                profiles.append(stats)
        
        return {
            'total_users': len(user_ids),
            'profiles': profiles
        }
    
    def find_profiles_by_mastery_range(self, 
                                      min_mastery: float = 0.0, 
                                      max_mastery: float = 1.0) -> List[str]:
        """
        Find user IDs with average mastery in the specified range.
        
        Args:
            min_mastery: Minimum average mastery
            max_mastery: Maximum average mastery
        
        Returns:
            List of user IDs matching the criteria
        """
        user_ids = self.list_user_ids()
        matching_users = []
        
        for user_id in user_ids:
            profile = self.get_profile(user_id)
            if profile:
                avg_mastery = sum(m.p_knowledge for m in profile.known_concepts.values()) / max(1, len(profile.known_concepts))
                if min_mastery <= avg_mastery <= max_mastery:
                    matching_users.append(user_id)
        
        return matching_users
    
    def get_profiles_by_preference(self, 
                                   preference_key: str, 
                                   preference_value: str) -> List[str]:
        """
        Find user IDs with a specific preference value.
        
        Args:
            preference_key: Preference attribute name (e.g., 'learning_pace')
            preference_value: Value to match
        
        Returns:
            List of user IDs matching the criteria
        """
        user_ids = self.list_user_ids()
        matching_users = []
        
        for user_id in user_ids:
            profile = self.get_profile(user_id)
            if profile:
                pref_value = getattr(profile.preferences, preference_key, None)
                if pref_value:
                    # Handle enum values
                    pref_value_str = pref_value.value if hasattr(pref_value, 'value') else str(pref_value)
                    if pref_value_str == preference_value:
                        matching_users.append(user_id)
        
        return matching_users
    
    def cleanup_old_backups(self, max_backups_per_user: int = 5):
        """
        Clean up old backup files, keeping only the most recent N backups per user.
        
        Args:
            max_backups_per_user: Maximum backups to keep per user
        """
        backup_dir = os.path.join(self.storage_path, "backups")
        if not os.path.exists(backup_dir):
            return
        
        # Group backups by user_id
        backups_by_user: Dict[str, List[str]] = {}
        
        for filename in os.listdir(backup_dir):
            if not filename.endswith('.json'):
                continue
            
            # Extract user_id from filename
            parts = filename.replace('_profile_', '|').split('|')
            if len(parts) == 2:
                user_id = parts[0]
                if user_id not in backups_by_user:
                    backups_by_user[user_id] = []
                backups_by_user[user_id].append(filename)
        
        # Delete old backups
        for user_id, backups in backups_by_user.items():
            if len(backups) > max_backups_per_user:
                # Sort by modification time
                backup_paths = [
                    os.path.join(backup_dir, b) for b in backups
                ]
                backup_paths.sort(key=lambda p: os.path.getmtime(p))
                
                # Delete oldest backups
                for path in backup_paths[:-max_backups_per_user]:
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Error deleting backup {path}: {e}")
