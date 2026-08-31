"""
Tests for User Profile Repository.

Tests cover:
- Profile persistence
- CRUD operations
- Import/export functionality
- Backup and recovery
"""

import pytest
import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta

from src.core.user.repository.user_profile_repository import UserProfileRepository
from src.core.user.models.user_profile import (
    UserProfile,
    LearningPreferences,
    LearningPace,
    ExplanationDepth
)


class TestUserProfileRepository:
    """Test suite for UserProfileRepository."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def repository(self, temp_dir):
        """Create a repository with temporary storage."""
        return UserProfileRepository(storage_path=os.path.join(temp_dir, "profiles/"))
    
    # ==================== Basic CRUD Tests ====================
    
    def test_repository_initialization(self, repository):
        """Test repository initializes with correct structure."""
        assert repository.storage_path.endswith("profiles/")
        assert os.path.exists(repository.storage_path)
    
    def test_save_profile(self, repository):
        """Test saving a profile."""
        profile = UserProfile("user_001")
        profile.update_concept_mastery("Python", 0.9)
        
        saved = repository.save_profile(profile)
        assert saved.user_id == "user_001"
        assert "Python" in saved.known_concepts
    
    def test_save_profile_creates_file(self, repository):
        """Test that save_profile creates a file."""
        profile = UserProfile("user_002")
        repository.save_profile(profile)
        
        filepath = repository._get_profile_path("user_002")
        assert os.path.exists(filepath)
    
    def test_get_profile(self, repository):
        """Test retrieving a saved profile."""
        profile = UserProfile("user_003")
        profile.update_concept_mastery("Java", 0.8)
        repository.save_profile(profile)
        
        retrieved = repository.get_profile("user_003")
        assert retrieved is not None
        assert retrieved.user_id == "user_003"
        assert "Java" in retrieved.known_concepts
        assert retrieved.known_concepts["Java"].p_knowledge == 0.8
    
    def test_get_nonexistent_profile(self, repository):
        """Test retrieving a non-existent profile."""
        retrieved = repository.get_profile("nonexistent_user")
        assert retrieved is None
    
    def test_profile_exists(self, repository):
        """Test checking if profile exists."""
        profile = UserProfile("user_004")
        repository.save_profile(profile)
        
        assert repository.profile_exists("user_004") is True
        assert repository.profile_exists("nonexistent") is False
    
    def test_delete_profile(self, repository):
        """Test deleting a profile."""
        profile = UserProfile("user_005")
        repository.save_profile(profile)
        
        result = repository.delete_profile("user_005")
        assert result is True
        assert repository.profile_exists("user_005") is False
    
    def test_delete_nonexistent_profile(self, repository):
        """Test deleting a non-existent profile."""
        result = repository.delete_profile("nonexistent")
        assert result is False
    
    def test_list_user_ids(self, repository):
        """Test listing all user IDs."""
        for i in range(3):
            profile = UserProfile(f"user_{i:03d}")
            repository.save_profile(profile)
        
        users = repository.list_user_ids()
        assert len(users) == 3
        assert "user_000" in users
        assert "user_001" in users
        assert "user_002" in users
    
    # ==================== Caching Tests ====================
    
    def test_cache_on_save(self, repository):
        """Test that profiles are cached after save."""
        profile = UserProfile("user_cache_001")
        repository.save_profile(profile)
        
        assert "user_cache_001" in repository.cache
    
    def test_cache_on_get(self, repository):
        """Test that profiles are cached after get."""
        profile = UserProfile("user_cache_002")
        repository.save_profile(profile)
        repository.cache.clear()
        
        repository.get_profile("user_cache_002")
        assert "user_cache_002" in repository.cache
    
    def test_clear_cache(self, repository):
        """Test clearing the cache."""
        profile = UserProfile("user_cache_003")
        repository.save_profile(profile)
        
        repository.clear_cache()
        assert len(repository.cache) == 0
    
    def test_reload_profile_bypasses_cache(self, repository):
        """Test that reload bypasses cache."""
        profile = UserProfile("user_reload")
        profile.update_concept_mastery("Python", 0.5)
        repository.save_profile(profile)
        
        # Modify cache
        repository.cache["user_reload"].known_concepts["Python"].p_knowledge = 0.9
        
        # Reload should get fresh data from disk
        reloaded = repository.reload_profile("user_reload")
        assert reloaded.known_concepts["Python"].p_knowledge == 0.5
    
    # ==================== Backup Tests ====================
    
    def test_backup_created_on_update(self, repository):
        """Test that backup is created when updating profile."""
        profile = UserProfile("user_backup")
        repository.save_profile(profile)
        
        # Update profile
        profile.update_concept_mastery("Python", 0.8)
        repository.save_profile(profile)
        
        # Check backup exists
        backup_dir = os.path.join(repository.storage_path, "backups")
        backups = os.listdir(backup_dir)
        assert len(backups) > 0
    
    def test_cleanup_old_backups(self, repository):
        """Test cleanup of old backup files."""
        profile = UserProfile("user_backup_cleanup")
        
        # Create multiple backups
        for i in range(7):
            profile.update_concept_mastery("Python", 0.1 * i)
            repository.save_profile(profile)
        
        # Cleanup, keeping only 5
        repository.cleanup_old_backups(max_backups_per_user=5)
        
        backup_dir = os.path.join(repository.storage_path, "backups")
        backups = [f for f in os.listdir(backup_dir) if "user_backup_cleanup" in f]
        assert len(backups) <= 5
    
    # ==================== Export/Import Tests ====================
    
    def test_export_profile(self, repository, temp_dir):
        """Test exporting profile to file."""
        profile = UserProfile("user_export")
        profile.update_concept_mastery("Python", 0.9)
        repository.save_profile(profile)
        
        export_path = os.path.join(temp_dir, "export.json")
        result = repository.export_profile("user_export", export_path)
        
        assert result is True
        assert os.path.exists(export_path)
    
    def test_export_nonexistent_profile(self, repository, temp_dir):
        """Test exporting non-existent profile."""
        export_path = os.path.join(temp_dir, "export.json")
        result = repository.export_profile("nonexistent", export_path)
        
        assert result is False
        assert not os.path.exists(export_path)
    
    def test_import_profile(self, repository, temp_dir):
        """Test importing profile from file."""
        # Create and export a profile
        profile = UserProfile("user_import_source")
        profile.update_concept_mastery("Python", 0.85)
        profile.preferences.learning_pace = LearningPace.FAST
        
        export_path = os.path.join(temp_dir, "export.json")
        export_data = {
            'profile': profile.to_dict(),
            'exported_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f)
        
        # Import
        imported = repository.import_profile(export_path)
        
        assert imported is not None
        assert imported.user_id == "user_import_source"
        assert "Python" in imported.known_concepts
        assert imported.preferences.learning_pace == LearningPace.FAST
    
    def test_import_direct_profile_data(self, repository, temp_dir):
        """Test importing direct profile data without wrapper."""
        profile = UserProfile("user_direct_import")
        profile.update_concept_mastery("Java", 0.7)
        
        export_path = os.path.join(temp_dir, "direct_export.json")
        
        with open(export_path, 'w') as f:
            json.dump(profile.to_dict(), f)
        
        imported = repository.import_profile(export_path)
        
        assert imported is not None
        assert imported.user_id == "user_direct_import"
        assert "Java" in imported.known_concepts
    
    def test_import_nonexistent_file(self, repository):
        """Test importing from non-existent file."""
        imported = repository.import_profile("/nonexistent/path/profile.json")
        assert imported is None
    
    def test_round_trip_export_import(self, repository, temp_dir):
        """Test that export followed by import preserves data."""
        # Create profile with data
        profile = UserProfile("user_roundtrip")
        profile.update_concept_mastery("Python", 0.95)
        profile.update_concept_mastery("Java", 0.65)
        profile.preferences.learning_pace = LearningPace.FAST
        profile.preferences.explanation_depth = ExplanationDepth.COMPREHENSIVE
        repository.save_profile(profile)
        
        # Export
        export_path = os.path.join(temp_dir, "roundtrip.json")
        repository.export_profile("user_roundtrip", export_path)
        
        # Import to new repository
        import_repo = UserProfileRepository(
            storage_path=os.path.join(temp_dir, "import_profiles/")
        )
        imported = import_repo.import_profile(export_path)
        
        # Verify
        assert imported.user_id == "user_roundtrip"
        assert len(imported.known_concepts) == 2
        assert imported.known_concepts["Python"].p_knowledge == 0.95
        assert imported.preferences.learning_pace == LearningPace.FAST
    
    # ==================== Query Tests ====================
    
    def test_get_profile_statistics(self, repository):
        """Test getting statistics for a profile."""
        profile = UserProfile("user_stats")
        profile.update_concept_mastery("Python", 0.9)
        profile.update_concept_mastery("Java", 0.5)
        repository.save_profile(profile)
        
        stats = repository.get_profile_statistics("user_stats")
        
        assert stats['user_id'] == "user_stats"
        assert stats['total_concepts'] == 2
        assert stats['known_concepts_count'] == 1
    
    def test_get_all_statistics(self, repository):
        """Test getting statistics for all profiles."""
        for i in range(3):
            profile = UserProfile(f"user_multi_{i}")
            profile.update_concept_mastery("Python", 0.5 + i * 0.1)
            repository.save_profile(profile)
        
        all_stats = repository.get_all_statistics()
        
        assert all_stats['total_users'] == 3
        assert len(all_stats['profiles']) == 3
    
    def test_find_profiles_by_mastery_range(self, repository):
        """Test finding profiles by mastery range."""
        for i in range(3):
            profile = UserProfile(f"user_mastery_{i}")
            profile.update_concept_mastery("Python", 0.3 + i * 0.2)
            repository.save_profile(profile)
        
        # Find profiles with mastery between 0.4 and 0.7
        matching = repository.find_profiles_by_mastery_range(min_mastery=0.4, max_mastery=0.7)
        
        assert len(matching) >= 1
        assert "user_mastery_1" in matching
    
    def test_get_profiles_by_preference(self, repository):
        """Test finding profiles by preference."""
        for i in range(3):
            profile = UserProfile(f"user_pref_{i}")
            if i % 2 == 0:
                profile.preferences.learning_pace = LearningPace.FAST
            else:
                profile.preferences.learning_pace = LearningPace.NORMAL
            repository.save_profile(profile)
        
        fast_learners = repository.get_profiles_by_preference(
            'learning_pace',
            LearningPace.FAST.value
        )
        
        assert len(fast_learners) >= 1
        assert "user_pref_0" in fast_learners
    
    # ==================== File Format Tests ====================
    
    def test_saved_file_is_valid_json(self, repository):
        """Test that saved profile is valid JSON."""
        profile = UserProfile("user_json_test")
        profile.update_concept_mastery("Python", 0.8)
        repository.save_profile(profile)
        
        filepath = repository._get_profile_path("user_json_test")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert data['user_id'] == "user_json_test"
        assert 'known_concepts' in data
    
    def test_saved_profile_includes_all_fields(self, repository):
        """Test that saved profile includes all necessary fields."""
        profile = UserProfile("user_fields_test")
        profile.update_concept_mastery("Python", 0.9)
        profile.preferences.learning_pace = LearningPace.FAST
        repository.save_profile(profile)
        
        retrieved = repository.get_profile("user_fields_test")
        
        assert retrieved.user_id == "user_fields_test"
        assert len(retrieved.known_concepts) > 0
        assert retrieved.preferences.learning_pace == LearningPace.FAST
    
    # ==================== Error Handling Tests ====================
    
    def test_corrupted_file_returns_none(self, repository, temp_dir):
        """Test handling of corrupted profile file."""
        # Create corrupted file
        filepath = repository._get_profile_path("corrupted")
        with open(filepath, 'w') as f:
            f.write("{ invalid json")
        
        result = repository.get_profile("corrupted")
        assert result is None
    
    def test_missing_required_fields_handled(self, repository, temp_dir):
        """Test handling of profile with missing fields."""
        filepath = repository._get_profile_path("incomplete")
        
        with open(filepath, 'w') as f:
            json.dump({'user_id': 'incomplete'}, f)
        
        result = repository.get_profile("incomplete")
        assert result is not None
        assert result.user_id == "incomplete"
    
    # ==================== Persistence Integration Tests ====================
    
    def test_persistence_across_instances(self, temp_dir):
        """Test that data persists across repository instances."""
        storage_path = os.path.join(temp_dir, "persistent/")
        
        # Create and save with first instance
        repo1 = UserProfileRepository(storage_path=storage_path)
        profile = UserProfile("user_persistent")
        profile.update_concept_mastery("Python", 0.95)
        repo1.save_profile(profile)
        
        # Create new instance and verify data exists
        repo2 = UserProfileRepository(storage_path=storage_path)
        retrieved = repo2.get_profile("user_persistent")
        
        assert retrieved is not None
        assert retrieved.known_concepts["Python"].p_knowledge == 0.95
    
    def test_concurrent_profile_operations(self, repository):
        """Test handling of rapid profile operations."""
        profile = UserProfile("user_concurrent")
        
        # Perform rapid updates
        for i in range(10):
            profile.update_concept_mastery(f"Concept_{i}", 0.1 * i)
            repository.save_profile(profile)
        
        # Verify final state
        retrieved = repository.get_profile("user_concurrent")
        assert len(retrieved.known_concepts) == 10
