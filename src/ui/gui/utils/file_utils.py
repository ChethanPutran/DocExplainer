import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime


class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def get_file_size(path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(path)
    
    @staticmethod
    def get_file_size_mb(path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(path) / (1024 * 1024)
    
    @staticmethod
    def get_file_extension(path: str) -> str:
        """Get file extension"""
        return os.path.splitext(path)[1].lower()
    
    @staticmethod
    def get_file_name(path: str) -> str:
        """Get file name without extension"""
        return os.path.splitext(os.path.basename(path))[0]
    
    @staticmethod
    def get_file_info(path: str) -> Dict[str, Any]:
        """Get file information"""
        stat = os.stat(path)
        return {
            'path': path,
            'name': os.path.basename(path),
            'name_without_ext': FileUtils.get_file_name(path),
            'extension': FileUtils.get_file_extension(path),
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'accessed': datetime.fromtimestamp(stat.st_atime)
        }
    
    @staticmethod
    def is_text_file(path: str) -> bool:
        """Check if file is a text file"""
        text_extensions = ['.txt', '.text', '.md', '.rst', '.json', '.xml', '.html', '.htm', '.css', '.js', '.py']
        return FileUtils.get_file_extension(path) in text_extensions
    
    @staticmethod
    def is_pdf_file(path: str) -> bool:
        """Check if file is a PDF"""
        return FileUtils.get_file_extension(path) == '.pdf'
    
    @staticmethod
    def read_text_file(path: str, encoding: str = 'utf-8') -> Optional[str]:
        """Read text file"""
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return None
    
    @staticmethod
    def write_text_file(path: str, content: str, encoding: str = 'utf-8') -> bool:
        """Write text file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing file {path}: {e}")
            return False
    
    @staticmethod
    def read_json_file(path: str) -> Optional[Dict]:
        """Read JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {path}: {e}")
            return None
    
    @staticmethod
    def write_json_file(path: str, data: Dict, pretty: bool = True) -> bool:
        """Write JSON file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, f, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing JSON file {path}: {e}")
            return False
    
    @staticmethod
    def get_recent_files(directory: str, extensions: List[str] = None, limit: int = 10) -> List[str]:
        """Get recent files from directory"""
        if not os.path.exists(directory):
            return []
        
        files = []
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                if extensions and FileUtils.get_file_extension(path) not in extensions:
                    continue
                files.append((path, os.path.getmtime(path)))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in files[:limit]]
    
    @staticmethod
    def ensure_directory(path: str) -> bool:
        """Ensure directory exists"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {path}: {e}")
            return False
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """Convert filename to safe version"""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    @staticmethod
    def get_unique_filename(directory: str, filename: str) -> str:
        """Get unique filename by adding number if file exists"""
        name, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        
        while os.path.exists(os.path.join(directory, new_filename)):
            new_filename = f"{name}_{counter}{ext}"
            counter += 1
        
        return os.path.join(directory, new_filename)