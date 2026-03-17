from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class UIConfig:
    """UI configuration with additional settings"""
    
    # Window settings
    window_width: int = 1200
    window_height: int = 800
    window_title: str = "Doc Explainer"
    window_maximized: bool = False
    
    # Theme settings
    theme: str = "light"  # light, dark, high_contrast, sepia
    font_size: int = 10
    font_family: str = "Segoe UI"
    
    # Sidebar settings
    sidebar_width: int = 350
    sidebar_visible: bool = True
    sidebar_position: str = "right"  # left, right
    
    # Voice settings
    voice_enabled: bool = True
    voice_input_device: str = "default"
    voice_output_enabled: bool = True
    voice_output_rate: int = 150
    voice_output_volume: float = 0.9
    
    # Document settings
    default_zoom: float = 1.0
    max_recent_files: int = 10
    auto_save_interval: int = 5  # minutes
    
    # Cache settings
    cache_documents: bool = True
    cache_size_mb: int = 500
    cache_location: str = str(Path.home() / '.doc_explainer' / 'cache')
    
    # LLM settings
    llm_provider: str = "gemini"
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 1.0
    llm_max_tokens: Optional[int] = None
    llm_timeout: Optional[int] = None
    
    # API Keys (will be stored securely)
    gemini_api_key: str = ""
    openai_api_key: str = ""
    
    # Knowledge Graph settings
    kg_enabled: bool = True
    kg_auto_build: bool = True
    
    # Memory settings
    memory_enabled: bool = True
    session_tracking: bool = True
    
    # Startup settings
    open_last_docs: bool = True
    check_updates: bool = True
    
    # Debug settings
    debug_mode: bool = False
    log_level: str = "INFO"
    show_splash: bool = True
    enable_profiling: bool = False
    
    # Privacy settings
    send_usage_stats: bool = False
    allow_telemetry: bool = False
    
    # Custom paths
    documents_path: str = str(Path.home() / 'Documents')
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'UIConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'window_width': self.window_width,
            'window_height': self.window_height,
            'window_title': self.window_title,
            'window_maximized': self.window_maximized,
            'theme': self.theme,
            'font_size': self.font_size,
            'font_family': self.font_family,
            'sidebar_width': self.sidebar_width,
            'sidebar_visible': self.sidebar_visible,
            'sidebar_position': self.sidebar_position,
            'voice_enabled': self.voice_enabled,
            'voice_input_device': self.voice_input_device,
            'voice_output_enabled': self.voice_output_enabled,
            'voice_output_rate': self.voice_output_rate,
            'voice_output_volume': self.voice_output_volume,
            'default_zoom': self.default_zoom,
            'max_recent_files': self.max_recent_files,
            'auto_save_interval': self.auto_save_interval,
            'cache_documents': self.cache_documents,
            'cache_size_mb': self.cache_size_mb,
            'cache_location': self.cache_location,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'llm_max_tokens': self.llm_max_tokens,
            'llm_timeout': self.llm_timeout,
            'kg_enabled': self.kg_enabled,
            'kg_auto_build': self.kg_auto_build,
            'memory_enabled': self.memory_enabled,
            'session_tracking': self.session_tracking,
            'open_last_docs': self.open_last_docs,
            'check_updates': self.check_updates,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level,
            'show_splash': self.show_splash,
            'enable_profiling': self.enable_profiling,
            'send_usage_stats': self.send_usage_stats,
            'allow_telemetry': self.allow_telemetry,
            'documents_path': self.documents_path,
            # Don't include API keys in normal serialization
        }
    
    def save(self, filepath: Optional[str] = None):
        """Save configuration to file"""
        if filepath is None:
            filepath = str(Path.home() / '.doc_explainer' / 'config' / 'ui_config.json')
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)