from dataclasses import dataclass


@dataclass
class Theme:
    """Base theme class"""
    name: str = "base"
    
    # Background colors
    background_primary: str = "#ffffff"
    background_secondary: str = "#f5f5f5"
    background_tertiary: str = "#e9ecef"
    
    # Text colors
    text_primary: str = "#212529"
    text_secondary: str = "#6c757d"
    text_disabled: str = "#adb5bd"
    text_on_accent: str = "#ffffff"
    
    # Accent colors
    accent_primary: str = "#007bff"
    accent_secondary: str = "#0056b3"
    accent_tertiary: str = "#003d80"
    
    # Status colors
    success: str = "#28a745"
    warning: str = "#ffc107"
    error: str = "#dc3545"
    info: str = "#17a2b8"
    
    # Border colors
    border_color: str = "#dee2e6"
    
    # Shadow
    shadow_color: str = "rgba(0,0,0,0.1)"


@dataclass
class LightTheme(Theme):
    """Light theme"""
    name: str = "light"
    background_primary: str = "#ffffff"
    background_secondary: str = "#f8f9fa"
    background_tertiary: str = "#e9ecef"
    text_primary: str = "#212529"
    text_secondary: str = "#6c757d"
    accent_primary: str = "#007bff"


@dataclass
class DarkTheme(Theme):
    """Dark theme"""
    name: str = "dark"
    background_primary: str = "#1e1e1e"
    background_secondary: str = "#2d2d2d"
    background_tertiary: str = "#3d3d3d"
    text_primary: str = "#ffffff"
    text_secondary: str = "#b0b0b0"
    text_disabled: str = "#808080"
    text_on_accent: str = "#ffffff"
    accent_primary: str = "#007bff"
    accent_secondary: str = "#0056b3"
    border_color: str = "#404040"


@dataclass
class HighContrastTheme(Theme):
    """High contrast theme"""
    name: str = "high_contrast"
    background_primary: str = "#000000"
    background_secondary: str = "#1a1a1a"
    background_tertiary: str = "#333333"
    text_primary: str = "#ffffff"
    text_secondary: str = "#ffff00"
    text_disabled: str = "#808080"
    text_on_accent: str = "#000000"
    accent_primary: str = "#ffff00"
    accent_secondary: str = "#ffaa00"
    border_color: str = "#ffffff"
    success: str = "#00ff00"
    warning: str = "#ffff00"
    error: str = "#ff0000"
    info: str = "#00ffff"


@dataclass
class SepiaTheme(Theme):
    """Sepia theme for reading"""
    name: str = "sepia"
    background_primary: str = "#fbf0d9"
    background_secondary: str = "#f5e6c9"
    background_tertiary: str = "#e8d5b5"
    text_primary: str = "#5f4b32"
    text_secondary: str = "#8b7a5a"
    accent_primary: str = "#8b4513"
    accent_secondary: str = "#654321"
    border_color: str = "#d2b48c"