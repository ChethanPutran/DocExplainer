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

    # Added for the refined look (all optional: older code ignores them)
    accent_soft: str = "#e3e9ff"          # light tint of the accent: selections, hovers, checked toolbar buttons
    hover: str = "#eceff4"                # neutral hover background
    font_family: str = '"Inter", "Segoe UI", "Noto Sans", "Helvetica Neue", "DejaVu Sans", sans-serif'
    font_size: int = 10                   # points
    radius: int = 6                       # px


@dataclass
class LightTheme(Theme):
    """Light theme: soft grey window, white surfaces, one indigo accent."""
    name: str = "light"
    background_primary: str = "#f4f5f8"
    background_secondary: str = "#ffffff"
    background_tertiary: str = "#e8ebf1"
    text_primary: str = "#1d2230"
    text_secondary: str = "#5f6b7d"
    text_disabled: str = "#a3acba"
    accent_primary: str = "#3d5afe"
    accent_secondary: str = "#2f47d1"
    accent_tertiary: str = "#243aa8"
    accent_soft: str = "#e4e9ff"
    hover: str = "#eceff6"
    border_color: str = "#dde1ea"
    success: str = "#1f9d55"
    warning: str = "#d98a00"
    error: str = "#d63a3a"
    info: str = "#1c8fb0"


@dataclass
class DarkTheme(Theme):
    """Dark theme: charcoal, slightly lifted surfaces, a brighter accent."""
    name: str = "dark"
    background_primary: str = "#14171c"
    background_secondary: str = "#1b1f26"
    background_tertiary: str = "#262b35"
    text_primary: str = "#e7eaf0"
    text_secondary: str = "#9aa4b5"
    text_disabled: str = "#5d6675"
    text_on_accent: str = "#ffffff"
    accent_primary: str = "#5b78ff"
    accent_secondary: str = "#4a64e0"
    accent_tertiary: str = "#3a50bb"
    accent_soft: str = "#242d52"
    hover: str = "#222733"
    border_color: str = "#2d333f"
    success: str = "#3fc380"
    warning: str = "#e7a93a"
    error: str = "#ef6b6b"
    info: str = "#4cb8d6"


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
    accent_soft: str = "#333300"
    hover: str = "#262626"


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
    accent_soft: str = "#efd9b0"
    hover: str = "#efdcb8"