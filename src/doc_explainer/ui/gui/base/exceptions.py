class UIError(Exception):
    """Base exception for UI module"""
    pass


class ViewerError(UIError):
    """Raised when document viewer operations fail"""
    pass


class WidgetError(UIError):
    """Raised when widget operations fail"""
    pass


class FileNotFoundError(UIError):
    """Raised when file is not found"""
    pass


class UnsupportedFileTypeError(UIError):
    """Raised when file type is not supported"""
    pass


class VoiceError(UIError):
    """Raised when voice operations fail"""
    pass