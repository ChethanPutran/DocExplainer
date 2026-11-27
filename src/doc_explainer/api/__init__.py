"""Doc Explainer API: the backend as a service (REST under /v1 and JSON-RPC 2.0 at /rpc)."""
from .app import create_app
from .settings import ApiSettings

__all__ = ["create_app", "ApiSettings"]
