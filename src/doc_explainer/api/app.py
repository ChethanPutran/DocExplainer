"""FastAPI application factory: the backend as a service."""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .errors import ServiceError
from .http_log import HttpLogMiddleware
from .rest import build_rest_router
from .rpc import build_rpc_router
from .service import API_VERSION, DocumentApiService, _version
from .settings import ApiSettings

logger = logging.getLogger(__name__)


def create_app(service: Optional[DocumentApiService] = None,
               settings: Optional[ApiSettings] = None) -> FastAPI:
    """Build the app. Pass ``service`` to inject one (tests); otherwise the real system is
    built at startup from the server's own config."""
    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            from .factory import build_system  # heavy: loads models
            logger.info("Building the system (data dir: %s)", settings.data_dir)
            app.state.service = DocumentApiService(build_system(settings), settings)
            logger.info("Doc Explainer API ready (protocols: %s)", ", ".join(settings.protocols))
        yield
        if getattr(app.state, "service", None) is not None:
            app.state.service.close()

    app = FastAPI(title="Doc Explainer API", version=_version(), lifespan=lifespan,
                  description=f"API {API_VERSION}. REST under /v1, JSON-RPC 2.0 at /rpc.")
    app.state.settings = settings
    app.state.service = service
    app.add_middleware(HttpLogMiddleware)       # silent until the server sets up http_logs.log

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.http_status,
                            content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
                   for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {
            "code": "invalid_params", "message": "Invalid request.", "details": details}})

    if settings.enable_rest:
        app.include_router(build_rest_router())
    if settings.enable_rpc:
        app.include_router(build_rpc_router())
    return app
