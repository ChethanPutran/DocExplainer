"""``doc-explainer-server``: run the backend as a service."""
from __future__ import annotations
import argparse
import ipaddress
import sys
from pathlib import Path
from typing import List, Optional

from .settings import ApiSettings


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doc-explainer-server",
        description="Doc Explainer backend service (REST under /v1, JSON-RPC 2.0 at /rpc).",
        epilog="Settings can also come from DOC_EXPLAINER_* environment variables "
               "(HOST, PORT, API_KEY, DATA_DIR, LLM_CONFIG, BACKEND_CONFIG, ...). "
               "Provider keys (OPENROUTER_API_KEY, AIROUTER_API_KEY, GEMINI_API_KEY, ...) come from the "
               "environment/.env; the provider itself is set in llm_config.yaml.")
    p.add_argument("--host", help="interface to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8000)")
    p.add_argument("--api-key", help="require this bearer token (prefer DOC_EXPLAINER_API_KEY: "
                                     "command lines are visible to other users)")
    p.add_argument("--data-dir", type=Path, help="where uploads, the registry and graphs are stored")
    p.add_argument("--llmconfig", help="LLM config file (default ~/.doc_explainer/config/llm_config.yaml)")
    p.add_argument("--backendconfig", help="backend config file")
    p.add_argument("--no-rest", action="store_true", help="do not serve the REST API")
    p.add_argument("--no-rpc", action="store_true", help="do not serve the JSON-RPC API")
    p.add_argument("--init-config", action="store_true",
                   help="create the default config files in ~/.doc_explainer/config and exit")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.init_config:
        from doc_explainer.config.bootstrap import DEFAULT_CONFIG_DIR, ensure_default_configs
        created = ensure_default_configs()
        for f in created:
            print(f"created {f}")
        if not created:
            print(f"nothing to do: config already exists in {DEFAULT_CONFIG_DIR}")
        return 0

    s = ApiSettings.from_env()
    if args.host: s.host = args.host
    if args.port: s.port = args.port
    if args.api_key: s.api_key = args.api_key
    if args.data_dir: s.data_dir = args.data_dir.expanduser()
    if args.llmconfig: s.llm_config_path = args.llmconfig
    if args.backendconfig: s.backend_config_path = args.backendconfig
    if args.no_rest: s.enable_rest = False
    if args.no_rpc: s.enable_rpc = False
    if not s.protocols:
        print("error: both protocols are disabled; nothing to serve", file=sys.stderr)
        return 2
    if not _is_loopback(s.host) and not s.api_key:
        print(f"error: refusing to listen on {s.host} without an API key.\n"
              "Set DOC_EXPLAINER_API_KEY (and use TLS in front: this server speaks plain HTTP).",
              file=sys.stderr)
        return 2

    import uvicorn
    from .app import create_app
    # access_log=False: the server writes its own, fuller request log to http_logs.log (api/http_log.py)
    uvicorn.run(create_app(settings=s), host=s.host, port=s.port, log_level=args.log_level, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
