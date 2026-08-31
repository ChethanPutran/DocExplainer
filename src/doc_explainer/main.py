#!/usr/bin/env python3
"""
Doc Explainer Application
Main entry point for the GUI application
"""

import sys
import argparse
import logging
import shutil
from pathlib import Path

from doc_explainer.config.backend import BackendConfig
from doc_explainer.config.logger import setup_logging


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def configure_logging(log_file: Path, log_level: str) -> logging.Logger:
    """
    Configure application logging.

    IMPORTANT:
    - Root logger stays at INFO so third-party libraries do not emit DEBUG logs.
    - Only the doc_explainer namespace follows the requested log level.
    """

    requested_level = getattr(logging, log_level.upper())

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler(sys.stdout)

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # Root logger
    #
    # Keep third-party libraries at INFO.
    # This prevents DEBUG logs from:
    #   httpcore
    #   httpx
    #   chromadb
    #   urllib3
    #   etc.
    # ------------------------------------------------------------------

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove handlers installed by previous logging configuration.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Our application
    #
    # doc_explainer.DEBUG is allowed when --log-level DEBUG is used.
    # ------------------------------------------------------------------

    app_logger = logging.getLogger("doc_explainer")
    app_logger.setLevel(requested_level)
    app_logger.propagate = True

    # ------------------------------------------------------------------
    # Explicitly keep noisy third-party libraries quiet.
    # ------------------------------------------------------------------

    third_party_loggers = [
        "httpcore",
        "httpx",
        "chromadb",
        "urllib3",
        "openai",
        "google",
        "PIL",
        "multipart",
        "asyncio",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""

    parser = argparse.ArgumentParser(
        description="Doc Explainer - Intelligent Document Explanation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.pdf
    %(prog)s --uiconfig ui.yaml
    %(prog)s --llmconfig llm.yaml
    %(prog)s --backendconfig backend.yaml
    %(prog)s --debug --no-splash document.html
    %(prog)s --theme dark document.txt
    %(prog)s --log-level DEBUG document.pdf
    %(prog)s --profile document.pdf
    %(prog)s --reset-config
    %(prog)s --clear-cache
        """,
    )

    parser.add_argument(
        "document",
        nargs="?",
        help="Document to open (PDF, TXT, HTML, etc.)",
    )

    parser.add_argument(
        "--uiconfig",
        "-c",
        help="UI configuration file path",
    )

    parser.add_argument(
        "--llmconfig",
        "-l",
        help="LLM configuration file path",
    )

    parser.add_argument(
        "--backendconfig",
        "-b",
        help="Backend configuration file path",
    )

    parser.add_argument(
        "--theme",
        "-t",
        choices=["light", "dark", "high_contrast", "sepia"],
        help="Theme to use",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable application debug mode",
    )

    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Disable splash screen",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set application logging level",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="Doc Explainer 1.0.0",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling",
    )

    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear application cache",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point."""

    # ------------------------------------------------------------------
    # Parse command-line arguments FIRST
    # ------------------------------------------------------------------

    parser = create_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Application directories
    # ------------------------------------------------------------------

    base_dir = Path.home() / ".doc_explainer"

    config_dir = base_dir / "config"
    log_dir = base_dir / "logs"
    resource_dir = base_dir / "resources"
    cache_dir = base_dir / "cache"

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    resource_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "app.log"

    # ------------------------------------------------------------------
    # Initial logging
    #
    # This is configured before BackendConfig because we need logging
    # while loading the backend configuration.
    # ------------------------------------------------------------------

    logger = configure_logging(
        log_file=log_file,
        log_level=args.log_level,
    )

    logger.info("=" * 70)
    logger.info("Starting Doc Explainer")
    logger.info("Command line arguments: %s", args)
    logger.info("Application log level: %s", args.log_level)
    logger.info("Log file: %s", log_file)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load backend configuration
    # ------------------------------------------------------------------

    backend_config_path = (
        Path(args.backendconfig).expanduser()
        if args.backendconfig
        else config_dir / "backend_config.yaml"
    )

    try:
        logger.info(
            "Loading backend configuration from %s",
            backend_config_path,
        )

        backend_config = BackendConfig.load(
            filepath=str(backend_config_path)
        )

        backend_config.validate()

        logger.info(
            "Backend configuration loaded successfully"
        )

    except Exception:
        logger.exception(
            "Failed to load backend configuration"
        )
        return 1

    # ------------------------------------------------------------------
    # Backend logging configuration
    #
    # setup_logging() may modify the root logger.
    # Therefore we restore our desired logging policy afterwards.
    # ------------------------------------------------------------------

    try:
        setup_logging(backend_config)

        # Re-apply our logging policy after setup_logging().
        logger = configure_logging(
            log_file=log_file,
            log_level=args.log_level,
        )

        logger.info("Logging initialized")
        logger.info("Application log level: %s", args.log_level)
        logger.info("Log file: %s", log_file)

    except Exception:
        logger.exception(
            "Failed to initialize application logging"
        )
        return 1

    # ------------------------------------------------------------------
    # Handle reset configuration
    # ------------------------------------------------------------------

    if args.reset_config:

        config_file = config_dir / "ui_config.json"

        if config_file.exists():

            try:
                config_file.unlink()

                logger.info(
                    "Configuration reset to defaults: %s",
                    config_file,
                )

            except Exception:
                logger.exception(
                    "Failed to reset configuration"
                )
                return 1

        else:

            logger.info(
                "No UI configuration file found: %s",
                config_file,
            )

        return 0

    # ------------------------------------------------------------------
    # Handle cache clearing
    # ------------------------------------------------------------------

    if args.clear_cache:

        try:

            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            logger.info(
                "Application cache cleared: %s",
                cache_dir,
            )

        except Exception:
            logger.exception(
                "Failed to clear application cache"
            )
            return 1

        return 0

    # ------------------------------------------------------------------
    # Copy application assets
    # ------------------------------------------------------------------

    try:

        if not resource_dir.exists() or not any(
            resource_dir.iterdir()
        ):

            source_path = (
                Path(__file__).parents[3] / "assets"
            )

            logger.info(
                "Looking for application assets at %s",
                source_path,
            )

            if source_path.exists():

                shutil.copytree(
                    source_path,
                    resource_dir,
                    dirs_exist_ok=True,
                )

                logger.info(
                    "Copied application assets to %s",
                    resource_dir,
                )

            else:

                logger.warning(
                    "Source assets directory not found: %s",
                    source_path,
                )

        else:

            logger.info(
                "Assets already exist at %s",
                resource_dir,
            )

    except Exception:
        logger.exception(
            "Failed to initialize application assets"
        )
        return 1

    # ------------------------------------------------------------------
    # Start GUI application
    # ------------------------------------------------------------------

    app = None

    try:

        from doc_explainer.ui.gui.app import DocExplainerApp

        logger.info("Creating DocExplainerApp")

        app = DocExplainerApp(
            config_paths={
                "ui": args.uiconfig,
                "llm": args.llmconfig,
                "backend": args.backendconfig,
            },
            debug=args.debug,
        )

        logger.info(
            "DocExplainerApp created successfully"
        )

        # --------------------------------------------------------------
        # Theme
        # --------------------------------------------------------------

        if args.theme:

            logger.info(
                "Command-line theme requested: %s",
                args.theme,
            )

            if hasattr(app, "theme_override"):
                app.theme_override = args.theme

        # --------------------------------------------------------------
        # Run application
        # --------------------------------------------------------------

        if args.profile:

            logger.info(
                "Performance profiling enabled"
            )

            import cProfile
            import pstats
            from io import StringIO

            profiler = cProfile.Profile()

            profiler.enable()

            try:

                exit_code = app.run(
                    document_path=args.document
                )

            finally:

                profiler.disable()

            profile_path = (
                base_dir / "profile.stats"
            )

            profiler.dump_stats(profile_path)

            logger.info(
                "Profile statistics saved to %s",
                profile_path,
            )

            # Print top functions to terminal
            output = StringIO()

            stats = pstats.Stats(
                profiler,
                stream=output,
            ).sort_stats("cumulative")

            stats.print_stats(20)

            print(output.getvalue())

        else:

            logger.info(
                "Starting application"
            )

            exit_code = app.run(
                document_path=args.document
            )

        logger.info(
            "Application exited with code %s",
            exit_code,
        )

        return exit_code

    except KeyboardInterrupt:

        logger.info(
            "Application interrupted by user"
        )

        return 130

    except Exception:

        logger.exception(
            "Fatal error while running Doc Explainer"
        )

        return 1

    finally:

        # --------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------

        try:

            if app is not None:

                logger.info(
                    "Cleaning up application resources"
                )

                app.cleanup()

        except Exception:

            logger.exception(
                "Error during application cleanup"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
