import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from doc_explainer.config.backend import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure application logging."""


    level = getattr(
        logging,
        config.level.upper(),
        logging.INFO,
    )

    handlers = []

    if config.console_enabled:
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)

    if config.file_enabled:
        log_dir = config.log_directory
        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / config.log_file

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding="utf-8",
        )

        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s - "
            "%(name)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
        handlers=handlers,
        force=True,
    )

    # Keep dependency diagnostics out of the application console at DEBUG.
    for logger_name in (
        "neo4j",
        "google_genai",
        "httpcore",
        "httpx",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)