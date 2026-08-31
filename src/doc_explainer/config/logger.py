import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from doc_explainer.config.backend import BackendConfig


def setup_logging(config: BackendConfig) -> None:
    """Configure application logging."""

    log_config = config.logging

    level = getattr(
        logging,
        log_config.level.upper(),
        logging.INFO,
    )

    handlers = []

    if log_config.console_enabled:
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)

    if log_config.file_enabled:
        log_dir = log_config.log_directory
        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / log_config.log_file

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.max_file_size_mb * 1024 * 1024,
            backupCount=log_config.backup_count,
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