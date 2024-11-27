import logging
import os
from .config import LoggerConfig


class Logger:
    """Custom logger for the training process.

    Args:
        config (LoggerConfig): Configuration for the logger.
    """

    def __init__(self, config: LoggerConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("TrainerLogger")
        self.logger.setLevel(config.level)
        self._add_file_handler()
        self._add_stream_handler()

    def _add_file_handler(self) -> None:
        """Add a file handler to the logger."""
        os.makedirs(self.config.log_dir, exist_ok=True)  # Ensure log directory exists
        log_file_path = os.path.join(self.config.log_dir, self.config.file_name)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(self.config.level)
        file_handler.setFormatter(
            logging.Formatter(self.config.format, datefmt="%Y-%m-%d | %H:%M:%S")
        )
        self.logger.addHandler(file_handler)

    def _add_stream_handler(self) -> None:
        """Add a stream handler to the logger."""
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.config.level)
        stream_handler.setFormatter(
            logging.Formatter(self.config.format, datefmt="%Y-%m-%d | %H:%M:%S")
        )
        self.logger.addHandler(stream_handler)

    def info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)
