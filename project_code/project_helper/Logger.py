import logging
import logging.handlers
import os
import pprint
from typing import Any, Dict


class Logger:
    """
    A static helper class for logging in Python.

    The `Logger` class provides methods for logging messages to the console and/or a file, using the standard Python
    `logging` module. It also provides a utility function for pretty-printing Python objects using `pprint`.

    Usage:
        - Call `Logger.debug(message)` to log a debug-level message.
        - Call `Logger.info(message)` to log an info-level message.
        - Call `Logger.warning(message)` to log a warning-level message.
        - Call `Logger.error(message)` to log an error-level message.

    Example:
        ```
        from logger import Logger

        # Log a debug message
        Logger.debug('This is a debug message')

        # Log an info message
        data = {'name': 'John', 'age': 30}
        Logger.info(data)

        # Log a warning message
        Logger.warning('This is a warning message')

        # Log an error message
        error = {'code': 500, 'message': 'Internal Server Error'}
        Logger.error(error)
        ```
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = cls.setup_logging()
        return cls._instance

    @staticmethod
    def setup_logging(log_file: str = "app.log") -> logging.Logger:
        """
        Set up logging to the console and a file.

        Args:
            log_file (str): Path to the log file.

        Returns:
            logging.Logger: A logger object with the specified logging configuration.
        """


        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Formatter for console
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")

        # Console handler (info level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Formatter for file
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # File handler (debug level)
        file_handler = logging.handlers.RotatingFileHandler(os.path.join(logs_dir, log_file), mode="a", maxBytes=1000000, backupCount=2)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Add a separator to the log file
        logger.handlers[-1].doRollover()
        logger.handlers[-1].stream.write("\n====================\n\n")

        return logger


    @staticmethod
    def debug(message: Any) -> None:
        """
        Log a debug-level message.

        Args:
            message (Any): The message to log. This can be any Python object.

        Returns:
            None
        """
        Logger()._instance.logger.debug(pprint.pformat(message))

    @staticmethod
    def info(message: Any) -> None:
        """
        Log an info-level message.

        Args:
            message (Any): The message to log. This can be any Python object.

        Returns:
            None
        """
        Logger()._instance.logger.info(pprint.pformat(message))

    @staticmethod
    def warning(message: Any) -> None:
        """
        Log a warning-level message.

        Args:
            message (Any): The message to log. This can be any Python object.

        Returns:
            None
        """
        Logger()._instance.logger.warning(pprint.pformat(message))

    @staticmethod
    def error(message: Dict[str, Any]) -> None:
        """
        Log an error-level message.

        Args:
            message (Dict[str, Any]): A dictionary representing the error message. This should contain at least a
                                      'message' key with a string value describing the error.

        Returns:
                None
        """
        logger = Logger()._instance.logger
        logger.error(pprint.pformat(message))
