

import logging
import pprint
from typing import Any, Dict

class Logger:
    @staticmethod
    def setup_logging(log_file: str = None) -> logging.Logger:
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Formatter for console
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

        # Console handler (info level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Formatter for file
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        # File handler (debug level)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger
    
    @staticmethod
    def debug(message: Any) -> None:
        logger = Logger.setup_logging()
        logger.debug(pprint.pformat(message))
        
    @staticmethod
    def info(message: Any) -> None:
        logger = Logger.setup_logging()
        logger.info(pprint.pformat(message))
        
    @staticmethod
    def warning(message: Any) -> None:
        logger = Logger.setup_logging()
        logger.warning(pprint.pformat(message))
        
    @staticmethod
    def error(message: Dict[str, Any]) -> None:
        logger = Logger.setup_logging()
        logger.error(pprint.pformat(message))
