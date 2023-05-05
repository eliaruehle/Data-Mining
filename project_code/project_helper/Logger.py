import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue


class Logger:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.logger = cls.__configure_logger()
        return cls.__instance

    @staticmethod
    def __configure_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(process)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        log_queue = Queue(-1)
        qh = QueueHandler(log_queue)
        qh.setLevel(logging.DEBUG)
        qh.setFormatter(formatter)
        logger.addHandler(qh)

        ql = QueueListener(log_queue, qh)
        ql.start()

        return logger

    @staticmethod
    def debug(message):
        Logger().__instance.logger.debug(message)

    @staticmethod
    def info(message):
        Logger().__instance.logger.info(message)

    @staticmethod
    def warning(message):
        Logger().__instance.logger.warning(message)

    @staticmethod
    def error(message):
        Logger().__instance.logger.error(message, exc_info=True)
