import logging
import os
import traceback
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from queue import Queue
from termcolor import colored
import pprint


class Logger:
    def __init__(
        self, name=__name__, log_file="log/myapp.log", log_level=logging.DEBUG
    ):
        # Create the log folder if it doesn't exist
        if not os.path.exists("log"):
            os.makedirs("log")

        # Configure the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(process)s - %(message)s - [%(filename)s:%(lineno)d - %(funcName)s]"
        )

        # StreamHandler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # FileHandler
        fh = TimedRotatingFileHandler(log_file, when="midnight")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # QueueHandler and QueueListener for thread safety
        log_queue = Queue(-1)
        qh = QueueHandler(log_queue)
        qh.setLevel(log_level)
        qh.setFormatter(formatter)
        self.logger.addHandler(qh)

        self.ql = QueueListener(log_queue, qh)
        self.ql.start()

    def debug(self, message):
        self.logger.debug(colored(self.pretty(message), "blue"))

    def info(self, message):
        self.logger.info(colored(self.pretty(message), "yellow"))

    def warning(self, message):
        self.logger.warning(colored(self.pretty(message), "magenta"))

    def error(self, message, exc_info=False):
        tb = traceback.format_exc()
        self.logger.error(colored(f"{self.pretty(message)}\n{tb}", "red"))

    def exception(self, message):
        self.logger.exception(colored(self.pretty(message), "red"))

    def stop(self):
        self.ql.stop()

    @staticmethod
    def pretty(message):
        if isinstance(message, dict) or isinstance(message, list):
            return pprint.pformat(message)
        else:
            return message


logger = Logger()
