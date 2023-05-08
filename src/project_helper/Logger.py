import logging
import os
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import Queue

# Create the log folder if it doesn't exist
if not os.path.exists('log'):
    os.makedirs('log')

# Configure the module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(process)s - %(message)s"
)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = RotatingFileHandler('log/myapp.log', maxBytes=1024*1024, backupCount=10)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

log_queue = Queue(-1)
qh = QueueHandler(log_queue)
qh.setLevel(logging.DEBUG)
qh.setFormatter(formatter)
logger.addHandler(qh)

ql = QueueListener(log_queue, qh)
ql.start()

class Logger:
    @staticmethod
    def debug(message):
        logger.debug(message)

    @staticmethod
    def info(message):
        logger.info(message)

    @staticmethod
    def warning(message):
        logger.warning(message)

    @staticmethod
    def error(message):
        logger.error(message, exc_info=True)
