import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

# create a custom logger instance
logger = logging.getLogger(__name__)

# create a queue to hold log records
log_queue = Queue(-1)

# create a QueueHandler that sends log records to the queue
queue_handler = QueueHandler(log_queue)

# add the queue handler to the logger
logger.addHandler(queue_handler)


def process_log_queue():
    """
    Process log records from the queue and emit them to the logger.
    """
    while True:
        try:
            record = log_queue.get()
            if record is None:
                break
            logger.handle(record)
        except Exception:
            logger.exception("Error processing log queue")


# create a QueueListener to manage the processing of log records from the queue
queue_listener = QueueListener(log_queue, process_log_queue)

