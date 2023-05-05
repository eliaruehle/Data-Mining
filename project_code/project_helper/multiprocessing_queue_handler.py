import multiprocessing

import Logger


class MultiProcessingQueueHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        try:
            self.queue.put(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a queue and a MultiProcessingQueueHandler
    queue = multiprocessing.Manager().Queue(-1)
    handler = MultiProcessingQueueHandler(queue)

    # add the handler to the logger
    logger.addHandler(handler)

    # create a separate process to handle logging messages
    process = multiprocessing.Process(target=log_worker, args=(queue,))
    process.daemon = True
    process.start()

    return logger

def log_worker(queue):
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logging.error('Error in log worker', exc_info=True)
