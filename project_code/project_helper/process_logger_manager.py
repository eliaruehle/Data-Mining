import logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener
from queue import Queue


class ProcessLoggerManager:
    def __init__(self):
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(process)s - %(message)s"
        )

        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Add queue handler
        log_queue = multiprocessing.Manager().Queue(-1)
        qh = QueueHandler(log_queue)
        qh.setLevel(logging.DEBUG)
        qh.setFormatter(formatter)
        logger.addHandler(qh)

        # Add queue listener
        ql = QueueListener(log_queue, qh)
        ql.start()

        return logger

    def run(self):
        # Create process queue
        process_queue = multiprocessing.Manager().Queue(-1)

        # Create process pool
        with multiprocessing.Pool(processes=4, initializer=self._setup_logger) as pool:
            # Submit tasks to the pool
            for i in range(10):
                pool.apply_async(self._worker, args=(process_queue,))

            # Wait for tasks to complete
            pool.close()
            pool.join()

        # Send stop signal to the logger queue
        log_queue = self.logger.handlers[1].queue
        log_queue.put_nowait(None)

    def _worker(self, process_queue):
        # Log a message
        self.logger.info(f"Worker process {multiprocessing.current_process().name} started")

        # Add task to process queue
        process_queue.put(f"Task processed by {multiprocessing.current_process().name}")

        # Log a message
        self.logger.info(f"Worker process {multiprocessing.current_process().name} finished")


if __name__ == "__main__":
    plm = ProcessLoggerManager()
    plm.run()
