import time

from .logger import logger


class TimeLess:
    """
    Class for analyzing processing time.
    """

    def __init__(self):
        self.start_list = []
        self.end_list = []
        self.index = 0

    def start(self):
        """Records starting time"""
        logger.info("Started recording")
        if self.start_list:
            self.index += 1
        self.start_list.append(time.time())

    def end(self):
        """Ends recording time"""
        self.end_list.append(time.time())
        total_time = self.end_list[self.index] - self.start_list[self.index]
        logger.info("Finished recording. Experiment time %s seconds", total_time)
        return total_time
