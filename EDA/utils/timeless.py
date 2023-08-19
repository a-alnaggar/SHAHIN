import time


class TimeLess:
    """
    Class for analyzing processing time.
    """

    def __int__(self):
        self.start_list = []
        self.end_list = []
        self.index = 0

    def start(self):
        if self.start_list:
            self.index += 1
        self.start_list.append(time.time())

    def end(self):
        self.end_list.append(time.time())
        return self.start_list[self.index] - self.end_list[self.index]
