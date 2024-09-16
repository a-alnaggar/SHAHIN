import time

from pandas import DataFrame
from pandas.io.parsers.readers import TextFileReader
from typing import TypeVar, Iterable, Callable, Sized, Union
from concurrent.futures import ProcessPoolExecutor

from itertools import islice
from utils.logger import logger

TimeLessObj = TypeVar("TimeLessObj")


class TimeLess:
    """
    Class for manging parallel execution and analyzing processing time. Each task list future is saved in dict
    `self.futures` where the key is `task_name`.
    """

    def __init__(self):
        self.start_list = []
        self.end_list = []
        self.index = 0
        self.futures = {}

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

    def execute_parallel(
        self,
        max_workers: int,
        df: Union[Iterable[DataFrame], Sized],
        task_name: str,
        func_handler: Callable,
        num_chunks: int = 0,
        *args,
    ) -> TimeLessObj:
        """
        Applies a constrained multiprocessing over a dataframe or an iterable, where the number of active tasks at the
        same time cannot exceed `max_workers`. Note that the chunk number must be always passed to the function.

        :param max_workers: Maximum no. of active tasks and cores concurrently.
        :param df: Chunked dataframe reader to apply tasks over.
        :param task_name: Name of the parallelized task.
        :param func_handler: Function to apply.
        :param num_chunks: Number of chunks of passed iterable.
        :param args: Function arguments.
        :return: TimeLessObj
        """
        if type(df) != TextFileReader:
            chunked_data = self.segment_iterable(
                max_workers if not num_chunks else num_chunks, df
            )
            del df
        else:
            chunked_data = df

        # Initialize task futures
        self.futures[task_name] = []

        with ProcessPoolExecutor(max_workers=max_workers) as exc:
            for i, chunk in enumerate(chunked_data, start=1):
                if len(self.futures[task_name]) == max_workers:
                    # Don't feed the executor more than four processes
                    while (
                        sum(1 for future in self.futures[task_name] if future.done())
                        < max_workers
                    ):
                        pass
                self.futures[task_name].append(
                    exc.submit(func_handler, chunk, i, *args)
                )

        return self

    def delete_futures(self, task_name: str) -> TimeLessObj:
        """
        Deletes specified task futures.
        :param task_name: Task futures to delete.
        :return:
        """
        del self.futures[task_name]
        logger.info("Deleted future results of task %s", task_name)
        return self

    @staticmethod
    def segment_iterable(num_chunks: int, iterable: Union[Iterable, Sized]):
        """
        Slices an iterable over the maximum no. of workers.
        :param num_chunks: No. of chunks.
        :param iterable: Iterable object
        :return:
        """
        chunk_size = len(iterable) // num_chunks
        it = iter(iterable)
        chunked_data = []
        while True:
            # Check if required size reached
            chunk = list(islice(it, chunk_size))
            if not chunk:
                break
            chunked_data.append(chunk)
        return chunked_data
