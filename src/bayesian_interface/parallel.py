import time
import typing
import types

import numpy as np

from concurrent.futures import ThreadPoolExecutor


class ThreadPool(ThreadPoolExecutor):
    def __init__(self, vectorize: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorize = vectorize

    def _split_data(self, arr: np.ndarray):
        nwalkers, ndim = arr.shape
        num_split = self._max_workers
        nwalkers_sp = int(nwalkers / num_split)
        sp_lst = []
        for i in range(num_split):
            if i < num_split - 1:
                idx = slice(i * nwalkers_sp, (i + 1) * nwalkers_sp)
            else:
                idx = slice(i * nwalkers_sp, nwalkers)
            sp_lst.append(arr[idx])
        return (tuple(sp_lst),)

    def _set_result(self, arr, result, nwalkers):
        num_split = self._max_workers
        nwalkers_sp = int(nwalkers / num_split)
        for i in range(num_split):
            if i < num_split - 1:
                idx = slice(i * nwalkers_sp, (i + 1) * nwalkers_sp)
            else:
                idx = slice(i * nwalkers_sp, nwalkers)
            result[idx] = arr[i]
        return result

    def map_normal(self, fn, *iterables, timeout=None):
        pass

    def map_vectorize(self, fn, *iterables, timeout=None):
        pass

    def map(self, fn, *iterables, timeout=None):
        if timeout is not None:
            end_time = timeout + time.monotonic()

        if self.vectorize:
            if isinstance(iterables[0], types.GeneratorType):
                iterables = np.array([list(i) for i in iterables])
            nwalkers = len(iterables[0])
            iterables = self._split_data(iterables[0])
        fs = [self.submit(fn, *args) for args in zip(*iterables)]

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                # reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    if timeout is None:
                        yield fs.pop().result()
                    else:
                        yield fs.pop().result(end_time - time.monotonic())
            finally:
                for future in fs:
                    future.cancel()

        if self.vectorize:
            result_lst = list(result_iterator())
            result = np.empty(nwalkers, dtype=np.float64)
            result = self._set_result(result_lst, result, nwalkers)
        else:
            result = list(result_iterator())
        return result


class PoolWrapper:
    """the pool wrapper class for the nest multi-process,thread"""

    def __init__(self, pool_cls: typing.Callable, *args, **kwargs):
        """Constractor
        :param pool_cls: the pool class (not initialized)
        :param args: the pool class __init__ arguments
        :param kwargs: the pool class __init__ kwargs
        """
        self._args = args
        self._kwargs = kwargs
        self._pool_cls = pool_cls
        self._pool = None

    @property
    def map(self):
        """map function
        :return: map function
        """
        if self._pool is None:
            self._pool = self._init_pool()
        return self._pool.map

    @property
    def submit(self):
        """submit method
        :return: submit method
        """
        if self._pool is None:
            self._pool = self._init_pool()
        return self._pool.submit

    def _init_pool(self):
        """initializing pool class
        :return: initialized pool class
        """
        return self._pool_cls(*self._args, **self._kwargs)

    def __getstate__(self):
        del self._pool
        self._pool = None
        return self.__dict__
