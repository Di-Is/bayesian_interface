import time
import typing

import numpy as np

from concurrent.futures import ThreadPoolExecutor


class ThreadPool(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def map(self, fn, *iterables, timeout=None, chunksize: int = 1):
        if timeout is not None:
            end_time = timeout + time.monotonic()

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

        result_lst = list(result_iterator())
        result = np.empty(nwalkers, dtype=np.float64)
        result = self._set_result(result_lst, result, nwalkers)
        return result
