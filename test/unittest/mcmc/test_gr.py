import os
import unittest

import numpy as np

from bayesian_interface.mcmc.convergence.convergence import Convergence
import bayesian_interface.mcmc.convergence.gr_rank as gr_rank


class TestDask(unittest.TestCase):
    def setUp(self) -> None:
        self.fpath = "test.h5"
        self.gpath = "test"
        self.name = "array"

    def tearDown(self) -> None:
        if os.path.exists(self.fpath):
            os.remove(self.fpath)

    def test_dask_memory_usage(self):

        import h5py
        import psutil
        from sys import getsizeof
        import dask.array as da
        from arviz.stats.diagnostics import _rhat_rank

        proc = psutil.Process()

        def rhat_rank(arr):

            if arr.ndim == 3:
                result = []
                for i in range(arr.shape[-1]):
                    result.append(_rhat_rank(arr[..., i]))
            elif arr.ndim == 2:
                result = _rhat_rank(arr)
            result = np.asarray(result)

            return result

        shape = (100, 200000, 10)
        fact = 2**20
        print("Memory size 1", proc.memory_info().rss / fact)
        with h5py.File(self.fpath, "w") as f:
            f.create_group(self.gpath)
            d = f[self.gpath].create_dataset(
                self.name, shape=shape, maxshape=shape, dtype=np.float32
            )
            print("Memory size 2", proc.memory_info().rss / fact)
            d[:] = np.random.rand(*shape).astype(dtype=np.float32)
            print("Memory size 3", proc.memory_info().rss / fact)

            chunks = (shape[0], shape[1], 1)
            darr = da.from_array(d, chunks=chunks)
            print("Memory size 4", proc.memory_info().rss / fact)

            # print(darr.chunks)
            # res = da.apply_over_axes(
            #     a=darr,
            #     axes=2,
            #     func=_rhat_rank,
            # )
            # print(res.compute())

            res = darr.map_blocks(
                rhat_rank,
                drop_axis=[0, 1],
                # new_axis=1,
                dtype=np.float32,
                meta=np.array([]),
            )
            print("Memory size 5", proc.memory_info().rss / fact)
            print(res.compute())
            print("Memory size 6", proc.memory_info().rss / fact)
            del res
            print("Memory size 7", proc.memory_info().rss / fact)

        # print(res.compute())
        #   print(res.compute())

        print("Memory size", proc.memory_info().rss / fact)
        print("Memory size", proc.memory_info().rss / fact)


class TestConvergence(unittest.TestCase):
    def test1(self):

        strategy = gr_rank.Strategy
        conv_check = Convergence(strategy)

        arr = np.random.rand(10, 10000, 3)
        vals = conv_check.check_convergence(arr)
        print(vals.convergence.get())


if __name__ == "__main__":
    unittest.main()
