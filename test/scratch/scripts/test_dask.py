import typing
import unittest

import numpy as np
import dask.array as da


def test_function(arr):
    return 1


class Test(unittest.TestCase):
    def test1(self):
        darr = da.random.random((10, 100, 3))
        darr.ap


if __name__ == "__main__":
    unittest.main()
