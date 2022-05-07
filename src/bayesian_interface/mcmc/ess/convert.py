import math
import typing

import numpy as np
import dask.array as da

from .ess import AbsFactory


class Convert(AbsFactory):
    def __init__(self, *extra_dim_length):
        if len(extra_dim_length) == 0:
            extra_dim_length = (1,)
        self._extra_dim_length = extra_dim_length

    @property
    def method_name(self) -> str:
        return "iat based"

    @property
    def expected_dim(self) -> int:
        return 2

    def compute(self, array: np.ndarray) -> np.ndarray:

        if isinstance(array, da.Array):
            darr = array
        else:
            darr = da.from_array(array)

        # res = darr.map_blocks(
        #     self._calc, drop_axis=[0], dtype=array.dtype, meta=np.array([])
        # )

        n = np.arange(len(array)) * math.prod(self._extra_dim_length)
        n = np.atleast_2d(n).T
        res = n / darr
        return res.compute()

    # @staticmethod
    # def _calc(array: np.ndarray) -> np.ndarray:
    #     from arviz.stats.diagnostics import _ess_bulk
    #
    #     if array.ndim == 2:
    #         result = []
    #         for i in range(array.shape[-1]):
    #             result.append(_ess_bulk(array[..., i]))
    #     elif array.ndim == 1:
    #         result = _ess_bulk(array)
    #     else:
    #         raise ValueError
    #     return np.asarray(result)
