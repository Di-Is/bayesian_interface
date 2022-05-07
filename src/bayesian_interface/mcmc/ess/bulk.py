import typing

import numpy as np
import dask.array as da

from .ess import AbsFactory
from .misc import check_dimension


class Bulk(AbsFactory):
    expected_dim = 2

    @property
    def method_name(self) -> str:
        return "bulk"

    def compute(self, array: np.ndarray) -> np.ndarray:
        check_dimension(array, self.expected_dim)

        chunks = []

        if isinstance(array, da.Array):
            darr = array
        else:
            darr = da.from_array(array, chunks=chunks)

        res = darr.map_blocks(
            self._calc, drop_axis=[0], dtype=array.dtype, meta=np.array([])
        )
        return res.compute()

    @staticmethod
    def _calc(array: np.ndarray) -> np.ndarray:
        from arviz.stats.diagnostics import _ess_bulk

        if array.ndim == 2:
            result = []
            for i in range(array.shape[-1]):
                result.append(_ess_bulk(array[..., i]))
        elif array.ndim == 1:
            result = _ess_bulk(array)
        else:
            raise ValueError
        return np.asarray(result)
