import typing

import numpy as np

from .convergence import AbsStrategy, ThresholdType
from .misc import check_dimension


class Strategy(AbsStrategy):
    @classmethod
    @property
    def threshold_default(cls) -> float:  # noqa
        return 1.01

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "traditional Gelman-Rubin indicator"

    @classmethod
    @property
    def threshold_type(cls) -> ThresholdType:  # noqa
        return ThresholdType.lower

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 3

    @classmethod
    def compute(cls, array: np.ndarray) -> np.ndarray:

        import dask.array as da

        darr = da.from_array(array, chunks=(array.shape[0], array.shape[1], 1))
        res = darr.map_blocks(
            cls._calc_gr, drop_axis=[0, 1], dtype=array.dtype, meta=np.array([])
        )
        return res.compute()

    @staticmethod
    def _calc_gr(array: np.ndarray) -> np.ndarray:
        from arviz.stats.diagnostics import _rhat_identity
        from arviz.utils import Numba

        Numba.enable_numba()

        if array.ndim == 3:
            result = []
            for i in range(array.shape[-1]):
                result.append(_rhat_identity(array[..., i]))
        elif array.ndim == 2:
            result = _rhat_identity(array)
        else:
            raise ValueError

        return np.asarray(result)
