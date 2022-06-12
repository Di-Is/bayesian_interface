import typing

import numpy as np
import dask.array as da
from dask.delayed import Delayed

from .convergence import AbsStrategy, ThresholdType
from .misc import check_dimension


class GR(AbsStrategy):
    def __init__(self, threshold: float = 1.01):
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "GR"

    @classmethod
    @property
    def threshold_type(cls) -> ThresholdType:  # noqa
        return ThresholdType.lower

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 3

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray | da.Array:

        match array:
            case np.ndarray():
                result = self._calc_criterion(array)
            case da.Array() | Delayed():
                result = self._calc_criterion(array)
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @staticmethod
    def _calc_criterion(array: np.ndarray) -> np.ndarray:
        from arviz.stats.diagnostics import _rhat_identity
        from arviz.utils import Numba

        Numba.enable_numba()
        result = _rhat_identity(array)
        return result

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def need_chain(self) -> bool:
        return True

    @property
    def drop_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return True
