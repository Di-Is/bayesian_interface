import typing

import numpy as np
import dask.array as da
from dask.delayed import Delayed, delayed

from .convergence import AbsStrategy, MagnitudeRelation
from ._gr_impl import calc_stable_psrf


class StableGR(AbsStrategy):
    def __init__(self, threshold: float = 1.01) -> None:
        super().__init__(threshold)

    @property
    def algorithm_name(cls) -> str:  # noqa
        return "stable_gr"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.lower

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 2

    def compute(self, array: np.ndarray) -> float:

        match array:
            case np.ndarray():
                result = self._calc_criterion(array)
            case da.Array() | Delayed():
                result = delayed(self._calc_criterion)(array)
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @staticmethod
    def _calc_criterion(array: np.ndarray) -> float:
        val, _, _ = calc_stable_psrf(np.atleast_3d(array))
        return val[..., 0]

    @property
    def need_chain(self) -> bool:
        return True

    @property
    def drop_chain(self) -> bool:
        return True

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False
