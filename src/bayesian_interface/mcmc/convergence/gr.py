"""Module to calculate convergence criterion for GelmanRubin
"""
import typing

import numpy as np
import dask.array as da
from dask.delayed import Delayed

from .convergence import AbsStrategy, MagnitudeRelation
from .misc import check_dimension


class GR(AbsStrategy):
    """Class to calculate convergence criterion for GelmanRubin"""

    def __init__(self, threshold: float = 1.01) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """convergence threshold
        :return: convergence threshold
        """
        return self._threshold

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "GR"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.lower

    @classmethod
    @property
    def expected_dim(cls) -> int | tuple[int, ...]:  # noqa
        return 2

    def compute(self, array: np.ndarray | da.Array) -> float:
        """calculate convergence criterion
        :param array: mcmc chain
        :return: criterion value
        """
        check_dimension(array, self.expected_dim)

        match array:
            case np.ndarray():
                result = self._calc_criterion(array)
            case da.Array() | Delayed():
                result = self._calc_criterion(array)
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @staticmethod
    def _calc_criterion(array: np.ndarray) -> float:
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
