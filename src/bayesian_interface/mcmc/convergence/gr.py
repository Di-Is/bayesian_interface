"""This module provides a class to compute convergence criterion(gelman rubin rhat).
"""
import numpy as np
import dask.array as da
from dask.delayed import Delayed

from .convergence import AbsStrategy, MagnitudeRelation
from .misc import check_dimension


class GR(AbsStrategy):
    """Class to compute convergence criterion of MCMC chains"""

    def __init__(self, threshold: float = 1.01) -> None:
        super().__init__(threshold)

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "gr"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.lower

    @classmethod
    @property
    def expected_dim(cls) -> int | tuple[int, ...]:  # noqa
        return 2

    def compute(self, array: np.ndarray | da.Array) -> float:
        """Compute gelman rubin rhat.
        :param array: input array
        :return: criterion value
        Ref. Gelman & Rubin, 1992
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
