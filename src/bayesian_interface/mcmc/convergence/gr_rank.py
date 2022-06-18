"""This module provides a class to compute convergence criterion(rank normalized gelman rubin rhat).
"""

import numpy as np
import dask.array as da
from dask.delayed import Delayed

from .convergence import AbsStrategy, MagnitudeRelation


class GRRank(AbsStrategy):
    """Class to compute convergence criterion of MCMC chains"""

    def __init__(self, threshold: float = 1.01) -> None:
        super().__init__(threshold)

    @property
    def algorithm_name(self) -> str:
        return "gr_rank"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.lower

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 2

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

    def compute(self, array: np.ndarray) -> float:
        """Compute rank normalized gelman rubin rhat.
        :param array: input array
        :return: criterion value
        Ref. Vehtari et al. (2019)
        """
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
        from arviz.stats.diagnostics import _rhat_rank
        from arviz.utils import Numba

        Numba.enable_numba()
        result = _rhat_rank(array)
        return result
