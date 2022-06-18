"""This module provides a class to compute convergence criterion(bulk ess).
"""
import math

import numpy as np
import dask.array as da
from dask.delayed import Delayed

from .convergence import AbsStrategy, MagnitudeRelation


class ESSBulk(AbsStrategy):
    """Class to compute convergence criterion of MCMC chains"""

    def __init__(self, threshold: float = 100000.0, **external_lengths) -> None:
        super().__init__(threshold)
        self._external_lengths = external_lengths

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        """expected input array dim is (chain, nsteps)
        :return:
        """
        return 2

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "ess_bulk"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.upper

    def compute(self, array: np.ndarray) -> float:
        """Compute bulk effective sample size.
        :param array: input array
        :return: criterion value
        Ref. Vehtari (2021)
        """
        from arviz.stats.diagnostics import _ess_bulk

        if self._external_lengths is None:
            extra = 1
        else:
            extra = math.prod(self._external_lengths.values())

        match array:
            case np.ndarray():
                result = _ess_bulk(array)
            case da.Array() | Delayed():
                result = _ess_bulk(array)
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")

        return result.sum() * extra

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False

    @property
    def need_chain(self) -> bool:
        return True

    @property
    def drop_chain(self) -> bool:
        return True
