"""This module provides a class to compute convergence criterion(ess calculated iat).
"""
import math
import typing

import numpy as np
import dask.array as da

from .convergence import AbsStrategy, MagnitudeRelation


class ESSFromIAT(AbsStrategy):
    """Class to compute convergence criterion of MCMC chains"""

    def __init__(self, threshold: float = 100000, **external_lengths) -> None:
        super().__init__(threshold)
        self._external_lengths = external_lengths

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        """expected input array dim is (chain, nsteps)
        :return:
        """
        return 2

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

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "ess_from_iat"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.upper

    def compute(self, array: np.ndarray | da.Array) -> float:
        """Compute effective sample size from integrated auto correlation time.
        :param array: input array
        :return: criterion value
        """
        nchain, nsteps = array.shape[:2]
        if len(self._external_lengths) == 0:
            extra = 1
        else:
            extra = math.prod(self._external_lengths.values())
        iats = array[:, -1]
        return (nchain * nsteps * extra / iats).sum(axis=0)
