"""This module provides a class to compute convergence criterion(fixed length convergence).
"""

import numpy as np
import dask.array as da
from dask.delayed import Delayed


from .convergence import AbsStrategy, MagnitudeRelation
from .misc import check_dimension


class Manual(AbsStrategy):
    """Class to compute convergence criterion of MCMC chains"""

    def __init__(self, threshold: float = 1000.0) -> None:
        super().__init__(threshold)

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "manual"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.upper_eq

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 1

    @classmethod
    def compute(cls, array: np.ndarray | da.Array) -> float:
        """Compute array nstep
        :param array: input array
        :return: criterion value
        """
        match array:
            case np.ndarray():
                result = len(array)
            case da.Array() | Delayed():
                result = da.asarray(len(array))
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False
