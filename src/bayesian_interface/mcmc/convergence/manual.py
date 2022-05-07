import typing

import numpy as np

from .convergence import AbsStrategy, ThresholdType
from .misc import check_dimension


class Strategy(AbsStrategy):
    @classmethod
    @property
    def threshold_default(cls) -> float:  # noqa
        return 1000.0

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "manual length burn-in"

    @classmethod
    @property
    def threshold_type(cls) -> ThresholdType:  # noqa
        return ThresholdType.upper_eq

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 2, 3

    @classmethod
    def compute(cls, array: np.ndarray) -> np.ndarray:

        nsteps = array[0]
        ndim = array[-1]
        return np.asarray([nsteps for _ in range(ndim)], dtype=float)
