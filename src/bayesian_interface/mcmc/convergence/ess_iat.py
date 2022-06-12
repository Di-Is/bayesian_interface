import typing
import math

import numpy as np

from .convergence import AbsStrategy, ThresholdType


class ESSIATStrategy(AbsStrategy):
    def __init__(self, threshold: typing.Optional[float] = 100000, *args) -> None:
        self._threshold = threshold
        self._external_lengths = args

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
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
    def threshold_type(cls) -> ThresholdType:  # noqa
        return ThresholdType.upper

    def compute(self, array: np.ndarray) -> np.ndarray:
        nchain, nsteps = array.shape[:2]
        if self._external_lengths is None:
            extra = 1
        else:
            extra = math.prod(self._external_lengths)
        iats = array[:, -1]
        return (nchain * nsteps * extra / iats).sum(axis=0)
