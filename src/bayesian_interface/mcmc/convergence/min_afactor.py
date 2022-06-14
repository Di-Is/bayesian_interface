import typing
import math

import numpy as np
import dask.array as da

from .convergence import AbsStrategy, MagnitudeRelation


class MinAfactorStrategy(AbsStrategy):
    def __init__(
        self, threshold: typing.Optional[float] = 50, **external_lengths
    ) -> None:
        super().__init__(threshold)
        self._external_lengths = external_lengths

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def expected_dim(self) -> int:
        return 1

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "min_afactor"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.upper

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray | da.Array:
        nsteps = len(array)
        if len(self._external_lengths) == 0:
            extra = 1
        else:
            extra = math.prod(self._external_lengths.values())
        return nsteps * extra / array[-1]
