import typing
import math

import numpy as np

from .convergence import AbsStrategy, MagnitudeRelation


class ESSFromIAT(AbsStrategy):
    def __init__(
        self, threshold: typing.Optional[float] = 100000, **external_lengths
    ) -> None:
        super().__init__(threshold)
        self._external_lengths = external_lengths

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
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.upper

    def compute(self, array: np.ndarray) -> float:
        nchain, nsteps = array.shape[:2]
        if len(self._external_lengths) == 0:
            extra = 1
        else:
            extra = math.prod(self._external_lengths.values())
        iats = array[:, -1]
        return (nchain * nsteps * extra / iats).sum(axis=0)
