import typing

import numpy as np
import dask.array as da

from .convergence import AbsStrategy, MagnitudeRelation

# TODO: Add docs


class MaxArchangeStrategy(AbsStrategy):
    def __init__(self, threshold: float = 0.03) -> None:
        super().__init__(threshold)

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
    def expected_dim(self) -> int | tuple[int, ...]:
        return 1

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "maxArchange"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.lower

    def compute(self, array: np.ndarray | da.Array) -> float:
        """Compute max_archange value
        :param array: iat_array
        :return:
        """
        indices = np.where(~np.isnan(array))[0]

        if isinstance(indices, da.Array):
            indices = indices.compute()

        if len(indices) < 2:
            return np.nan

        # dtau / (dstep / tau)
        diat = array[indices[-1]] - array[indices[-2]]
        dstep = indices[-1] - indices[-2]
        cri = diat / (dstep / array[indices[-1]])
        return np.abs(cri)
