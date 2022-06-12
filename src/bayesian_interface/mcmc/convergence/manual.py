import typing

import numpy as np
import dask.array as da
from dask.delayed import Delayed


from .convergence import AbsStrategy, ThresholdType
from .misc import check_dimension


class Manual(AbsStrategy):
    def __init__(
        self,
        threshold: float = 1000.0,
    ) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

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
        return 1

    @classmethod
    def compute(cls, array: np.ndarray) -> np.ndarray:
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
