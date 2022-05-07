import typing
from enum import Enum
from abc import abstractmethod, ABCMeta
import operator as op

import numpy as np

from bayesian_interface.data_structure.convergence import (
    ConvergenceResult,
    ConvergenceResultFactory,
)
from .misc import check_dimension


class ThresholdType(Enum):
    equal = 0
    lower = 1
    lower_eq = 2
    upper = 3
    upper_eq = 4


def check_relationship(threshold_type: ThresholdType, a, b) -> bool:
    match threshold_type:
        case ThresholdType.equal:
            return op.eq(a, b)
        case ThresholdType.lower:
            return op.lt(a, b)
        case ThresholdType.lower_eq:
            return op.le(a, b)
        case ThresholdType.upper:
            return op.gt(a, b)
        case ThresholdType.upper_eq:
            return op.ge(a, b)


class AbsStrategy(metaclass=ABCMeta):
    @property
    @abstractmethod
    def expected_dim(self) -> int | tuple[int, ...]:
        pass

    @classmethod
    @abstractmethod
    def compute(cls, array: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    @property
    @abstractmethod
    def threshold_type(cls) -> ThresholdType:  # noqa
        pass

    @classmethod
    @property
    @abstractmethod
    def algorithm_name(cls) -> str:  # noqa
        pass

    @classmethod
    @property
    @abstractmethod
    def threshold_default(cls) -> float:  # noqa
        pass


class Convergence:
    def __init__(
        self,
        strategy: AbsStrategy | AbsStrategy.__class__,
        data: typing.Optional[ConvergenceResult] = None,
    ):
        self._strategy = strategy

        if data is None:
            data = ConvergenceResultFactory.create()
        self._data = data

    def _init_data(
        self,
        data: ConvergenceResult,
        ndim: int,
    ):
        # attr
        data.criterion_method.set(self._strategy.algorithm_name)
        data.threshold_type.set(self._strategy.threshold_type)
        # array
        data.criterion_values.create((0, ndim), maxshape=(None, ndim), dtype=np.float32)
        data.convergences.create((ndim,), maxshape=(ndim,), dtype=np.float32)
        data.convergence_steps.create((ndim,), maxshape=(ndim,), dtype=np.int32)
        data.steps.create((0,), maxshape=(None,), dtype=np.int32)
        return data

    def check_initialized(self):
        names = [
            "criterion_method",
            "threshold_type",
            "criterion_values",
            "steps",
            "convergences",
            "convergence",
        ]
        flag = all([getattr(self._data, name).has() for name in names])
        return flag

    def check_convergence(
        self, array: np.ndarray, threshold: typing.Optional[float] = None
    ) -> ConvergenceResult:

        check_dimension(array, self._strategy.expected_dim)

        if not self.check_initialized():
            self._data = self._init_data(self._data, array.shape[-1])

        if threshold is None:
            threshold = self._strategy.threshold_default

        criterion_values = self._strategy.compute(array)
        convergences = [
            check_relationship(self._strategy.threshold_type, cri, threshold)
            for cri in criterion_values
        ]

        dim_axis = array.ndim - 1
        step_axis = dim_axis - 1

        self._data.threshold.set(threshold)
        self._data.threshold_type.set(self._strategy.threshold_type)
        # self._data.criterion_values.append(np.atleast_2d(criterion_values), axis=0)
        self._data.criterion_values.resize(array.shape[step_axis : dim_axis + 1])
        self._data.criterion_values.set(np.atleast_2d(criterion_values), idx=-1)
        self._data.steps.append(np.array(array.shape[1]), axis=0)
        self._data.convergences.set(convergences)
        self._data.convergence.set(all(convergences))
        return self._data
