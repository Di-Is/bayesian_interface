import os
import typing
from abc import ABCMeta, abstractmethod

import numpy as np
import dask.array as da
from dask.delayed import delayed, Delayed

from .misc import check_dimension
from .autocorrtime import AbsStrategy
import bayesian_interface.mcmc.autocorr.normal as normal


class StrategyBase(AbsStrategy):
    def __init__(self, strategy: normal.StrategyBase):
        self._strategy = strategy

    def compute(self, array: np.ndarray) -> np.ndarray | da.Array:
        raise NotImplementedError

    @property
    def method_name(self) -> str:
        raise NotImplementedError

    @property
    def need_chain(self) -> bool:
        raise NotImplementedError

    @property
    def need_dim(self) -> bool:
        raise NotImplementedError

    @property
    def drop_chain(self) -> bool:
        raise NotImplementedError

    @property
    def drop_dim(self) -> bool:
        raise NotImplementedError


class FlattenCalcStrategy(StrategyBase):
    expected_dim = 2

    def __init__(self, strategy: normal.StrategyBase):
        super().__init__(strategy)

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray:
        check_dimension(array, self.expected_dim)
        match array:
            case np.ndarray() | da.Array() | Delayed():
                result = self._strategy.compute(array.transpose([1, 0]).reshape(-1))
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @classmethod
    @property
    def method_name(cls) -> str:  # noqa
        name = f"Ensemble: {'FlattenCalc'}"
        return name

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False


class MeanCalcStrategy(StrategyBase):
    expected_dim = 2

    def __init__(self, strategy: normal.StrategyBase):
        super().__init__(strategy)

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray | da.Array:
        check_dimension(array, self.expected_dim)
        match array:
            case np.ndarray() | da.Array() | Delayed():
                result = self._strategy.compute(array.mean(axis=1))
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @classmethod
    @property
    def method_name(cls) -> str:  # noqa
        return f"Ensemble: {'MeanCalc'}"

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False


class CalcMeanStrategy(StrategyBase):
    expected_dim = 2

    def __init__(self, strategy: normal.StrategyBase):
        super().__init__(strategy)

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray | da.Array:

        match array:
            case np.ndarray():
                result = np.apply_along_axis(self._strategy.compute, 0, array).mean()
            case da.Array() | Delayed():
                result = da.apply_along_axis(self._strategy.compute, 0, array).mean()
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")

        return result

    @classmethod
    @property
    def method_name(cls) -> str:  # noqa
        name = f"Ensemble: {'CalcMean'}"
        return name

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False


class AssignmentStrategy(StrategyBase):
    expected_dim = 2

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray | da.Array:
        check_dimension(array, self.expected_dim)
        result = self._strategy.compute(array)
        return result

    @classmethod
    @property
    def method_name(cls) -> str:  # noqa
        name = f"Ensemble: {'Assignment'}"
        return name

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False


ENSEMBLE_IAT_METHODS = (
    FlattenCalcStrategy.method_name,
    MeanCalcStrategy.method_name,
    CalcMeanStrategy.method_name,
    AssignmentStrategy.method_name,
)
