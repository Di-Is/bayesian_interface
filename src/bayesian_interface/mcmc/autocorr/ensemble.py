import os
import typing
from abc import ABCMeta, abstractmethod

import numpy as np
import dask.array as da

from .misc import check_dimension
from .autocorrtime import AbsStrategy
import bayesian_interface.mcmc.autocorr.normal as normal


class StrategyBase(AbsStrategy):
    def __init__(self, strategy: normal.StrategyBase):
        self._strategy = strategy

    def compute(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def method_name(self) -> str:
        raise NotImplementedError


import time


class StrategyFlattenCalc(StrategyBase):
    expected_dim = 3

    def __init__(self, strategy: normal.StrategyBase):
        super().__init__(strategy)
        self._dask = False

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray:
        check_dimension(array, self.expected_dim)

        if self._dask or isinstance(array, da.Array):
            if isinstance(array, da.Array):
                darr = array
                # darr = darr.rechunk((array.shape[0], array.shape[1], 1))
            else:
                chunk = (array.shape[0], array.shape[1], 1)
                darr = da.from_array(array)
        else:
            darr = array

        iats = self._strategy.compute(
            darr.transpose([1, 0, 2]).reshape(-1, array.shape[-1])
        )
        return iats

    @property
    def method_name(self) -> str:
        name = f"Ensemble: {'FlattenCalc'}, Normal: {self._strategy.method_name}"
        return name


class StrategyMeanCalc(StrategyBase):
    expected_dim = 3

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray:
        check_dimension(array, self.expected_dim)

        if isinstance(array, da.Array):
            darr = array
            darr = darr.rechunk((array.shape[0], 1, 1))
        else:
            darr = da.from_array(array, chunks=(array.shape[0], 1, 1))
        iats = self._strategy.compute(darr.mean(axis=1))
        return iats

    @property
    def method_name(self) -> str:
        name = f"Ensemble: {'MeanCalc'}, Normal: {self._strategy.method_name}"
        return name


class StrategyCalcMean(StrategyBase):
    expected_dim = 3

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray:
        check_dimension(array, self.expected_dim)

        if isinstance(array, da.Array):
            darr = array
            darr = darr.rechunk((array.shape[0], 1, 1))
        else:
            darr = da.from_array(array, chunks=(array.shape[0], 1, 1))

        res = darr.map_blocks(
            self._calc,
            drop_axis=0,
            #   chunks=(array.shape[0], 1, 1),
            dtype=array.dtype,
            meta=np.array([]),
        )
        # iats = self._strategy.compute(darr)
        iats = res.mean(axis=0).compute()
        return iats

    def _calc(self, array: np.ndarray):
        if array.ndim == 3:
            result = []
            result.append(self._strategy.compute(array[..., 0, :]))
        else:
            raise ValueError
        return np.asarray(result, dtype=array.dtype)

    @property
    def method_name(self) -> str:
        name = f"Ensemble: {'CalcMean'}, Normal: {self._strategy.method_name}"
        return name


class StrategyAssignment(StrategyBase):
    expected_dim = 3

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray:
        check_dimension(array, self.expected_dim)

        if isinstance(array, da.Array):
            darr = array
            darr = darr.rechunk((array.shape[0], array.shape[1], 1))
        else:
            darr = da.from_array(array, chunks=(array.shape[0], array.shape[1], 1))

        iats = self._strategy.compute(darr)

        return iats

    @property
    def method_name(self) -> str:
        name = f"Ensemble: {'Assignment'}, Normal: {self._strategy.method_name}"
        return name
