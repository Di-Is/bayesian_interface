import typing
from abc import ABCMeta, abstractmethod

import numpy as np
import dask.array as da

from bayesian_interface.data_structure.autocorr import (
    AutoCorrResultFactory,
    AutoCorrResult,
)


class AbsStrategy(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, array: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        pass


class AutoCorrTime:
    def __init__(
        self, strategy: AbsStrategy, data: typing.Optional[AutoCorrResult] = None
    ):
        self._strategy = strategy
        if data is None:
            data = AutoCorrResultFactory.create()
        self._data = data
        self._dask = False

    def check_initialize(self) -> bool:
        names = ("iats", "steps", "method_name")
        flag = all([getattr(self._data, name).has() for name in names])
        return flag

    def compute(self, array: np.ndarray) -> AutoCorrResult:

        if not self.check_initialize():
            ndim = array.shape[-1]
            self._data.iats.create(
                shape=(0, ndim), maxshape=(None, ndim), dtype=array.dtype
            )
            self._data.steps.create(shape=(0,), maxshape=(None,), dtype=int)
            self._data.method_name.set(self._strategy.method_name)

        chunk = tuple(
            array.shape[i] if i != array.ndim - 1 else None for i in range(array.ndim)
        )
        if self._dask or isinstance(array, da.Array):
            darr = da.from_array(array, chunks=chunk)
        else:
            darr = array

        if array.ndim == 3:
            nsteps = array.shape[0]
            ndim = array.shape[2]
        else:
            nsteps = array.shape[0]
            ndim = array.shape[1]
        iats = self._strategy.compute(darr)

        arr = np.full((len(array) - self._data.iats.shape[0], array.shape[-1]), np.nan)
        arr[-1] = iats
        idx = slice(self._data.iats.shape[0], None)
        self._data.iats.resize((nsteps, ndim))
        self._data.iats.set(arr, idx=idx)
        self._data.steps.append(np.asarray([len(array)]), axis=0)
        return self._data
