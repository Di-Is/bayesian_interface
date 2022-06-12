"""
Pre-processing of MCMC chains for use in the calculation of convergence statistics
"""

import typing
from enum import Enum
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, fields

import numpy as np
import dask.array as da

import bayesian_interface.data_structure as bay_data
import bayesian_interface.mcmc.autocorr as bay_acor
from .misc import check_dimension
from ...logger import Logger

logger = Logger(__name__)


class AbsPreprocess(metaclass=ABCMeta):
    @abstractmethod
    def compute(
        self, array: np.ndarray | da.Array, on_chain: bool, on_dim: bool
    ) -> np.ndarray | da.Array:
        """
        :param array: raw mcmc chain
        :param on_chain: Whether to have a chain dimension
        :param on_dim: Whether to have a dim(param) dimension
        :return: processed array
        """
        pass

    # @property
    # @abstractmethod
    # def need_chain(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def need_dim(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def drop_chain(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def drop_dim(self):
    #     pass


class AutoCorrPreProcess(AbsPreprocess):
    def __init__(self, iat: bay_acor.AutoCorrTime):
        self._iat = iat

    def compute(
        self, array: np.ndarray | da.Array, on_chain: bool, on_dim: bool
    ) -> np.ndarray | da.Array:
        """"""
        res = self._iat.compute(array, on_chain, on_dim)

        res_shape = []
        if on_chain:
            res_shape.append(array.shape[0])
            res_shape.append(array.shape[1])
        else:
            res_shape.append(array.shape[0])
        if on_dim:
            res_shape.append(array.shape[-1])

        if isinstance(array, da.Array):
            result = da.full(res_shape, np.nan)
        else:
            result = np.full(res_shape, np.nan)
        for i, step in enumerate(res.steps):
            result[:, step - 1] = res.iats[:, i]
        return result


class EnsembleCompressor(AbsPreprocess):
    def __init__(self, func: typing.Callable = np.mean):
        self._compress_func = func

    def compute(
        self, array: np.ndarray | da.Array, on_chain: bool, on_dim: bool
    ) -> np.ndarray | da.Array:

        if on_chain and on_dim:
            axes = {0, 1, array.ndim - 1}
        elif on_chain:
            axes = {0, 1}
        elif on_dim:
            axes = {0, array.ndim - 1}
        else:
            axes = {0}

        axes = set(np.arange(array.ndim)) - axes
        axes = tuple(axes)

        res = self._compress_func(array, axis=axes)
        return res

    @property
    def need_chain(self):
        return False

    @property
    def need_dim(self):
        return False

    @property
    def drop_chain(self):
        return False

    @property
    def drop_dim(self):
        return False

    @property
    def need_external(self):
        return True

    @property
    def drop_external(self):
        return True
