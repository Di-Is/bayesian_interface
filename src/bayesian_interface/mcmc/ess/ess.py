import typing
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, fields
import math

import numpy as np

import bayesian_interface.data_structure as bay_data
from .misc import check_dimension


@dataclass
class ESSResult:

    method_name: bay_data.AbsAttr
    steps: bay_data.AbsArray
    esss: bay_data.AbsArray


class ESSResultFactory(bay_data.AbsDataFactory):
    @classmethod
    @property
    def dataclass(cls) -> typing.Type[bay_data.T]:  # noqa
        return ESSResult

    @classmethod
    @property
    def merge_param(cls) -> bay_data.AbsDefaultParam.__class__:  # noqa
        return ESSResultParamFactory


class AbsFactory(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, array: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        pass

    @property
    @abstractmethod
    def expected_dim(self) -> int:
        pass


class EssCalc:
    def __init__(
        self, strategy: AbsFactory, data: typing.Optional[ESSResult] = None
    ) -> None:
        self._strategy = strategy
        if data is None:
            data = ESSResultFactory.create()
        self._data = data

    def check_initialized(self):
        names = [
            "esss",
            "method_name",
            "steps",
        ]
        flag = all([getattr(self._data, name).has() for name in names])
        return flag

    def compute(self, array: np.ndarray) -> ESSResult:
        check_dimension(array, self._strategy.expected_dim)

        if not self.check_initialized():
            ndim = array.shape[-1]
            nchain = len(array)
            self._data.esss.create(
                shape=(nchain, 0, ndim), maxshape=(None, 0, ndim), dtype=array.dtype
            )
            self._data.steps.create(shape=(0,), maxshape=(0,), dtype=int)
            self._data.method_name.set(self._strategy.method_name)

        esss = self._strategy.compute(array)

        self._data.steps.append(np.asarray([len(array)]), axis=0)
        self._data.esss.resize(array.shape)
        self._data.esss.set(esss)

        return self._data


class ConvertIAT2ESS:
    expected_dim = 2

    def __init__(self, data: typing.Optional[ESSResult] = None) -> None:
        self._data = data
        if data is None:
            self._data = ESSResultFactory.create()

    def check_initialized(self):
        names = [
            "esss",
            "method_name",
            "steps",
        ]
        flag = all([getattr(self._data, name).has() for name in names])
        return flag

    @property
    def method_name(self) -> str:
        return "iat based calculation"

    def compute(self, iats: np.ndarray, nsteps: int, *external_dim_lens) -> ESSResult:

        if not self.check_initialized():
            ndim = iats.shape[-1]
            self._data.esss.merge_param(
                shape=(0, ndim), maxshape=(0, ndim), dtype=iats.dtype
            )
            self._data.steps.merge_param(shape=(0,), maxshape=(0,), dtype=int)
            self._data.method_name.set(self.method_name)

        if len(external_dim_lens) == 0:
            external_dim_lens = [1]

        length = nsteps * math.prod(external_dim_lens)
        esss = length / iats

        self._data.steps.append(np.asarray(nsteps), axis=0)
        # self._data.esss.append(esss, axis=0)
        self._data.esss.resize(iats.shape)
        self._data.esss.set(esss)
        return self._data


class ESSResultParamFactory(bay_data.AbsDefaultParam):
    attrs = tuple(var.name for var in fields(ESSResult) if var.type is bay_data.AbsAttr)
    arrays = tuple(
        var.name for var in fields(ESSResult) if var.type is bay_data.AbsArray
    )

    @classmethod
    def get_memory(cls):
        return {name: {} for name in cls.attrs + cls.arrays}

    @classmethod
    def get_hdf5(cls):
        attr_cfg = {"gpath": "ess"}
        array_cfg = attr_cfg | {"compression": "gzip", "compression_opts": 9}
        res = {name: attr_cfg for name in cls.attrs}
        res |= {name: array_cfg for name in cls.arrays}
        return res

    @classmethod
    def get_netcdf4(cls):
        attr_cfg = {"gpath": "ess"}
        array_cfg = attr_cfg | {"compression": "gzip", "compression_opts": 9}
        res = {name: attr_cfg for name in cls.attrs}
        res |= {name: array_cfg for name in cls.arrays}
        return res
