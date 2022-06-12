import typing
from abc import ABCMeta, abstractmethod

import numpy as np
import dask.array as da
from dask.delayed import Delayed

import bayesian_interface.data_structure as bay_data
from bayesian_interface.logger import Logger

logger = Logger(__name__)

# TODO Add docs
# TODO Refactoring


class AutoCorrResult(bay_data.AbsData):

    method_name: bay_data.AbsAttr
    steps: bay_data.AbsArray
    iats: bay_data.AbsArray

    @classmethod
    def memory_dflt_par(cls):
        return {name: {} for name in cls.get_attr_names() + cls.get_array_names()}

    @classmethod
    def hdf5_dflt_par(cls):
        attr_cfg = {"gpath": "mcmc"}
        array_cfg = attr_cfg | {"compression": "gzip", "compression_opts": 9}
        res = {name: attr_cfg for name in cls.get_attr_names()}
        res |= {name: array_cfg for name in cls.get_array_names()}
        return res

    @classmethod
    def netcdf4_dflt_par(cls):
        attr_cfg = {"gpath": "mcmc"}
        array_cfg = attr_cfg | {"compression": "gzip", "compression_opts": 9}
        res = {name: attr_cfg for name in cls.get_attr_names()}
        res |= {name: array_cfg for name in cls.get_array_names()}
        return res


class AbsStrategy(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, array: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        pass

    @property
    @abstractmethod
    def need_chain(self) -> bool:
        pass

    @property
    @abstractmethod
    def need_dim(self) -> bool:
        pass

    @property
    @abstractmethod
    def drop_chain(self) -> bool:
        pass

    @property
    @abstractmethod
    def drop_dim(self) -> bool:
        pass


class AutoCorrTime:
    def __init__(
        self,
        strategy: AbsStrategy,
        data: typing.Optional[AutoCorrResult] = None,
        dask: bool = True,
    ):
        """Calculate
        :param strategy:
        :param data:
        :param dask:
        """
        self._strategy = strategy
        if data is None:
            data = AutoCorrResult()
        self.data = data
        self.dask = dask

    def check_initialize(self) -> bool:
        names = ("iats", "steps", "method_name")
        flag = all([getattr(self.data, name).has() for name in names])
        return flag

    def init_data(self):
        pass

    def compute(
        self,
        array: np.ndarray | da.Array,
        on_chain: bool = False,
        on_dim: bool = False,
    ) -> AutoCorrResult:
        """
        :param array:
        :param step_axis:
        :param dim_axis:
        :return:
        """
        logger.info("Starting calculate auto-correlation time.")

        if on_chain:
            nchain, nstep = array.shape[0:2]
            step_axis = 1
        else:
            nchain, nstep = 0, array.shape[0]
            step_axis = 0

        if on_dim:
            ndim = array.shape[-1]
        else:
            ndim = 0

        if not on_chain and (self._strategy.need_chain, self._strategy.drop_chain):
            raise ValueError("chain dimension is not exist.")

        if not on_dim and (self._strategy.need_dim, self._strategy.drop_dim):
            raise ValueError("dim dimension is not exist.")

        if not self.check_initialize():
            shape = []
            maxshape = []
            if on_chain:
                shape.append(nchain)
                maxshape.append(nchain)
            shape.append(0)
            maxshape.append(None)
            if on_dim:
                shape.append(ndim)
                maxshape.append(ndim)

            self.data.iats.create(
                shape=tuple(shape), maxshape=tuple(maxshape), dtype=array.dtype
            )
            self.data.steps.create(shape=(0,), maxshape=(None,), dtype=int)
            self.data.method_name.set(self._strategy.method_name)

        # convert to dask array
        if self.dask and not isinstance(array, da.Array):
            chunks = bay_data.make_chunks(array.shape, on_chain, on_dim)
            array = da.from_array(array, chunks=chunks)

        res_shape = []
        if on_chain:
            res_shape.append(nchain)
        if on_dim:
            res_shape.append(ndim)

        # chain方向をsliceしてstrategyに投げるかどうか
        chain_slc = (
            on_chain and not self._strategy.need_chain and not self._strategy.drop_chain
        )
        # dim方向をsliceしてstrategyに投げるかどうか
        dim_slc = on_dim and not self._strategy.need_dim and not self._strategy.drop_dim

        # strategyにアレイを入れる際のindexを生成
        res = []
        if chain_slc and dim_slc:
            for chain_id in range(nchain):
                res_t = []
                for dim_id in range(ndim):
                    res_t.append(self._strategy.compute(array[chain_id, ..., dim_id]))
                res.append(res_t)
        elif chain_slc:
            for chain_id in range(nchain):
                res.append(self._strategy.compute(array[chain_id]))
        elif dim_slc:
            for dim_id in range(ndim):
                res.append(self._strategy.compute(array[..., dim_id]))
        else:
            res = self._strategy.compute(array)

        match array:
            case np.ndarray():
                res = np.asarray(res)
            case da.Array() | Delayed():
                logger.info("Execute dask compute.")
                res = da.compute(res)
                res = res[0]
                res = np.asarray(res)

        if on_chain and self._strategy.drop_chain:
            res = res[np.newaxis]
            res = np.repeat(res[None], nchain, axis=0)
        if on_dim and self._strategy.drop_dim:
            res = res[..., np.newaxis]
            res = np.repeat(res[None], ndim, axis=-1)

        idx = []
        resize_shape = []
        num_step = self.data.iats.shape[step_axis]
        if on_chain:
            resize_shape.append(nchain)
            idx.append(slice(None))
        idx.append(num_step)
        resize_shape.append(num_step + 1)
        if on_dim:
            resize_shape.append(ndim)

        self.data.iats.resize(tuple(resize_shape))
        self.data.iats.set(res, idx=tuple(idx))
        self.data.steps.append(np.asarray(nstep), axis=0)
        return self.data
