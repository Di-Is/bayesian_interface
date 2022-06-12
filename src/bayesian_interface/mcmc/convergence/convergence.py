import typing
from enum import Enum
from abc import abstractmethod, ABCMeta

import numpy as np
import dask.array as da

import bayesian_interface.data_structure as bay_data
from .pre_process import AbsPreprocess
from ...logger import Logger

logger = Logger(__name__)

# TODO: Add docs
# TODO: Add logging


class ConvergenceResult(bay_data.AbsData):
    # Criterion method name
    criterion_method: bay_data.AbsAttr
    # convergence flag
    convergence: bay_data.AbsAttr
    # The number of converge step
    # convergence_step: bay_data.AbsAttr
    # The Threshold convergence criterion
    threshold: bay_data.AbsAttr
    # Threshold Type
    threshold_type: bay_data.AbsAttr  # upper or lower equal
    # Number of Chain ID for check convergence
    #  chain_ids: bay_data.AbsAttr

    # The checked step
    steps: bay_data.AbsArray
    # The checked criterion values corresponding steps
    criterion_values: bay_data.AbsArray
    convergences: bay_data.AbsArray

    # convergence_steps: bay_data.AbsArray

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


class ThresholdType(Enum):
    equal = 0
    lower = 1
    lower_eq = 2
    upper = 3
    upper_eq = 4


def check_relationship(threshold_type: ThresholdType, a: np.ndarray, b) -> bool:
    match threshold_type:
        case ThresholdType.equal:
            return a == b
        case ThresholdType.lower:
            return a < b
        case ThresholdType.lower_eq:
            return a <= b
        case ThresholdType.upper:
            return a > b
        case ThresholdType.upper_eq:
            return a >= b


class AbsStrategy(metaclass=ABCMeta):
    @property
    @abstractmethod
    def expected_dim(self) -> int | tuple[int, ...]:
        """expected input array dimension
        :return: dimension number
        """
        pass

    @abstractmethod
    def compute(self, array: np.ndarray) -> np.ndarray | da.Array:
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

    @property
    @abstractmethod
    def threshold(self) -> float:
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


class Convergence:
    """
    1. execute convergence check using strategy class
    2. save criterion values to class
    """

    def __init__(
        self,
        strategy: AbsStrategy,
        data: ConvergenceResult = None,
        pre_process: typing.Optional[AbsPreprocess] = None,
        dask: bool = True,
    ):
        self._strategy = strategy
        self._pre_process = pre_process
        if data is None:
            data = ConvergenceResult()
        self.data = data
        self.dask = dask

    def check_initialized(self):
        names = [
            "criterion_method",
            "threshold_type",
            "criterion_values",
            "steps",
            "convergences",
            "convergence",
        ]
        flag = all([getattr(self.data, name).has() for name in names])
        return flag

    def check_convergence(
        self,
        array: np.ndarray | da.Array,
        on_chain: bool,
        on_dim: bool,
        threshold: typing.Optional[float] = None,
    ) -> ConvergenceResult:
        """
        :param array: mcmc chain
        :param on_chain:
        :param on_dim:
        :param threshold:
        :return:
        """
        logger.info(f"[{self._strategy.algorithm_name}] Start convergence check.")

        if array.ndim < sum([on_chain, on_dim]) + 1:
            raise ValueError("The dimension of the input array is missing.")

        if self._strategy.need_chain and not on_chain:
            raise ValueError("The dimension of the input array is missing.")

        if self._strategy.need_dim and not on_dim:
            raise ValueError("The dimension of the input array is missing.")

        if threshold is None:
            threshold = self._strategy.threshold

        if on_chain:
            nchain = array.shape[0]
            nstep = array.shape[1]
        else:
            nchain = 0
            nstep = array.shape[0]

        if on_dim:
            ndim = array.shape[-1]
        else:
            ndim = 0

        # Convert to dask array at pre-process
        if self.dask and not isinstance(array, da.Array):
            # NOTE: external dimensionに対する処理が書き換わるかも
            chunks = bay_data.make_chunks(array.shape, on_chain, on_dim)
            array = da.from_array(array, chunks=chunks)

        if self.data.criterion_method.has():
            if self.data.criterion_method.get() != self._strategy.algorithm_name:
                raise ValueError("mismatch strategy and data.")

        # do chain preprocess
        # TODO: 前処理前後にdask周りの処理を実装
        if self._pre_process is not None:
            # メリット: データの保存がしやすい
            # デメリット: preprocessが必要なのは、Strategyで処理するため
            # 2つには対応関係がある。Strategyに処理を記述したほうが自然
            array = self._pre_process.compute(array, on_chain, on_dim)

        # chain方向をsliceしてstrategyに投げるかどうか
        chain_slc = (
            on_chain and not self._strategy.need_chain and not self._strategy.drop_chain
        )
        # dim方向をsliceしてstrategyに投げるかどうか
        dim_slc = on_dim and not self._strategy.need_dim and not self._strategy.drop_dim

        # strategyにアレイを入れる際のindexを生成
        # NOTE: 速度が遅い可能性有り
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

        # compute criterion values
        if isinstance(array, da.Array):
            res = da.compute(res)[0]
            res = np.asarray(res)
        else:
            res = res

        # padding criterion value array
        if on_chain and self._strategy.drop_chain:
            res = np.repeat(res[None], nchain, axis=0)
        if on_dim and self._strategy.drop_dim:
            res = np.repeat(res[..., None], ndim, axis=-1)

        logger.info(
            f"[{self._strategy.algorithm_name}] "
            f"(min, mean, max): ({np.min(res):.3f}, {np.mean(res):.3f}, {np.max(res):.3f}) "
            f"threshold: {self._strategy.threshold:.3f}"
        )

        # initialize data
        # TODO:ifブロックをメソッド化するか、shape,maxshapeのリファクタリングを実施

        if not self.check_initialized():
            if on_chain and on_dim:
                cri_shape = (nchain, 0, ndim)
                cri_max = (nchain, None, ndim)
                convs_shape = (nchain, ndim)
                convs_max = (nchain, ndim)
            elif on_chain:
                cri_shape = (nchain, 0)
                cri_max = (nchain, None)
                convs_shape = nchain
                convs_max = nchain
            elif on_dim:
                cri_shape = (0, ndim)
                cri_max = (None, ndim)
                convs_shape = (ndim,)
                convs_max = (ndim,)
            else:
                cri_shape = (1,)
                cri_max = (None,)
                convs_shape = (1,)
                convs_max = (1,)
            # attr
            self.data.criterion_method.set(self._strategy.algorithm_name)
            self.data.threshold_type.set(self._strategy.threshold_type)
            # array
            self.data.criterion_values.create(
                cri_shape, maxshape=cri_max, dtype=np.float32
            )
            self.data.convergences.create(
                convs_shape, maxshape=convs_max, dtype=np.bool
            )
            self.data.steps.create(shape=(0,), maxshape=(None,), dtype=int)

        convergences = check_relationship(self._strategy.threshold_type, res, threshold)
        self.data.threshold.set(threshold)
        self.data.threshold_type.set(self._strategy.threshold_type)
        if on_chain:
            self.data.criterion_values.append(res[:, np.newaxis], axis=1)
        else:
            self.data.criterion_values.append(res[np.newaxis], axis=0)
        self.data.steps.append(np.asarray(nstep), axis=0)
        self.data.convergences.set(np.asarray(convergences))
        self.data.convergence.set(np.all(convergences))

        logger.info(f"[{self._strategy.algorithm_name}] End convergence check.")
        return self.data

    @property
    def algorithm_name(self) -> str:
        return self._strategy.algorithm_name
