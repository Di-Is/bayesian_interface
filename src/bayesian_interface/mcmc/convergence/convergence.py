import typing
from enum import Enum
from abc import abstractmethod, ABCMeta

import numpy as np
import dask.array as da

import bayesian_interface.data_structure as bay_data
from .pre_process import AbsPreprocess
from ...logger import Logger

logger = Logger("mcmc.convergence")

# TODO: Add logging


class ConvergenceResult(bay_data.AbsData):
    """Class to store the result of convergence check"""

    # Criterion method name
    criterion_method: bay_data.AbsAttr
    # convergence flag
    convergence: bay_data.AbsAttr
    # The number of converge step
    convergence_step: bay_data.AbsAttr
    # The Threshold convergence criterion
    threshold: bay_data.AbsAttr
    # Threshold Type
    threshold_type: bay_data.AbsAttr  # upper or lower equal
    # The checked step
    # shape: (check_step)
    steps: bay_data.AbsArray
    # The checked criterion values corresponding steps
    # shape: (Op[chain], check_step, Op[dim])
    criterion_values: bay_data.AbsArray
    # convergence flags
    # shape: (Op[chain], Op[dim]), minimum is (1,), not scaler.
    convergences: bay_data.AbsArray
    # The number of converge steps
    # shape: (Op[chain], Op[dim]), minimum is (1,), not scaler.
    convergence_steps: bay_data.AbsArray

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


class MagnitudeRelation(Enum):
    """enum for magnitude relation"""

    equal = 0
    lower = 1
    lower_eq = 2
    upper = 3
    upper_eq = 4


def check_relation(
    relation_type: MagnitudeRelation, a: float | np.ndarray, b: float
) -> bool | np.ndarray:
    """check magnitude relation between array and float or float and float
    :param relation_type: magnitude relation type
    :param a: array or float (ex. calculated criterion value)
    :param b: float (ex. threshold value)
    :return: flag
    """
    match relation_type:
        case MagnitudeRelation.equal:
            return a == b
        case MagnitudeRelation.lower:
            return a < b
        case MagnitudeRelation.lower_eq:
            return a <= b
        case MagnitudeRelation.upper:
            return a > b
        case MagnitudeRelation.upper_eq:
            return a >= b


class AbsStrategy(metaclass=ABCMeta):
    """Class for computing convergence statistics
    This class is based on the strategy pattern.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @abstractmethod
    def compute(self, array: np.ndarray) -> float:
        """Compute convergence criterion value
        :param array: mcmc chain or pre-processed mcmc chain
        :return: convergence criterion value
        """
        pass

    @property
    @abstractmethod
    def expected_dim(self) -> int | tuple[int, ...]:
        """Expected input array dimension
        :return: dimension number
        """
        pass

    @classmethod
    @property
    @abstractmethod
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        """The relationship between the threshold and
        the statistic that is considered to be convergent.
        :return: threshold type
        """
        pass

    @classmethod
    @property
    @abstractmethod
    def algorithm_name(cls) -> str:  # noqa
        """algorithm name of convergence criterion
        :return: algorithm name
        """
        pass

    @property
    @abstractmethod
    def need_chain(self) -> bool:
        """Whether multiple mcmc chains are needed to compute convergence statistics
        :return: flag
        """
        pass

    @property
    @abstractmethod
    def drop_chain(self) -> bool:
        """Whether the dimension of the mcmc chain is lost after computing the convergence statistic
        :return: flag
        """
        pass

    @property
    @abstractmethod
    def need_dim(self) -> bool:
        """Whether multiple parameter dimensions are needed to compute convergence statistics
        :return: flag
        """
        pass

    @property
    @abstractmethod
    def drop_dim(self) -> bool:
        """Whether the dimension of the parameter is lost after computing the convergence statistic
        :return: flag
        """
        pass


class Convergence:
    """Class to determine convergence of MCMC chains"""

    def __init__(
        self,
        strategy: AbsStrategy,
        data: ConvergenceResult = None,
        pre_process: typing.Optional[AbsPreprocess] = None,
        dask: bool = True,
    ):
        """Constractor
        :param strategy: strategy class for calculating convergence criterion. (dependency injection)
        :param data: the data class for saving convergence criterion
        :param pre_process: A class that preprocesses the MCMC chain before computing convergence statistics.
        :param dask: whether to calculate using dask.
        """
        self._strategy = strategy
        self._pre_process = pre_process
        if data is None:
            data = ConvergenceResult()
        self.data = data
        self.dask = dask

    def check_initialized(self) -> bool:
        """Check if data is initialized.
        :return: flag
        """
        names = [
            "criterion_method",
            "threshold_type",
            "criterion_values",
            "steps",
            "convergence_steps",
            "convergence_step",
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
        :param array: mcmc chain.
        :param on_chain: Whether to have a chain dimension.
        :param on_dim: Whether to have a dim(param) dimension.
        :param threshold: convergence Threshold.
        :return: convergence result.
        """
        logger.info(f"[{self._strategy.algorithm_name}] Start convergence check.")

        # Error Check
        if array.ndim < sum([on_chain, on_dim]) + 1:
            raise ValueError("The dimension of the input array is missing.")

        if self._strategy.need_chain and not on_chain:
            raise ValueError("The dimension of the input array is missing.")

        if self._strategy.need_dim and not on_dim:
            raise ValueError("The dimension of the input array is missing.")

        if self.data.criterion_method.has():
            if self.data.criterion_method.get() != self._strategy.algorithm_name:
                raise ValueError("mismatch strategy and data.")

        if threshold is None:
            threshold = self._strategy.threshold

        nchain, nstep, ndim = bay_data.get_array_dim(array.shape, on_chain, on_dim)

        # Convert to dask array before pre-process
        if self.dask and not isinstance(array, da.Array):
            chunks = bay_data.make_chunks(array.shape, on_chain, on_dim)
            array = da.from_array(array, chunks=chunks)

        # Pre-process the MCMC chain
        # TODO: 前処理前後にdask周りの処理を実装
        if self._pre_process is not None:
            array = self._pre_process.compute(array, on_chain, on_dim)

        # Convert to dask array after pre-process
        if self.dask and not isinstance(array, da.Array):
            chunks = bay_data.make_chunks(array.shape, on_chain, on_dim)
            array = da.from_array(array, chunks=chunks)

        # Whether to slice and pass in the chain direction when computing convergence statistics in the strategy class
        chain_slc = (
            on_chain and not self._strategy.need_chain and not self._strategy.drop_chain
        )
        # Whether to slice and pass in the dim direction when computing convergence statistics in the strategy class
        dim_slc = on_dim and not self._strategy.need_dim and not self._strategy.drop_dim

        # execute compute
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

        if isinstance(array, da.Array):
            res = da.compute(res)[0]
        res = np.asarray(res)

        # Align the dimensions of the computed convergence statistic
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
        if not self.check_initialized():
            if on_chain and on_dim:
                cri_shape = (nchain, 0, ndim)
                cri_max = (nchain, None, ndim)
                convs_shape = (nchain, ndim)
            elif on_chain:
                cri_shape = (nchain, 0)
                cri_max = (nchain, None)
                convs_shape = nchain
            elif on_dim:
                cri_shape = (0, ndim)
                cri_max = (None, ndim)
                convs_shape = (ndim,)
            else:
                cri_shape = (1,)
                cri_max = (None,)
                convs_shape = (1,)

            # attr
            self.data.criterion_method.set(self._strategy.algorithm_name)
            self.data.threshold_type.set(self._strategy.threshold_type)
            # array
            self.data.criterion_values.create(
                cri_shape, maxshape=cri_max, dtype=np.float32
            )
            self.data.convergences.create(
                convs_shape, maxshape=convs_shape, dtype=np.bool
            )
            self.data.convergences.set(np.full(convs_shape, False))
            self.data.convergence_steps.create(
                convs_shape, maxshape=convs_shape, dtype=int
            )
            self.data.steps.create(shape=(0,), maxshape=(None,), dtype=int)

        convergences = check_relation(self._strategy.threshold_type, res, threshold)
        convergence = np.all(convergences)
        self.data.threshold.set(threshold)
        self.data.threshold_type.set(self._strategy.threshold_type)
        if on_chain:
            self.data.criterion_values.append(res[:, np.newaxis], axis=1)
        else:
            self.data.criterion_values.append(res[np.newaxis], axis=0)

        if self.data.convergence_step.has():
            if self.data.convergence_step == -1 and convergence:
                self.data.convergence_step.set(nstep)
            else:
                self.data.convergence_step.set(-1)
        else:
            if convergence:
                self.data.convergence_step.set(nstep)
            else:
                self.data.convergence_step.set(-1)

        if self.data.convergences.has():
            old_flags = self.data.convergences[:]
        else:
            old_flags = np.full(convergences.shape, False)

        if self.data.convergence_steps.has():
            conv_steps = self.data.convergence_steps[:]
        else:
            conv_steps = np.full(convergences.shape, -1)

        for idx in np.ndindex(*convergences.shape):
            match (bool(convergences[idx]), bool(old_flags[idx])):
                case (True, True):
                    pass
                case (True, False):
                    conv_steps[idx] = nstep
                case (False, True):
                    conv_steps[idx] = -1
                case (False, False):
                    conv_steps[idx] = -1
        self.data.convergence_steps.set(conv_steps)

        self.data.steps.append(np.asarray(nstep), axis=0)
        self.data.convergences.set(np.asarray(convergences))
        self.data.convergence.set(np.all(convergences))
        logger.info(f"[{self._strategy.algorithm_name}] End convergence check.")
        return self.data

    @property
    def algorithm_name(self) -> str:
        """algorithm name for convergence check
        :return: algorithm name
        """
        return self._strategy.algorithm_name
