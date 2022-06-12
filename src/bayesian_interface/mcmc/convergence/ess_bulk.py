import typing
import math

import numpy as np

from .convergence import AbsStrategy, MagnitudeRelation
import bayesian_interface.mcmc.autocorr.autocorrtime as iat
import bayesian_interface.mcmc.autocorr.ensemble as iat_ens
from bayesian_interface.mcmc.autocorr import AutoCorrResultFactory
from .misc import check_dimension


class ESSIATStrategy(AbsStrategy):
    def __init__(self, threshold: typing.Optional[float] = None) -> None:
        self.threshold = threshold

    @classmethod
    @property
    def threshold(cls) -> float:  # noqa
        return 100000.0

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 3

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "effective_sample_size_bulk"

    @classmethod
    @property
    def threshold_type(cls) -> MagnitudeRelation:  # noqa
        return MagnitudeRelation.upper

    def compute(self, array: np.ndarray) -> np.ndarray:
        from arviz.stats.diagnostics import _ess_bulk

        nchain, nsteps = array.shape[:2]
        arr = np.asarray([_ess_bulk(array[i]) for i in range(nchain)])
        return arr.sum()
