import typing

import numpy as np

from .convergence import AbsStrategy, ThresholdType
import bayesian_interface.mcmc.autocorr.autocorrtime as iat
import bayesian_interface.mcmc.autocorr.ensemble as iat_ens
from .misc import check_dimension


class MinAfactorStrategy(AbsStrategy):
    def __init__(
        self, autocorr: iat.AutoCorrTime, threshold: typing.Optional[float] = 50
    ) -> None:
        self._autocorr = autocorr
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def expected_dim(self) -> int | tuple[int, ...]:
        return 3, 4

    @classmethod
    @property
    def algorithm_name(cls) -> str:  # noqa
        return "minAfactor"

    @classmethod
    @property
    def threshold_type(cls) -> ThresholdType:  # noqa
        return ThresholdType.upper

    def compute(self, array: np.ndarray) -> np.ndarray:

        nsteps = array.shape[1]
        if array.ndim > 3:
            extra = array.shape[2]
        else:
            extra = 1

        if (
            self._autocorr.data.steps.has()
            and nsteps in self._autocorr.data.steps.get()
        ):
            idx = np.where(self._autocorr.data.steps == nsteps)[0][0]
            iats = self._autocorr.data.iats.get(idx=(slice(None), idx))
        else:
            iats = self._autocorr.compute(array).iats.get(idx=(slice(None), -1))
        return nsteps * extra / iats
