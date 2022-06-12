import typing

import zeus
import numpy as np

from .mcmc import AbsStrategy


class ZeusStrategy(AbsStrategy):
    """Adapter class for zeus.EnsembleSampler"""

    def __init__(
        self,
        nwalkers: int,
        ndim: int,
        lnprob: typing.Callable,
        dtype: typing.Any = np.float64,
        **kwargs: typing.Any,
    ) -> None:
        """Constractor
        :param nwalkers: the number of nwalkers
        :param ndim: the number of dimension
        :param lnprob: the log probability function
        :param kwargs: kwargs for __init__ in emcee.EnsembleSampler class
        """
        super().__init__()
        self._sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob, **kwargs)

    @classmethod
    @property
    def method_name(cls) -> str:
        return "/mcmc/zeus/ensemble_sampler"

    def sampling(
        self,
        initial_state: np.ndarray,
        nsteps: int,
        *,
        progress: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Executing mcmc sampling
        :param initial_state: the sampling initial state
        :param nsteps: the number of sampling steps
        :param progress: whether display or not display
        :return: the sampling result
        """

        self._sampler.run_mcmc(initial_state, nsteps, progress=False, **kwargs)  # noqa
        chain = self._sampler.get_chain()
        self._sampler.reset()
        return chain
