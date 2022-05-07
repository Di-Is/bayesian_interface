import typing

import emcee
import numpy as np

from .mcmc import AbsStrategy


# Overriding __getstate__ for invalidating to delete pool member variable in emcee.EnsembleSampler
def __getstate__(self):
    return self.__dict__


emcee.EnsembleSampler.__getstate__ = __getstate__


class EmceeStrategy(AbsStrategy):
    """Adapter class for emcee.EnsembleSampler"""

    def __init__(
        self,
        nwalkers: int,
        ndim: int,
        lnprob: typing.Callable,
        dtype: typing.Any = np.float64,
        **kwargs,
    ) -> None:
        """Constractor
        :param nwalkers: the number of nwalkers
        :param ndim: the number of dimension
        :param lnprob: the log probability function
        :param nsteps_chunk: the chunk size for spliting nsteps
        :param kwargs: kwargs for __init__ in emcee.EnsembleSampler class
        """
        super().__init__()
        self._sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            backend=emcee.backends.Backend(dtype=dtype),
            **kwargs,
        )

    def method_name(self) -> str:
        return "/mcmc/emcee/ensemble_sampler"

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
        :param progress:
        :return: the sampling result
        """
        self._sampler.run_mcmc(initial_state, nsteps, progress=progress, **kwargs)
        chain = self._sampler.get_chain()
        self._sampler.reset()
        return chain
