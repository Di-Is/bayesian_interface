from abc import ABCMeta, abstractmethod

import numpy as np


class AbsMCMC(metaclass=ABCMeta):
    @abstractmethod
    def run_mcmc(self, initial_state: np.ndarray, nsteps: int, **kwargs):
        pass


class MCMCBase(AbsMCMC):
    def run_mcmc(self, initial_state: np.ndarray, nsteps: int, **kwargs):
        raise NotImplementedError("Not override.")


class EnsembleMCMCBase(AbsMCMC):
    """Ensemble方式のMCMCのAdapterクラス"""

    def run_mcmc(self, initial_state: np.ndarray, nsteps: int, **kwargs):
        raise NotImplementedError("Not override.")

    def set_pool(self, value):
        self._sampler.pool = value
