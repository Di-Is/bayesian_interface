from abc import ABCMeta, abstractmethod

import numpy as np


class AbsMCMC(metaclass=ABCMeta):
    """Abstract class for mcmc sampler"""

    @abstractmethod
    def run_mcmc(self, initial_state: np.ndarray, nsteps: int, **kwargs):
        pass


class MCMCBase(AbsMCMC):
    """Base class for normal mcmc sampler"""

    def run_mcmc(self, initial_state: np.ndarray, nsteps: int, **kwargs):
        raise NotImplementedError("Not override.")


class EnsembleMCMCBase(AbsMCMC):
    """Base class for ensemble mcmc sampler"""

    def run_mcmc(self, initial_state: np.ndarray, nsteps: int, **kwargs):
        raise NotImplementedError("Not override.")
