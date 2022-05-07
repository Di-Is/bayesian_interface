import numpy as np

from ..sampler_base import AbsSampler


class NestedBase(AbsSampler):
    """Base class for normal mcmc sampler"""

    def __init__(self):
        self._data = None

    def sampling(self, initial_state: np.ndarray, nsteps: int, **kwargs):
        raise NotImplementedError("Not override.")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
