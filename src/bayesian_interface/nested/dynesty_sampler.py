import typing

import numpy as np
import dynesty

from nested_base import NestedBase


class NestedSampler(NestedBase):
    def __init__(
        self,
        lnlike: typing.Callable,
        lnprior: typing.Callable,
        ndim: int,
        data: typing.Optional[None] = None,
        chain_id: int = 1,
        **kwargs
    ):
        super().__init__()
        self._data = data
        self._chain_id = chain_id
        self._sampler = dynesty.NestedSampler(lnlike, lnprior, ndim, **kwargs)

    def sampling(self, **kwargs):
        self._sampler.run_nested(**kwargs)


class DynamicNestedSampler(NestedBase):
    def __init__(
        self,
        lnlike: typing.Callable,
        lnprior: typing.Callable,
        ndim: int,
        data: typing.Optional[None] = None,
        chain_id: int = 1,
        **kwargs
    ):
        super().__init__()
        self._data = data
        self._chain_id = chain_id
        self._sampler = dynesty.NestedSampler(lnlike, lnprior, ndim, **kwargs)

    def sampling(self, **kwargs):
        self._sampler.run_nested(**kwargs)
