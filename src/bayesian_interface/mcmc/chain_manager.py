import typing

import numpy as np

from ..data_structure.chain import SamplingResult
from ..parallel import ThreadPool


def run_mcmc(fn, *args):
    return fn.run_mcmc(*args)


class ChainManager:
    def __init__(
        self,
        sampler_cls: typing.Callable,
        nchain: int,
        mpl_data: typing.Optional[dict[int, SamplingResult]] = None,
        pool=None,
        chain_worker: int = 1,
        **sampler_kwargs: dict,
    ):
        self._sampler_cls = sampler_cls
        # self._lnprob = lnprob
        self._nchain = nchain
        self._mpl_data = mpl_data
        self._chain_worker = chain_worker
        self._pool = pool
        self._sampler_kwargs = sampler_kwargs

    @property
    def _map(self) -> typing.Callable:
        if self._pool is None:
            _map = map
        else:
            _map = self._pool.map
        return _map

    def sampling(
        self, init_states: dict[int, np.ndarray], nsteps: dict[int, int]
    ) -> dict[int, SamplingResult]:
        """MCMC法でのサンプリングを実行する"""
        if init_states.keys() != nsteps.keys():
            raise ValueError("Not match nsteps.")
        chain_ids = list(init_states.keys())

        pool_lst = [PoolWrapper(ThreadPool, self._chain_worker) for _id in chain_ids]
        pool_lst = [None for _id in chain_ids]
        sampler_lst = [
            self._sampler_cls(
                **self._sampler_kwargs,
                data=self._mpl_data[_id] if self._mpl_data is not None else None,
                pool=pool_lst[_id],
            )
            for _id in chain_ids
        ]
        init_states_lst = [init_states[_id] for _id in chain_ids]
        nsteps_lst = [nsteps[_id] for _id in chain_ids]
        results = list(
            self._map(
                run_mcmc,
                sampler_lst,
                init_states_lst,
                nsteps_lst,
            )
        )
        results = {_id: results[_id] for _id in chain_ids}
        return results


class PoolWrapper:
    """実行時にプールを初期化するクラス"""

    def __init__(self, pool_cls: typing.Callable, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._pool_cls = pool_cls
        self._pool = None

    @property
    def map(self):
        if self._pool is None:
            self._pool = self._init_pool()
        return self._pool.map

    @property
    def submit(self):
        if self._pool is None:
            self._pool = self._init_pool()
        return self._pool.submit

    def _init_pool(self):
        return self._pool_cls(*self._args, **self._kwargs)
