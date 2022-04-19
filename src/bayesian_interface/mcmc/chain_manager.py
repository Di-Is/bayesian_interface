import typing

import numpy as np

from ..mcmc.mcmc_base import AbsMCMC
from ..data_structure.chain import SamplingResult
from ..parallel import ThreadPool, PoolWrapper


def run_mcmc(sampler: AbsMCMC, *args):
    """
    :param sampler: initialized sampler class
    :param args: run_mcmc arguments
    :return: sampler_result
    """
    return sampler.run_mcmc(*args)


class ChainManager:
    def __init__(
        self,
        sampler_cls: AbsMCMC.__class__,
        nchain: int,
        mpl_data: typing.Optional[dict[int, SamplingResult]] = None,
        pool=None,
        chain_worker: int = 1,
        **sampler_kwargs: dict,
    ) -> None:
        """Constractor
        :param sampler_cls: non-initialized sampler class
        :param nchain: the number of chain
        :param mpl_data: the dict of sampling result data class
        :param pool: pool class
        :param chain_worker: the number of worker for each chain
        :param sampler_kwargs: sampler __init__ kwargs
        """
        self._sampler_cls = sampler_cls
        self._nchain = nchain
        self._mpl_data = mpl_data
        self._chain_worker = chain_worker
        self._pool = pool
        self._sampler_kwargs = sampler_kwargs

    def sampling(
        self, init_states: dict[int, np.ndarray], nsteps: dict[int, int]
    ) -> dict[int, SamplingResult]:
        """Executing mcmc sampling
        :param init_states: the initial state
        :param nsteps: the number of sampling step
        :return: sampling result data class
        """

        if init_states.keys() != nsteps.keys() and len(nsteps) == self._nchain:
            raise ValueError("Not match nsteps.")
        chain_ids = list(init_states.keys())

        if self._pool is None:
            _map = map
        else:
            _map = self._pool.map

        pool_lst = [PoolWrapper(ThreadPool, self._chain_worker) for _id in chain_ids]
        mpl_data = (
            self._mpl_data if self._mpl_data is not None else len(chain_ids) * [None]
        )
        sampler_lst = [
            self._sampler_cls(
                data=mpl_data[_id],
                pool=pool_lst[_id],
                **self._sampler_kwargs,
            )
            for _id in chain_ids
        ]
        init_states_lst = [init_states[_id] for _id in chain_ids]
        nsteps_lst = [nsteps[_id] for _id in chain_ids]
        results = list(
            _map(
                run_mcmc,
                sampler_lst,
                init_states_lst,
                nsteps_lst,
            )
        )
        results = {_id: results[_id] for _id in chain_ids}
        return results
