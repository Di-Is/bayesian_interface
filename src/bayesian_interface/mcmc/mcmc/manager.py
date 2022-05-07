import typing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bayesian_interface.data_structure.chain import SamplingResult
from bayesian_interface.mcmc.mcmc.mcmc import MCMCSampler


def sampling(
    sampler: MCMCSampler,
    initial_state: np.ndarray,
    nsteps: int,
    kwargs: typing.Optional[dict] = None,
):
    """
    :param sampler: initialized sampler class
    :param initial_state: sampling kwargs
    :param nsteps: sampling kwargs
    :param kwargs: sampling kwargs
    :return: sampler_result
    """
    kwargs = {} if kwargs is None else kwargs
    return sampler.sampling(initial_state, nsteps, **kwargs)


class SamplerManager:
    def __init__(
        self,
        samplers: dict[int, MCMCSampler],
        data: typing.Optional[SamplingResult] = None,
        pool: typing.Optional[ProcessPoolExecutor] = None,
    ) -> None:
        """Constractor
        :param samplers: non-initialized sampler class
        :param pool: pool class
        """

        self._samplers = samplers
        self._pool = pool
        if data is None:
            data = None
        self._data = data

    @property
    def nchain(self) -> int:
        return len(self._samplers)

    def sampling(
        self,
        initial_states: dict[int, np.ndarray],
        nsteps: dict[int, int],
        kwargs: typing.Optional[dict[int, dict]] = None,
    ) -> dict[int, SamplingResult]:
        """Executing mcmc sampling
        :param initial_states: sampling method kwargs for each
        :param nsteps: sampling method kwargs for each
        :param kwargs: sampling method kwargs for each
        :return: sampling result data cla ss
        """
        kwargs = {_id: {} for _id in range(self.nchain)} if kwargs is None else kwargs

        chain_ids = range(self.nchain)

        # select map func
        if self._pool is None:
            _map = map
        else:
            _map = self._pool.map

        # do sampling
        samplers = self._samplers
        results = list(
            _map(
                sampling,
                [samplers[_id] for _id in chain_ids],
                [initial_states[_id] for _id in chain_ids],
                [nsteps[_id] for _id in chain_ids],
                [kwargs[_id] for _id in chain_ids],
            )
        )
        results = {_id: results[_id] for _id in chain_ids}

        # Setting result data into sampler member
        # because changes in sub-process are not taken over to main-process
        if self._pool is not None:
            for _id in chain_ids:
                self._samplers[_id].data = results[_id]

        return results
