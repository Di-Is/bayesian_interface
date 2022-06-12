import typing
import time
from concurrent.futures import Executor
import enum
from copy import deepcopy

import numpy as np

from bayesian_interface.mcmc.mcmc.mcmc import (
    MCMCSampler,
    split_nsteps,
    SamplingResult,
)
from ...logger import get_progress_bar, Logger
from ...parallel import DummyExecutor

logger = Logger(__name__)


class TaskState(enum.Enum):
    running = 0
    done = 1
    wait = 2
    finish = 3


class SamplerManager:
    sleep_time = 1e-2

    def __init__(
        self,
        samplers: dict[int, MCMCSampler],
        chunksize: int = 100,
        data: SamplingResult = None,
        pool: Executor = DummyExecutor(),
    ) -> None:
        """Constractor
        :param samplers: non-initialized sampler class
        :param pool: pool class
        """

        self._samplers = samplers
        self._pool = pool
        if data is None:
            data = SamplingResult()
        self.data = data
        self._chunksize = chunksize

    @property
    def nchain(self) -> int:
        return len(self._samplers)

    def init_data(
        self,
        step_back_shape: tuple[..., int],
        dtype: typing.Any,
        chain_ids: list[int],
        sampler_names: list[str],
    ):
        self.data.chain.create(
            shape=(self.nchain, 0) + step_back_shape,
            maxshape=(None, 0) + step_back_shape,
            dtype=dtype,
        )
        self.data.chain_id.set(chain_ids)
        self.data.sampler_name.set(sampler_names)

    def check_initialized(self) -> bool:
        names = ["chain", "sampler_name", "chain_id"]
        flags = [getattr(self.data, name).has() for name in names]
        return all(flags)

    @staticmethod
    def test(
        sampler: MCMCSampler, init_state: np.ndarray, nsteps: int, **kwargs
    ) -> MCMCSampler:
        sampler.sampling(init_state, nsteps, **kwargs)
        return sampler

    def _pool_sampling(
        self,
        initial_states: np.ndarray,
        nsteps: int,
        progress: bool = True,
        kwargs: typing.Optional[dict] = None,
    ):

        initial_states = deepcopy(initial_states)
        chain_ids = range(self.nchain)
        steps = split_nsteps(nsteps, self._chunksize).tolist()

        que_dat = {_id: [] for _id in chain_ids}
        tsk_sts = {_id: None for _id in chain_ids}
        tsk_ids = {_id: 0 for _id in chain_ids}
        states = {_id: TaskState.wait for _id in chain_ids}
        bars = {
            _id: get_progress_bar(
                display=progress,
                total=nsteps,
                desc=f"[chain {_id}]",
                position=_id,
                leave=True,
            )
            for _id in chain_ids
        }

        # すべてのタスク終了かつ、保存終了までループ
        add_count = 0
        while add_count < len(steps):
            # check state
            for _id in chain_ids:

                if states[_id] is TaskState.finish:
                    continue

                # state update
                if states[_id] is TaskState.running and tsk_sts[_id].done():
                    states[_id] = TaskState.done

                if states[_id] is TaskState.done:
                    res_sampler = tsk_sts[_id].result()
                    # update state
                    initial_states[_id] = res_sampler.data.chain.get(idx=-1)
                    que_dat[_id].append(res_sampler.data)
                    res_sampler.data = SamplingResult()
                    self._samplers[_id] = res_sampler
                    bars[_id].update(steps[tsk_ids[_id]])
                    tsk_ids[_id] += 1
                    if tsk_ids[_id] == len(steps):
                        states[_id] = TaskState.finish
                    else:
                        states[_id] = TaskState.wait

                if states[_id] is TaskState.wait:
                    tsk_sts[_id] = self._pool.submit(
                        self.test,
                        self._samplers[_id],
                        initial_states[_id],
                        steps[tsk_ids[_id]],
                        **{"progress": False},
                        **kwargs[_id],
                    )
                    states[_id] = TaskState.running

            # concat result
            if all([que for que in que_dat.values()]):
                add_count += 1

                res_samps = [que_dat[_id].pop(0) for _id in chain_ids]
                array = np.stack([res.chain.get() for res in res_samps])

                if not self.check_initialized():
                    self.data.chain.create(
                        shape=(self.nchain, 0) + res_samps[0].chain.shape[1:],
                        maxshape=(None, None) + res_samps[0].chain.shape[1:],
                        dtype=array.dtype,
                    )
                    self.data.chain_id.set([res.chain_id.get() for res in res_samps])
                    self.data.sampler_name.set(
                        [res.sampler_name.get() for res in res_samps]
                    )
                self.data.chain.append(array, axis=1)

            else:
                time.sleep(self.sleep_time)

    def sampling(
        self,
        initial_states: dict[int, np.ndarray],
        nsteps: int,
        progress: bool = True,
        kwargs: typing.Optional[dict[int, dict]] = None,
    ) -> SamplingResult:
        """Executing mcmc sampling
        :param initial_states: sampling method kwargs for each
        :param nsteps: sampling method kwargs for each
        :param progress: sampling method kwargs for each
        :param kwargs: sampling method kwargs for each
        :return: sampling result data cla ss
        """
        logger.info("[Sampling] Start MCMC sampling.")
        kwargs = {_id: {} for _id in range(self.nchain)} if kwargs is None else kwargs
        self._pool_sampling(initial_states, nsteps, progress, kwargs)
        logger.info("[Sampling] End MCMC sampling.")
        return self.data
