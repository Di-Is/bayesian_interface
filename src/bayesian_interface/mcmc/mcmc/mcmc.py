import typing
from abc import ABCMeta, abstractmethod

import numpy as np

from bayesian_interface.data_structure.chain import (
    SamplingResult,
    SamplingResultFactory,
)
from ...logger import get_progress_bar


class AbsStrategy(metaclass=ABCMeta):
    @abstractmethod
    def sampling(self, initial_state: np.ndarray, nsteps: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        pass


class MCMCSampler:
    """Base class for normal mcmc sampler"""

    def __init__(
        self,
        strategy: AbsStrategy,
        chain_id: int = 0,
        dtype: typing.Any = np.float64,
        data: typing.Optional[SamplingResult] = None,
        save_step: int = 100,
    ):
        self._strategy = strategy
        if data is None:
            data = SamplingResultFactory.create()
        self._data = data
        self._chain_id = chain_id
        self._dtype = dtype
        self._save_step = save_step

    def sampling(
        self, initial_state: np.ndarray, nsteps: int, progress: bool = True
    ) -> SamplingResult:

        if not self.check_initialized():
            self._data.sampler_name.set(self._strategy.method_name)
            self._data.chain_id.set(self._chain_id)
            self._data.chain.create(
                shape=(0,) + initial_state.shape,
                dtype=self._dtype,
                maxshape=(None,) + initial_state.shape,
            )

        for step in get_progress_bar(
            iterable=split_nsteps(nsteps, self._save_step),
            display=progress,
            # total=nsteps,
            desc=f"[chain {self._chain_id}]",
            position=self._chain_id,
            leave=False,
        ):
            chain = self._strategy.sampling(initial_state, step)
            self._data.chain.append(chain, axis=0)
            initial_state = chain[0]
        return self._data

    def check_initialized(self) -> bool:
        names = ("chain_id", "sampler_name", "chain")
        flag = all([getattr(self._data, name).has() for name in names])
        return flag

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


def split_nsteps(nsteps: int, nsteps_chunk: int) -> np.ndarray:
    """nstepsをチャンク刻みに分解する
    :param nsteps: MCMCチェーンの長さ
    :param nsteps_chunk: チャンクの長さ
    :return: 分割されたnstepsのリスト
    """
    if nsteps <= nsteps_chunk:
        chunks_nsteps = [nsteps]
    else:
        num_split = int(np.ceil(nsteps / nsteps_chunk))
        chunks_nsteps = []
        for i in range(num_split):
            if i == num_split - 1:
                value = nsteps - sum(chunks_nsteps)
            else:
                value = nsteps_chunk
            chunks_nsteps.append(value)

    return np.array(chunks_nsteps, dtype=np.int64)
