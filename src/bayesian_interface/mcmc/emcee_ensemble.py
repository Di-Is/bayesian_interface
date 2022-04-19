import typing

import emcee
import numpy as np

from .mcmc_base import EnsembleMCMCBase
from ..logger import get_progress_bar
from ..misc_calc import split_nsteps
from bayesian_interface.data_structure.chain import SamplingResult, SamplingResultMaker


# Overriding __getstate__ for invalidating to delete pool member variable in emcee.EnsembleSampler
def __getstate__(self):
    return self.__dict__


emcee.EnsembleSampler.__getstate__ = __getstate__

SAMPLING_METHOD = "/mcmc/emcee/ensemble_sampler"


class EnsembleSampler(EnsembleMCMCBase):
    """Adapter class for emcee.EnsembleSampler"""

    def __init__(
        self,
        nwalkers: int,
        ndim: int,
        lnprob: typing.Callable,
        data: typing.Optional[SamplingResult] = None,
        chain_id: int = 0,
        nsteps_chunk: int = 100,
        **kwargs: dict,
    ) -> None:
        """Constractor
        :param nwalkers: the number of nwalkers
        :param ndim: the number of dimension
        :param lnprob: the log probability function
        :param data: the dataclass for sampling result
        :param chain_id: the chain id
        :param nsteps_chunk: the chunk size for spliting nsteps
        :param kwargs: kwargs for __init__ in emcee.EnsembleSampler class
        """
        self._data = self._init_data(data, chain_id, nwalkers, ndim)
        self._nsteps_chunk = nsteps_chunk
        self._chain_id = chain_id
        self._sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, **kwargs)

    def run_mcmc(
        self, init_state: np.ndarray, nsteps: int, progress: bool = True
    ) -> SamplingResult:
        """Executing mcmc sampling
        :param init_state: the sampling initial state
        :param nsteps: the number of sampling steps
        :param progress: whether display or not display
        :return: the sampling result
        """

        bar = get_progress_bar(
            display=progress,
            total=nsteps,
            desc=f"[chain {self._chain_id}]",
            position=self._chain_id,
            leave=False,
        )
        for i, nsteps_ in enumerate(split_nsteps(nsteps, self._nsteps_chunk)):
            self._sampler.run_mcmc(init_state, nsteps_)
            self._save_chain()
            init_state = self._sampler.get_last_sample()
            self._sampler.reset()
            bar.update(nsteps_)
        bar.close()

        return self._data

    @staticmethod
    def _init_data(
        data: typing.Optional[SamplingResult], chain_id: int, nwalkers: int, ndim: int
    ):
        """initializing sampling result data class
        :param data: the instance of sampling result data class
        :param chain_id: chain id
        :param nwalkers: the number of walker
        :param ndim: the number of parameter dimension
        :return: initialized data class
        """

        def init(dat: SamplingResult) -> SamplingResult:
            """initializing sampler result class
            :param dat: un-initialized data class
            :return: initialized data class
            """
            dat.sampler_name.set(SAMPLING_METHOD)
            dat.chain_id.set(chain_id)
            dat.chain.create((0, nwalkers, ndim), maxshape=(None, nwalkers, ndim))
            dat.lnprob.create((0, nwalkers), maxshape=(None, nwalkers))
            return dat

        if data is None:
            data = init(SamplingResultMaker().get())
        else:
            if not all(
                [
                    getattr(data, name).has()
                    for name in ["sampler_name", "chain_id", "chain", "lnprob"]
                ]
            ):
                data = init(data)
        return data

    def _save_chain(self) -> None:
        """Saving sampling result"""
        self._data.chain.append(self._sampler.get_chain(), axis=0)
        self._data.lnprob.append(self._sampler.get_log_prob(), axis=0)

    # @staticmethod
    # def _get_moves(moves: list[dict]) -> list:
    #     """emcee.movesのインスタンスを取得する
    #     :return: 取得したインスタンス
    #     """
    #     moves_en = []
    #     for move_dct in moves:
    #         name_method = move_dct["method"]
    #         try:
    #             move_class = emcee.moves.__dict__[name_method]
    #             keys_other = move_dct.keys() - {"method", "prob"}
    #             kwargs = {key: move_dct[key] for key in keys_other}
    #             moves_en.append((move_class(**kwargs), move_dct["prob"]))
    #         except KeyError:
    #             raise AttributeError(f"{name_method}はemcee.moves以下に定義されていません。")
    #     return moves_en
