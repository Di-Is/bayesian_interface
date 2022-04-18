import typing

import zeus
import numpy as np

from .mcmc_base import EnsembleMCMCBase
from ..logger import get_progress_bar
from ..misc_calc import split_nsteps
from ..data_structure.chain import SamplingResult, SamplingResultMaker


class EnsembleSampler(EnsembleMCMCBase):
    """emceeのサンプラー
    Adapterパターン
    """

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
        """
        :param chain_dat: データ格納用のクラス
        :param lnprob: 尤度関数
        :param nsteps_chunk: 何ステップおきにデータを保存するか
        """
        self._data = self._init_data(data, chain_id, ndim, nwalkers)
        self._nsteps_chunk = nsteps_chunk
        self._chain_id = chain_id
        self._sampler = zeus.EnsembleSampler(nwalkers, ndim, lnprob, **kwargs)

    @staticmethod
    def _init_data(data: typing.Optional[SamplingResult], chain_id, ndim, nwalkers):
        def init(dat):
            dat.sampler_name.set("/mcmc/zeus/ensemble_sampler")
            dat.chain_id.set(chain_id)
            dat.chain.create((0, nwalkers, ndim), maxshape=(None, nwalkers, ndim))
            dat.lnprob.create((0, nwalkers), maxshape=(None, nwalkers))
            return dat

        if data is None:
            data = SamplingResultMaker().get()
            data = init(data)
        else:
            if not all(
                [
                    getattr(data, name).has()
                    for name in ["sampler_name", "chain_id", "chain", "lnprob"]
                ]
            ):
                data = init(data)
        return data

    def run_mcmc(
        self, init_state: np.ndarray, nsteps: int, progress: bool = True
    ) -> SamplingResult:

        bar = get_progress_bar(
            display=progress,
            total=nsteps,
            desc=f"[chain {self._chain_id}]",
            position=self._chain_id,
            leave=False,
        )
        for i, nsteps_ in enumerate(split_nsteps(nsteps, self._nsteps_chunk)):
            self._sampler.run_mcmc(init_state, nsteps_, progress=False)
            self._save_chain()
            init_state = self._sampler.get_last_sample()
            self._sampler.reset()
            bar.update(nsteps_)
        bar.close()
        return self._data

    def _save_chain(self):
        """サンプリング結果を保存する
        :return:
        """
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
    #             move_class = zeus.moves.__dict__[name_method]
    #             keys_other = move_dct.keys() - {"method", "prob"}
    #             kwargs = {key: move_dct[key] for key in keys_other}
    #             moves_en.append((move_class(**kwargs), move_dct["prob"]))
    #         except KeyError:
    #             raise AttributeError(f"{name_method}はemcee.moves以下に定義されていません。")
    #     return moves_en
