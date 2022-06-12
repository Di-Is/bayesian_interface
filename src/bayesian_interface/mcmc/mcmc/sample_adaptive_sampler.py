import typing

import numpy as np
from numba import njit
from numba._dispatcher import Dispatcher

from .mcmc import AbsStrategy


@njit("f8(f8[:],f8[:,:],f8[:])")
def proposal(mu: np.ndarray, cov: np.ndarray, sita: float):
    """# 規格化していない提案分布qの対数確率密度

    :param mu:
    :param cov:
    :param sita:
    :return:
    """
    A = -0.5 * np.linalg.slogdet(cov)[1]
    B = -0.5 * (sita - mu).T @ np.linalg.solve(cov, (sita - mu))
    return A + B


# 提案分布qからのサンプリング
@njit
def sampling_multinormal(mu: np.ndarray, cov: np.ndarray, ndim: int) -> np.ndarray:
    return np.linalg.cholesky(cov) @ np.random.randn(ndim) + mu


# 平均値の逐次更新
@njit()
def update_mean(mean, new_s, old_s, N):
    mu = mean + 1 / N * (new_s - old_s)
    return mu


def sa_mcmc(lnprob, nparticle, nsteps, ndim, s, s_mean, s_cov, s_replace, q, r):
    # # Step.1:初期化
    nsteps += 1
    random_number = np.random.rand(nsteps)
    chain = np.zeros((nsteps, ndim))

    # Step.2:サンプリング
    cnt = 0
    for i in range(nsteps):
        # 多次元正規分布からサンプリング
        s_ = sampling_multinormal(s_mean, s_cov, ndim)
        q[-1] = lnprob(s_)
        for n in range(nparticle + 1):
            s_replace[:, :] = s[:, :]
            s_replace[n, :] = s_
            s_mean_replace = update_mean(s_mean, s_, s[n, :], nparticle)
            s_cov_replace = np.cov(s_replace[:-1, :].T)
            p = proposal(s_mean_replace, s_cov_replace, sita=s[n, :])
            r[n] = p - q[n]
        r = np.exp(r - r.max())
        r = r / np.sum(r)

        # 棄却ステップ:J=N+1以外の場合は採択
        c = 0
        for j in range(nparticle + 1):
            c += r[j]
            if c > random_number[i]:
                J = j
                break
        s_mean = update_mean(s_mean, s_, s[J, :], nparticle)
        s[J, :] = s_
        s_cov = np.cov(s[:-1, :].T)
        q[J] = q[-1]
        if J != nparticle + 1:
            chain[cnt] = s_
            cnt += 1
    return chain[: cnt - 1, :], s, s_mean, s_cov, s_replace, q, r


sa_mcmc_jit = njit(sa_mcmc)


class SampleAdaptiveSampler:
    def __init__(
        self, nparticle: int, ndim: int, lnprob: typing.Callable, dtype: typing.Any
    ):
        self._lnprob = lnprob
        self._nparticle = nparticle
        self._ndim = ndim
        self._s = np.random.randn(self._nparticle + 1, self._ndim)
        self._s_mean = np.zeros(self._ndim)
        self._s_replace = np.zeros((self._nparticle + 1, self._ndim))
        self._q = np.zeros(self._nparticle + 1)
        self._r = np.zeros(self._nparticle + 1)

        # 平均、共分散行列及び提案分布から得られる対数確率密度の初期値の設定
        for d in range(self._ndim):
            self._s_mean[d] = self._s[:-1, d].mean()
        self._s_cov = np.cov(self._s[:-1, :].T)
        for n in range(self._nparticle):
            self._q[n] = self._lnprob(self._s[n, :])

    def run_mcmc(self, init_state: np.ndarray, nsteps: int) -> np.ndarray:

        if isinstance(self._lnprob, Dispatcher):
            func = sa_mcmc_jit
        else:
            func = sa_mcmc

        (
            data,
            self._s,
            self._s_mean,
            self._s_cov,
            self._s_replace,
            self._q,
            self._r,
        ) = func(
            self._lnprob,
            self._nparticle,
            nsteps,
            self._ndim,
            self._s,
            self._s_mean,
            self._s_cov,
            self._s_replace,
            self._q,
            self._r,
        )
        return data


class SampleAdaptiveStrategy(AbsStrategy):
    """Adapter class for emcee.EnsembleSampler"""

    def __init__(
        self,
        nparticle: int,
        ndim: int,
        lnprob: typing.Callable,
        dtype: typing.Any = np.float64,
        **kwargs,
    ) -> None:
        """Constractor
        :param nwalkers: the number of nwalkers
        :param ndim: the number of dimension
        :param lnprob: the log probability function
        :param nsteps_chunk: the chunk size for spliting nsteps
        :param kwargs: kwargs for __init__ in emcee.EnsembleSampler class
        """
        super().__init__()
        self._sampler = SampleAdaptiveSampler(
            nparticle,
            ndim,
            lnprob,
            dtype,
            **kwargs,
        )

    @classmethod
    @property
    def method_name(self) -> str:
        return "/mcmc/samcmc/samcmc_sampler"

    def sampling(
        self,
        initial_state: np.ndarray,
        nsteps: int,
        *,
        progress: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Executing mcmc sampling
        :param initial_state: the sampling initial state
        :param nsteps: the number of sampling steps
        :param progress:
        :return: the sampling result
        """
        chain = self._sampler.run_mcmc(initial_state, nsteps, **kwargs)
        #  chain = self._sampler.get_chain()
        #    self._sampler.reset()
        return chain
