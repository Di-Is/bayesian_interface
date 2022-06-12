import typing

import emcee.moves
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from concurrent.futures import ProcessPoolExecutor, Executor

from bayesian_interface.mcmc.mcmc.mcmc import MCMCSampler
from bayesian_interface.mcmc.mcmc.emcee_sampler import EmceeEnsemble
from bayesian_interface.mcmc.mcmc.zeus_sampler import ZeusStrategy
from bayesian_interface.mcmc.mcmc.manager import SamplerManager

from bayesian_interface.mcmc.convergence.convergence import Convergence
from bayesian_interface.mcmc.convergence.gr_rank import Strategy

from bayesian_interface.mcmc.autocorr.autocorrtime import AutoCorrTime
from bayesian_interface.mcmc.autocorr.ensemble import (
    StrategyFlattenCalc,
    StrategyCalcMean,
)
from bayesian_interface.mcmc.autocorr.normal import FFTStrategy

from bayesian_interface.mcmc.ess.ess import ConvertIAT2ESS, EssCalc
from bayesian_interface.mcmc.ess.convert import Convert


def main():
    nchain = 2
    nprocess = 2
    nwalkers = 256
    ndim = 10
    save_step = 100
    check_step = 1000
    threshold_sample = 100000

    pool = ProcessPoolExecutor(nprocess)

    mu = 0.0
    cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    @nb.njit()
    def lnprob(p):
        diff = p - mu
        return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

    func(nchain, lnprob, threshold_sample, pool, check_step, save_step)


def func(
    nchain: int,
    lnprob: typing.Callable,
    threshold_sample: int = 1e6,
    pool: typing.Optional[Executor] = None,
    check_step: int = 1000,
    save_step: int = 100,
):
    nwalkers = 256
    ndim = 10
    nsteps = check_step
    chunksize = save_step

    sampler_strategy = EmceeEnsemble(
        nwalkers, ndim, lnprob, moves=[(emcee.moves.KDEMove(), 1)]
    )

    samplers = {
        _id: MCMCSampler(sampler_strategy, save_step=chunksize, chain_id=_id)
        for _id in range(nchain)
    }
    mgr = SamplerManager(samplers, pool=pool, chunksize=chunksize)

    # Convergence check
    conv_strategy = Strategy()
    conv_checker = Convergence(conv_strategy)

    # MCMC
    init_states = {_id: np.random.rand(nwalkers, ndim) for _id in range(nchain)}
    converge = False
    while not converge:
        res_sampling = mgr.sampling(init_states, nsteps)
        chain = res_sampling.chain.get()
        init_states = {_id: chain[_id, -1] for _id in range(nchain)}

        res_conv = conv_checker.check_convergence(chain.mean(axis=2), threshold=1.1)
        converge = res_conv.convergence.get()

    print(f"{res_conv.convergence.get()=}")
    print(f"{res_conv.criterion_values.get()[-1]=}")

    nburn = chain.shape[1]

    # IAT check
    iat_strategy = StrategyCalcMean(FFTStrategy())
    iat_checker = AutoCorrTime(iat_strategy)
    # ESS calc
    ess_checker = EssCalc(Convert(nwalkers))
    ess = 0
    while ess < threshold_sample:
        res_sampling = mgr.sampling(init_states, nsteps)
        chain = res_sampling.chain.get(
            idx=(slice(None), slice(nburn, None)), copy=False
        )
        init_states = {_id: chain[_id, -1] for _id in range(nchain)}
        res_iat = iat_checker.compute(chain)
        iats = res_iat.iats.get(idx=(slice(None), -1))

        res_ess = ess_checker.compute(res_iat.iats.get())
        esss = res_ess.esss.get(idx=(slice(None), -1))
        ess = esss.sum(axis=0).min()
        # if ess > threshold_sample:
        #     break
        # # residual nsteps for achieving threshold ess sampling
        # resid_ess = threshold_sample - ess
        #
        # nsteps = int((resid_ess * iats.max() / nwalkers / nchain) * 1.1)
        # print(f"ess: {ess}, new_nsteps: {nsteps}")

    print(f"{int(ess)=}")


if __name__ == "__main__":
    main()
