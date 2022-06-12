import typing
from concurrent.futures import ProcessPoolExecutor

import emcee.moves
import matplotlib.pyplot as plt
import numpy as np
import numba as nb

from bayesian_interface.mcmc.mcmc.manager import SamplerManager
from bayesian_interface.mcmc.mcmc.mcmc import MCMCSampler
from bayesian_interface.mcmc.mcmc.emcee_sampler import EmceeEnsemble

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
    nwalkers = 256
    ndim = 3
    nsteps = 1000
    nchain = 2
    nprocess = 2
    chunksize = 200

    mu = 0.0
    cov = 0.5 - np.random.rand(ndim, ndim)
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    @nb.njit()
    def lnprob(p):
        diff = p - mu
        return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

    sampler_strategy = EmceeEnsemble(
        nwalkers, ndim, lnprob, moves=[(emcee.moves.WalkMove(), 1.0)]
    )

    samplers = {
        _id: MCMCSampler(sampler_strategy, save_step=chunksize, chain_id=_id)
        for _id in range(nchain)
    }
    pool = ProcessPoolExecutor(nprocess)
    # pool = None
    mgr = SamplerManager(samplers, pool=pool, chunksize=chunksize)

    # MCMC
    init_states = {_id: np.random.rand(nwalkers, ndim) for _id in range(nchain)}
    res_sampling = mgr.sampling(init_states, nsteps)

    chain = res_sampling.chain.get()
    # fig, axes = plt.subplots(nchain, 1)
    # for i, ax in enumerate(axes):
    #     ax.plot(chain[i, ..., 0])
    # plt.show()

    # Convergence check
    conv_strategy = Strategy()
    conv_checker = Convergence(conv_strategy)

    res_conv = conv_checker.check_convergence(chain.mean(axis=2))

    print(f"{res_conv.convergence.get()=}")
    print(f"{res_conv.criterion_values.get()[-1]=}")

    # IAT check
    iat_strategy = StrategyCalcMean(FFTStrategy())
    iat_checker = AutoCorrTime(iat_strategy)
    res_iat = iat_checker.compute(chain)

    print(f"{res_iat.iats.get()=}")

    # ESS calc
    ess_checker = EssCalc(Convert(nwalkers))
    res_ess = ess_checker.compute(res_iat.iats.get())

    print(f"{res_ess.esss.get()=}")

    # Plot


if __name__ == "__main__":
    main()
