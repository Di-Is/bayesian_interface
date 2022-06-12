import typing

import matplotlib.pyplot as plt
import numpy as np
import numba as nb

from bayesian_interface.mcmc.mcmc.mcmc import MCMCSampler
from bayesian_interface.mcmc.mcmc.emcee_sampler import EmceeEnsemble
from bayesian_interface.mcmc.mcmc.zeus_sampler import ZeusStrategy
from bayesian_interface.mcmc.mcmc.sample_adaptive_sampler import SampleAdaptiveStrategy

from bayesian_interface.mcmc.convergence.convergence import Convergence
from bayesian_interface.mcmc.convergence.gr_rank import Strategy

from bayesian_interface.mcmc.autocorr.autocorrtime import AutoCorrTime
from bayesian_interface.mcmc.autocorr.ensemble import StrategyFlattenCalc
from bayesian_interface.mcmc.autocorr.normal import FFTStrategy

from bayesian_interface.mcmc.ess.ess import ConvertIAT2ESS, EssCalc
from bayesian_interface.mcmc.ess.convert import Convert


def main():
    nwalkers = 100
    ndim = 2

    mu = 0.0
    cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    @nb.njit()
    def lnprob(p):
        diff = p - mu
        return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

    sampler_strategy = SampleAdaptiveStrategy(
        nwalkers,
        ndim,
        lnprob,
    )
    sampler = MCMCSampler(sampler_strategy, save_step=1000)

    # MCMC
    init_state = np.random.rand(nwalkers, ndim)
    nsteps = 5000
    res_sampling = sampler.sampling(init_state, nsteps)
    chain = res_sampling.chain.get()
    print(chain.shape)
    plt.plot(chain[..., 0])
    plt.plot(chain[..., 1])
    plt.show()

    # Convergence check
    conv_strategy = Strategy()
    conv_checker = Convergence(conv_strategy)

    # res_conv = conv_checker.check_convergence(chain.transpose([1, 0, 2]))
    #   res_conv = conv_checker.check_convergence(chain)

    # print(f"{res_conv.convergence.get()=}")
    # print(f"{res_conv.criterion_values.get()[-1]=}")

    # IAT check
    # iat_strategy = StrategyFlattenCalc(FFTStrategy())
    iat_strategy = FFTStrategy()
    iat_checker = AutoCorrTime(iat_strategy)
    res_iat = iat_checker.compute(chain.reshape(1, nsteps, ndim))
    # res_iat = iat_checker.compute(chain)
    print(f"{res_iat.iats.get()=}")

    # ESS calc
    ess_checker = EssCalc(Convert())
    res_ess = ess_checker.compute(res_iat.iats.get())

    print(f"{res_ess.esss.get()=}")

    # Plot


if __name__ == "__main__":
    main()
