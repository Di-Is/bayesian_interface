import typing

import numpy as np
import numba as nb

from bayesian_interface.mcmc.mcmc.mcmc import MCMCSampler
from bayesian_interface.mcmc.mcmc.emcee_sampler import EmceeStrategy
from bayesian_interface.mcmc.mcmc.zeus_sampler import ZeusStrategy

from bayesian_interface.mcmc.convergence.convergence import Convergence
from bayesian_interface.mcmc.convergence.gr_rank import Strategy

from bayesian_interface.mcmc.autocorr.autocorrtime import AutoCorrTime
from bayesian_interface.mcmc.autocorr.ensemble import StrategyFlattenCalc
from bayesian_interface.mcmc.autocorr.normal import FFTStrategy

from bayesian_interface.mcmc.ess.ess import ConvertIAT2ESS, EssCalc
from bayesian_interface.mcmc.ess.convert import Convert


def main():
    nwalkers = 256
    ndim = 100
    x = np.random.rand(ndim)

    @nb.njit()
    def lnprob(p):
        return -0.5 * np.sum(x * p**2)

    sampler_strategy = EmceeStrategy(
        nwalkers,
        ndim,
        lnprob,
    )
    sampler = MCMCSampler(sampler_strategy, save_step=1000)

    # MCMC
    init_state = np.random.rand(nwalkers, ndim)
    nsteps = 3000
    res_sampling = sampler.sampling(init_state, nsteps)
    chain = res_sampling.chain.get()

    # Convergence check
    conv_strategy = Strategy()
    conv_checker = Convergence(conv_strategy)

    res_conv = conv_checker.check_convergence(chain.transpose([1, 0, 2]))

    print(f"{res_conv.convergence.get()=}")
    print(f"{res_conv.criterion_values.get()[-1]=}")

    # IAT check
    iat_strategy = StrategyFlattenCalc(FFTStrategy())
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
