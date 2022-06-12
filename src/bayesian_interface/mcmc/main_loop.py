import typing

import emcee.moves
import matplotlib.pyplot as plt
import numpy as np
import numba as nb

from bayesian_interface.mcmc.mcmc.mcmc import MCMCSampler
from bayesian_interface.mcmc.mcmc.emcee_sampler import EmceeEnsemble
from bayesian_interface.mcmc.mcmc.zeus_sampler import ZeusStrategy

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
    ndim = 2
    x = np.random.rand(ndim)

    mu = 0.0
    cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    @nb.njit()
    def lnprob(p):
        diff = p - mu
        return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

    sampler_strategy = EmceeEnsemble(
        nwalkers, ndim, lnprob, moves=[(emcee.moves.KDEMove(), 1)]
    )
    sampler = MCMCSampler(sampler_strategy, save_step=1000)

    # Convergence check
    conv_strategy = Strategy()
    conv_checker = Convergence(conv_strategy)

    nsteps = 1000
    threshold_sample = 100000
    while True:
        # MCMC
        init_state = np.random.rand(nwalkers, ndim)
        res_sampling = sampler.sampling(init_state, nsteps)
        chain = res_sampling.chain.get(copy=False)
        init_state = chain[-1]

        res_conv = conv_checker.check_convergence(chain.transpose([1, 0, 2]))
        if res_conv.convergence.get():
            break

    print(f"{res_conv.convergence.get()=}")
    print(f"{res_conv.criterion_values.get()[-1]=}")

    nburn = len(chain)

    # IAT check
    iat_strategy = StrategyCalcMean(FFTStrategy())
    iat_checker = AutoCorrTime(iat_strategy)
    # ESS calc
    ess_checker = EssCalc(Convert(nwalkers))

    while True:
        res_sampling = sampler.sampling(init_state, nsteps)
        chain = res_sampling.chain.get(idx=slice(nburn, None), copy=False)
        init_state = chain[-1]
        res_iat = iat_checker.compute(chain)

        res_ess = ess_checker.compute(res_iat.iats.get())
        if res_ess.esss.get(idx=-1).min() > threshold_sample:
            break
        # residual nsteps for achieving threshold ess sampling
        nsteps = int(
            (threshold_sample - res_ess.esss.get(idx=-1).min())
            * res_iat.iats.get(idx=-1).max()
            / nwalkers
            * 1.1
        )
        print(f"new_nsteps: {nsteps}")

    print(f"{res_iat.iats.get()=}")
    print(f"{res_ess.esss.get()=}")

    # Plot
    import arviz

    thin = int(res_iat.iats.get(idx=-1).max())
    axes = arviz.plot_posterior(
        chain[::thin].reshape((1,) + chain[::thin].shape),
        combine_dims={"x_dim_0", "draw"},
    )
    fig = plt.gcf()
    fig.savefig("post.png")
    del fig

    fig, axes = plt.subplots(ndim, 2)
    for i in range(ndim):
        arviz.plot_trace(
            chain.reshape((1,) + chain.shape)[..., i],
            # combine_dims={"x_dim_0", "draw"},
            var_names=["x"],
            compact=True,
            combined=False,
            axes=np.array([axes[i]]),
        )
    fig = plt.gcf()
    fig.savefig("trace.png")
    del fig


if __name__ == "__main__":
    main()
