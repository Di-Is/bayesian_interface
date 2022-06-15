import os
import typing

import emcee.moves
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import dask.array as da
from concurrent.futures import ProcessPoolExecutor, Executor

import bayesian_interface.data_structure as bay_data
import bayesian_interface.mcmc.mcmc as bay_mcmc
import bayesian_interface.mcmc.convergence as bay_conv
import bayesian_interface.mcmc.autocorr as bay_acor
from bayesian_interface.logger import Logger

logger = Logger(__name__)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class McmcManagerFactory:
    @classmethod
    def create(
        cls,
        strategy: bay_mcmc.AbsStrategy,
        nchain: int = 1,
        data: typing.Optional[bay_mcmc.SamplingResult] = None,
        pool: typing.Optional[Executor] = None,
        chunksize: int = 100,
    ):
        samplers = {
            _id: bay_mcmc.MCMCSampler(strategy, chain_id=_id) for _id in range(nchain)
        }
        res = bay_mcmc.SamplerManager(
            samplers, data=data, pool=pool, chunksize=chunksize
        )
        return res


# class ConvFactory:
#     @classmethod
#     def create(
#         cls,
#         strategies: list[bay_conv.AbsStrategy],
#         data: list[bay_conv.ConvergenceResult],
#     ):
#         # if strategies.keys() != data.keys():
#         #      raise ValueError("Mismatch keys")
#         res = []
#         for st in strategies:
#             name = st.algorithm_name
#             idx = [d.criterion_method.get() for d in data].index(name)
#             res.append(bay_conv.Convergence(strategies[idx], data[idx]))
#
#         # res = [bay_conv.Convergence(strategies[key], data[key]) for key in data.keys()]
#         return res


class Convergence:
    def __init__(self, convs: typing.Optional[list[bay_conv.Convergence]] = None):
        self._convs = convs

    def check_convergence(self, chain: np.ndarray | da.Array):
        for conv in self._convs:
            res_conv = conv.check_convergence(chain, True, True)

    @property
    def convergence(self) -> bool:
        if self._convs is None:
            return True
        else:
            for conv in self._convs:
                if not conv.data.convergence.has():
                    return False
            return all([conv.data.convergence.get() for conv in self._convs])

    @property
    def conv_step(self) -> int | float:
        if self._convs is None:
            return 0
        else:
            for conv in self._convs:
                if not conv.data.convergence.has():
                    return np.nan
            return np.max([max(conv.data.steps.get()) for conv in self._convs])

    @property
    def data(self):
        dct = {}
        for conv in self._convs:
            dct[conv.algorithm_name] = conv.data

        return dct


# class ConvDataFactory:
#     @classmethod
#     def create(
#         cls,
#         strategies: list[bay_conv.AbsStrategy],
#         save_type: bay_data.SaveType,
#         each: typing.Optional[dict] = None,
#         **common,
#     ) -> list[bay_conv.ConvergenceResult]:
#
#         res = []
#         for st in strategies:
#             res.append(
#                 bay_conv.ConvergenceResult(
#                     save_type,
#                     each,
#                     gpath=f"/convergence/{st.algorithm_name}",
#                     **common,
#                 )
#             )
#         return res


class McmcSampling:
    def __init__(
        self,
        sampler: bay_mcmc.SamplerManager,
        pre_convs: Convergence = Convergence(),
        post_convs: Convergence = Convergence(),
        interval: int = 1000,
    ) -> None:
        self._sampler = sampler
        self._pre_convs = pre_convs
        self._post_convs = post_convs
        self._interval = interval

    @property
    def sampling_result(self) -> bay_mcmc.SamplingResult:
        return self._sampler.data

    @property
    def pre_convergence_result(self) -> dict[str, bay_conv.ConvergenceResult]:
        return self._pre_convs.data

    @property
    def post_convergence_result(self) -> dict[str, bay_conv.ConvergenceResult]:
        return self._post_convs.data

    @property
    def nburn(self) -> int:
        return self._pre_convs.conv_step

    def sampling(
        self,
        init_states: typing.Optional[np.ndarray] = None,
    ) -> None:

        if init_states is None:
            try:
                init_states = self._sampler.data.chain.get[:, -1]
            except Exception:
                raise ValueError(
                    "It has never been executed. Please start sampling with init-state."
                )

        # burn-in phase
        while not self._pre_convs.convergence:
            res_sampling = self._sampler.sampling(init_states, self._interval)
            chain_da = res_sampling.chain.get(copy=False)
            init_states = res_sampling.chain[:, -1]
            self._pre_convs.check_convergence(chain_da)

        # get burn-in length
        nburn = self._pre_convs.conv_step
        logger.info("converged")

        # burn-out phase(sampling phase)
        while not self._post_convs.convergence:
            res_sampling = self._sampler.sampling(init_states, self._interval)
            idx = (slice(None), slice(nburn, None))
            chain = res_sampling.chain.get(idx=idx, copy=False)
            init_states = res_sampling.chain[:, -1]
            self._post_convs.check_convergence(chain)


def main():
    nchain = 4
    nprocess = 4
    ndim = 10
    save_step = 1000
    check_step = 1000
    threshold_sample = 100000

    pool = ProcessPoolExecutor(nprocess)
    mu = 1.0
    cov = 0.5 - 10 * np.random.rand(ndim**2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    @nb.njit(cache=True, fastmath=True, nogil=True)
    def lnprob(p: np.ndarray) -> float:
        diff = p - mu
        return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

    # create samplers
    # sa_samp(nchain, lnprob, ndim, threshold_sample, pool, check_step, save_step)
    emcee_samp(nchain, lnprob, ndim, 512, threshold_sample, pool, check_step, save_step)


def sa_samp(
    nchain: int,
    lnprob: typing.Callable,
    ndim: int,
    threshold_sample: int = 1e6,
    pool: typing.Optional[Executor] = None,
    interval: int = 1000,
    save_step: int = 3000,
):
    sampler_kwargs = {"nparticle": 150}

    fpath = "mcmc.h5"
    if os.path.exists(fpath):
        os.remove(fpath)

    # sampler
    sampler_strategy = bay_mcmc.SampleAdaptiveStrategy(
        ndim=ndim, lnprob=lnprob, **sampler_kwargs
    )
    sampling_res = bay_mcmc.SamplingResult(
        save_type=bay_data.SaveType.hdf5, fpath=fpath
    )
    mgr = McmcManagerFactory.create(
        sampler_strategy, nchain, sampling_res, pool, chunksize=save_step
    )

    # convergence strategy
    pre_conv = Convergence(
        [
            bay_conv.Convergence(bay_conv.GRRank()),
            bay_conv.Convergence(bay_conv.StableGRMaxEigen()),
            bay_conv.Convergence(
                bay_conv.MaxArchangeStrategy(),
                pre_process=bay_conv.AutoCorrPreProcess(
                    bay_acor.AutoCorrTime(bay_acor.FFTStrategy())
                ),
            ),
        ]
    )

    post_conv = Convergence(
        [
            bay_conv.Convergence(
                bay_conv.ESSIATStrategy(
                    threshold=threshold_sample,
                ),
                pre_process=bay_conv.AutoCorrPreProcess(
                    bay_acor.AutoCorrTime((bay_acor.FFTStrategy()))
                ),
            )
        ]
    )

    # build sampling instance
    sampler = McmcSampling(mgr, pre_conv, post_conv, interval=interval)
    sampler.sampling(np.random.randn(nchain, 128, ndim))

    res = sampler.sampling_result
    nburn = sampler.nburn
    chain = res.chain[:, nburn:]
    import arviz

    fig, axes = plt.subplots(ndim, 2, figsize=(8, ndim * 4))
    for i in range(ndim):
        arviz.plot_trace(
            chain[..., i],
            # combine_dims={"x_dim_0", "draw"},
            var_names=["x"],
            compact=True,
            combined=False,
            axes=np.array([axes[i]]),
        )
    fig = plt.gcf()
    fig.savefig("trace.png")
    del fig


def emcee_samp(
    nchain: int,
    lnprob: typing.Callable,
    ndim: int,
    nwalker: int,
    threshold_sample: int = 1e6,
    pool: typing.Optional[Executor] = None,
    interval: int = 2000,
    save_step: int = 1000,
):

    fpath = "mcmc.h5"
    if os.path.exists(fpath):
        os.remove(fpath)

    # sampler
    sampler_strategy = bay_mcmc.EmceeEnsemble(
        ndim=ndim,
        lnprob=lnprob,
        nwalkers=nwalker,
        moves=[(emcee.moves.DEMove(), 1.0), (emcee.moves.DESnookerMove(), 0.1)],
    )
    sampling_res = bay_mcmc.SamplingResult(
        save_type=bay_data.SaveType.hdf5, fpath=fpath
    )
    mgr = McmcManagerFactory.create(
        sampler_strategy, nchain, sampling_res, pool, chunksize=save_step
    )

    # convergence strategy
    pre_conv = Convergence(
        [
            bay_conv.Convergence(bay_conv.Manual()),
            bay_conv.Convergence(
                bay_conv.GR(), pre_process=bay_conv.EnsembleCompressor()
            ),
            bay_conv.Convergence(
                bay_conv.GRRank(), pre_process=bay_conv.EnsembleCompressor()
            ),
            bay_conv.Convergence(
                bay_conv.StableGR(),
                pre_process=bay_conv.EnsembleCompressor(),
            ),
            bay_conv.Convergence(
                bay_conv.StableGRDeterminant(),
                pre_process=bay_conv.EnsembleCompressor(),
            ),
            bay_conv.Convergence(
                bay_conv.StableGRMaxEigen(), pre_process=bay_conv.EnsembleCompressor()
            ),
            bay_conv.Convergence(
                bay_conv.MaxArchangeStrategy(),
                pre_process=bay_conv.AutoCorrPreProcess(
                    bay_acor.AutoCorrTime(
                        bay_acor.CalcMeanStrategy(bay_acor.FFTStrategy())
                    )
                ),
            ),
            bay_conv.Convergence(
                bay_conv.MinAfactorStrategy(100, nwalker=nwalker),
                pre_process=bay_conv.AutoCorrPreProcess(
                    bay_acor.AutoCorrTime(
                        bay_acor.CalcMeanStrategy(bay_acor.FFTStrategy()),
                    )
                ),
            ),
            bay_conv.Convergence(
                bay_conv.ESSBulk(nwalker=nwalker),
                pre_process=bay_conv.pre_process.EnsembleCompressor(),
            ),
        ]
    )
    iat_inst = bay_acor.AutoCorrTime(bay_acor.CalcMeanStrategy(bay_acor.FFTStrategy()))
    post_conv = Convergence(
        [
            bay_conv.Convergence(
                bay_conv.ESSFromIAT(threshold_sample, nwalker=nwalker),
                pre_process=bay_conv.AutoCorrPreProcess(iat_inst),
            )
        ]
    )

    # build sampling instance
    sampler = McmcSampling(mgr, pre_conv, post_conv, interval=interval)
    init_state = 10.0 * np.random.randn(nchain, nwalker, ndim) + 10.0 * np.random.randn(
        nchain, nwalker, ndim
    )
    sampler.sampling(init_state)

    res = sampler.sampling_result
    nburn = sampler.nburn
    iat_max = iat_inst.data.iats[:, -1].max()
    chain = res.chain[:, nburn :: int(iat_max)].reshape(nchain, -1, ndim)
    import arviz

    fig, axes = plt.subplots(ndim, 2, figsize=(8, ndim * 3))
    for i in range(ndim):
        arviz.plot_trace(
            chain[..., i],
            # combine_dims={"x_dim_0", "draw"},
            var_names=["x"],
            compact=True,
            combined=False,
            axes=np.array([axes[i]]),
        )
    fig = plt.gcf()
    fig.savefig("trace.png")
    del fig

    for key in sampler.pre_convergence_result.keys():
        PlotCriterion().plot(sampler.pre_convergence_result[key])


class PlotCriterion:
    def plot(self, conv_result: bay_conv.ConvergenceResult):

        ndim = conv_result.convergences.shape[-1]
        fig, axes = plt.subplots(4, 4, figsize=(12, 8))
        for dim_id in range(ndim):
            axes.flat[dim_id].plot(
                conv_result.steps, conv_result.criterion_values[0, ..., dim_id]
            )
            axes.flat[dim_id].axhline(conv_result.threshold.get(), ls="dashed")
        plt.savefig(f"{conv_result.criterion_method}.png")
        plt.close()


if __name__ == "__main__":
    main()
