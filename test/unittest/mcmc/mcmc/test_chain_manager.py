import os
import unittest
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bayesian_interface.mcmc.mcmc.manager import SamplerManager
import bayesian_interface.mcmc.mcmc.emcee_sampler as emcee_ensemble
from bayesian_interface.mcmc.mcmc.mcmc import MCMCSampler
import bayesian_interface.mcmc.mcmc.zeus_sampler as zeus_ensemble
from bayesian_interface.parallel import ThreadPool, PoolWrapper


os.environ["OMP_NUM_THREADS"] = "1"


class TestChainManager(unittest.TestCase):
    def test_sampling_emcee(self):
        ndim = 14
        nwalkers = 256
        nsteps = 1000
        nchain = 2

        import numba

        @numba.njit(nogil=True)
        def lnprob(p):
            x = np.random.rand(ndim)
            return -0.5 * np.sum(x * p**2)

        init_states = {i: np.random.rand(nwalkers, ndim) for i in range(nchain)}
        strategy = emcee_ensemble.EmceeStrategy(nwalkers, ndim, lnprob)
        samplers = {i: MCMCSampler(strategy, i) for i in range(nchain)}
        nsteps = {i: nsteps for i in range(nchain)}
        kwargs = {i: {"progress": False} for i in range(nchain)}

        cm = SamplerManager(
            samplers,
            ProcessPoolExecutor(nchain),
        )

        mpl_data = cm.sampling(init_states, nsteps, kwargs)
        for _id, data in mpl_data.items():
            self.assertEqual(data.chain.shape, (nsteps[_id], nwalkers, ndim))
        mpl_data = cm.sampling(init_states, nsteps, kwargs)
        for _id, data in mpl_data.items():
            self.assertEqual(data.chain.shape, (nsteps[_id] * 2, nwalkers, ndim))

    def test_sampling_emcee_pool(self):
        ndim = 14
        nwalkers = 256
        nsteps = 2000
        nchain = 2
        worker = 2

        import numba

        x = np.ones(ndim) + np.random.randn(ndim)

        @numba.njit(nogil=True)
        def lnprob(p):
            return -0.5 * np.sum((x - p) ** 2)

        init_states = {i: np.random.rand(nwalkers, ndim) for i in range(nchain)}
        strategy = emcee_ensemble.EmceeStrategy(
            nwalkers, ndim, lnprob, pool=PoolWrapper(ThreadPool, False, worker)
        )
        samplers = {i: MCMCSampler(strategy, i) for i in range(nchain)}
        nsteps = {i: nsteps for i in range(nchain)}
        kwargs = {i: {"progress": False} for i in range(nchain)}

        cm = SamplerManager(
            samplers,
            ProcessPoolExecutor(nchain),
        )
        mpl_data = cm.sampling(init_states, nsteps, kwargs)

    def test_sampling_zeus(self):
        ndim = 14
        nwalkers = 256
        nsteps = 1000
        nchain = 2

        import numba

        @numba.njit(nogil=True)
        def lnprob(p):
            x = np.random.rand(ndim)
            return -0.5 * np.sum(x * p**2)

        init_states = {i: np.random.rand(nwalkers, ndim) for i in range(nchain)}
        strategy = zeus_ensemble.ZeusStrategy(nwalkers, ndim, lnprob, maxiter=1e8)
        samplers = {i: MCMCSampler(strategy, i) for i in range(nchain)}
        nsteps = {i: nsteps for i in range(nchain)}
        kwargs = {i: {"progress": False} for i in range(nchain)}

        cm = SamplerManager(
            samplers,
            ProcessPoolExecutor(nchain),
        )

        mpl_data = cm.sampling(init_states, nsteps, kwargs)
        for _id, data in mpl_data.items():
            self.assertEqual(data.chain.shape, (nsteps[_id], nwalkers, ndim))
        mpl_data = cm.sampling(init_states, nsteps, kwargs)
        for _id, data in mpl_data.items():
            self.assertEqual(data.chain.shape, (nsteps[_id] * 2, nwalkers, ndim))


if __name__ == "__main__":
    unittest.main()
