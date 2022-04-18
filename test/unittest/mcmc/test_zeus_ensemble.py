import os
import unittest

import numpy as np

from bayesian_interface.mcmc.zeus_ensemble import EnsembleSampler
from bayesian_interface.data_structure.chain import SamplingResultMaker


class TestEnsembleSampler(unittest.TestCase):
    def test_memory(self):
        ndim = 4
        nwalkers = 10
        nsteps = 100

        def lnprob(p):
            x = np.random.rand(ndim)
            return -0.5 * np.sum(x * p**2)

        init_state = np.random.randn(nwalkers, ndim)
        sampler = EnsembleSampler(ndim, nwalkers, lnprob, maxiter=1e8)
        data = sampler.run_mcmc(init_state, nsteps, progress=True)
        sampler = EnsembleSampler(ndim, nwalkers, lnprob, data)
        data = sampler.run_mcmc(data.chain.get(0), nsteps, progress=True)

    def test_hdf5(self):
        ndim = 4
        nwalkers = 10
        nsteps = 100

        def lnprob(p):
            x = 1.0 / np.random.rand(ndim)
            return -0.5 * np.sum(x * p**2)

        fpath = "test.h5"
        data = SamplingResultMaker().get("hdf5", fpath=fpath, gpath="mcmc")
        init_state = np.random.randn(nwalkers, ndim)
        sampler = EnsembleSampler(ndim, nwalkers, lnprob, data, maxiter=1e8)
        data = sampler.run_mcmc(init_state, nsteps, progress=True)
        sampler = EnsembleSampler(ndim, nwalkers, lnprob, data)
        data = sampler.run_mcmc(data.chain.get(0), nsteps, progress=True)
        os.remove(fpath)


if __name__ == "__main__":
    unittest.main()
