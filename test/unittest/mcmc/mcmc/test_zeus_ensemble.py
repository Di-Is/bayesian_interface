import os
import unittest

import numpy as np

from bayesian_interface.mcmc.mcmc.zeus_sampler import ZeusStrategy
from bayesian_interface.data_structure.chain import SamplingResultFactory


class TestEnsembleSampler(unittest.TestCase):
    def test_sampling(self):
        ndim = 2
        nwalkers = 10
        nsteps = 100

        def lnprob(p):
            x = np.random.rand(ndim)
            return -0.5 * np.sum(x * p**2)

        init_state = np.random.randn(nwalkers, ndim)
        sampler = ZeusStrategy(nwalkers, ndim, lnprob, maxiter=1e8)
        chain = sampler.sampling(init_state, nsteps)
        self.assertEqual(chain.shape, (nsteps, nwalkers, ndim))

        sampler = ZeusStrategy(nwalkers, ndim, lnprob, maxiter=1e8)
        chain = sampler.sampling(chain[0], nsteps)
        self.assertEqual(chain.shape, (nsteps, nwalkers, ndim))


if __name__ == "__main__":
    unittest.main()
