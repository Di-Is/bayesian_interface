import os
import unittest

import numpy as np

from bayesian_interface.mcmc.mcmc.emcee_sampler import EmceeEnsemble
from bayesian_interface.data_structure.chain import SamplingResultFactory


class TestEnsembleSampler(unittest.TestCase):
    def test_sampling(self):
        ndim = 4
        nwalkers = 20
        nsteps = 1000

        def lnprob(p):
            x = np.random.rand(ndim)
            return -0.5 * np.sum(x * p**2)

        init_state = np.random.randn(nwalkers, ndim)
        sampler = EmceeEnsemble(nwalkers, ndim, lnprob)
        chain = sampler.sampling(init_state, nsteps)
        self.assertEqual(chain.shape, (nsteps, nwalkers, ndim))

        sampler = EmceeEnsemble(nwalkers, ndim, lnprob)
        chain = sampler.sampling(chain[0], nsteps)
        self.assertEqual(chain.shape, (nsteps, nwalkers, ndim))


if __name__ == "__main__":
    unittest.main()
