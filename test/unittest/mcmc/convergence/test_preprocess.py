import unittest

import numpy as np
import dask.array as da

import bayesian_interface.mcmc.convergence as bay_conv
import bayesian_interface.mcmc.autocorr as bay_acor


class TestAutoCorrPreProcess(unittest.TestCase):
    def setUp(self):
        self.inst = bay_conv.AutoCorrPreProcess(
            bay_acor.AutoCorrTime(bay_acor.FFTStrategy())
        )

    def test_n_1(self):
        nchain, nstep, ndim = 4, 1000, 5
        arr = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(arr, True, True)
        print(res)
