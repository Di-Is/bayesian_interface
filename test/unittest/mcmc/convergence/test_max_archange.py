import unittest

import numpy as np

import bayesian_interface.mcmc.convergence as bay_conv
import bayesian_interface.mcmc.autocorr as bay_acor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self._fpath = "./mcmc.h5"

    def tearDown(self) -> None:
        pass

    def test_1d(self):
        iat_strategy = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(dask=False))
        inst = bay_conv.MaxArchangeStrategy(iat_strategy, dask=False)

        array = np.random.rand(100)
        res = inst.compute(array, 0)
        print(1, res.shape)

    def test_2d(self):
        iat_strategy = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(dask=False))
        inst = bay_conv.MaxArchangeStrategy(iat_strategy, dask=False)

        nstep = 100
        ndim = 2
        array = np.random.rand(nstep, ndim)
        res = inst.compute(array, 0)
        self.assertEqual((ndim,), res.shape)
