import unittest

import numpy as np
import dask.array as da

import bayesian_interface.mcmc.convergence as bay_conv


class TestManual(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_conv.Convergence(bay_conv.Manual())

    def test_n(self):
        nchain, nstep, ndim = 2, 3000, 5
        arr = np.random.rand(nchain, nstep, ndim)
        res = self.inst.check_convergence(arr, True, True)
        print(res)


class TestGRMaxEigen(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_conv.Convergence(bay_conv.StableGRMaxEigen())

    def test_n(self):
        nchain, nstep, ndim = 2, 3, 5
        arr = np.random.rand(nchain, nstep, ndim)
        res = self.inst.check_convergence(arr, True, True)

    def test_e_nochain_onchain(self):
        nstep, ndim = 3, 5
        arr = np.random.rand(nstep, ndim)
        with self.assertRaises(ValueError):
            res = self.inst.check_convergence(arr, False, True)

    def test_e_nochain(self):
        nstep, ndim = 3, 5
        arr = np.random.rand(nstep, ndim)
        with self.assertRaises(ValueError):
            res = self.inst.check_convergence(arr, True, True)

    def test_e_nodim_ondim(self):
        nchain, nstep = 3, 5
        arr = np.random.rand(nchain, nstep)
        with self.assertRaises(ValueError):
            res = self.inst.check_convergence(arr, True, False)

    def test_e_nodim(self):
        nchain, nstep = 3, 5
        arr = np.random.rand(nchain, nstep)
        with self.assertRaises(ValueError):
            res = self.inst.check_convergence(arr, True, True)

    def test_e_nochain_nodim(self):
        nstep = 5
        arr = np.random.rand(nstep)
        with self.assertRaises(ValueError):
            res = self.inst.check_convergence(arr, False, False)

    def test_e_nochain_nodim_onchain_ondim(self):
        nstep = 5
        arr = np.random.rand(nstep)
        with self.assertRaises(ValueError):
            res = self.inst.check_convergence(arr, True, True)
