import unittest

import numpy as np
import dask.array as da

import bayesian_interface.mcmc.autocorr as bay_acor


class TestNpDim1(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(), dask=False)

    def test_iats_ndim(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.iats.ndim, 1)

    def test_iats_shape(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.iats.shape, (1,))

    def test_steps_value(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps[:], np.asarray([1000]))

    def test_steps_shape(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps.shape, (1,))

    def test_steps_ndim(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps.ndim, 1)


class TestNpDim2(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(), dask=False)

    def test_iats_ndim(self):
        nstep, ndim = 1000, 10
        inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy())
        array = np.random.rand(nstep, ndim)
        res = self.inst.compute(array, False, True)
        self.assertEqual(res.iats.ndim, 2)

    def test_iats_shape(self):
        nstep, ndim = 1000, 10
        inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy())
        array = np.random.rand(nstep, ndim)
        res = self.inst.compute(array, False, True)
        self.assertEqual(res.iats.shape, (1, ndim))

    def test_steps_shape(self):
        nstep, ndim = 1000, 10
        inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy())
        array = np.random.rand(nstep, ndim)
        res = self.inst.compute(array, False, True)
        self.assertEqual(res.steps.shape, (1,))

    def test_steps_ndim(self):
        nstep, ndim = 1000, 10
        inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy())
        array = np.random.rand(nstep, ndim)
        res = self.inst.compute(array, False, True)
        self.assertEqual(1, res.steps.ndim)

    def test_steps_value(self):
        nstep, ndim = 1000, 10
        inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy())
        array = np.random.rand(nstep, ndim)
        res = self.inst.compute(array, False, True)
        self.assertEqual(res.steps[:], np.asarray([1000]))


class TestNpDim3(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(), dask=False)

    def test_iats_ndim(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.iats.ndim, 3)

    def test_iats_shape(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.iats.shape, (nchain, 1, ndim))

    def test_steps_shape(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.steps.shape, (1,))

    def test_steps_ndim(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(1, res.steps.ndim)

    def test_steps_value(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.steps[:], np.asarray([1000]))


class TestNpDim1Dask(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(), dask=True)

    def test_iats_ndim(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.iats.ndim, 1)

    def test_iats_shape(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.iats.shape, (1,))

    def test_steps_value(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps[:], np.asarray([1000]))

    def test_steps_shape(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps.shape, (1,))

    def test_steps_ndim(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps.ndim, 1)


class TestNpDim1Dask(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(), dask=True)

    def test_iats_ndim(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.iats.ndim, 1)

    def test_iats_shape(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.iats.shape, (1,))

    def test_steps_value(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps[:], np.asarray([1000]))

    def test_steps_shape(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps.shape, (1,))

    def test_steps_ndim(self):
        array = np.random.rand(1000)
        res = self.inst.compute(array, False, False)
        self.assertEqual(res.steps.ndim, 1)


class TestNpDim3Dask(unittest.TestCase):
    def setUp(self) -> None:
        self.inst = bay_acor.AutoCorrTime(bay_acor.FFTStrategy(), dask=True)

    def test_iats_ndim(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.iats.ndim, 3)

    def test_iats_shape(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.iats.shape, (nchain, 1, ndim))

    def test_steps_shape(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.steps.shape, (1,))

    def test_steps_ndim(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(1, res.steps.ndim)

    def test_steps_value(self):
        nchain, nstep, ndim = 2, 1000, 10
        array = np.random.rand(nchain, nstep, ndim)
        res = self.inst.compute(array, True, True)
        self.assertEqual(res.steps[:], np.asarray([1000]))


if __name__ == "__main__":
    unittest.main()
