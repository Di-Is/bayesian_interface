import unittest

import numpy as np
import dask.array as da

import bayesian_interface.mcmc.autocorr as bay_acor


class TestNp2DaskDim1(unittest.TestCase):
    @unittest.expectedFailure
    def test_nstep_boundary(self):
        """Check step boundary
        :return:
        :Note:
        エラーが発生するが、nstep=1で
        自己相関時間は計算しないため、expected failurを付与
        """
        nstep = 1
        array = np.random.rand(nstep)
        inst = bay_acor.GPStrategy()
        res = inst.compute(array)
        self.assertIsInstance(res, da.Array)
        res.compute()

    def test_type(self):
        """Check variable type.
        nstep=1 boundary test is failre
        nstep=10 is normal test
        """
        nstep = 100
        array = np.random.rand(nstep)
        inst = bay_acor.GPStrategy()
        res = inst.compute(array)
        self.assertIsInstance(res, da.Array)
        res.compute()

    def test_shape(self):
        """Check result shape
        :return:
        """
        nstep = 100
        array = np.random.rand(nstep)
        inst = bay_acor.GPStrategy()
        res = inst.compute(array)
        self.assertEqual(res.shape, ())
        self.assertIsInstance(res, da.Array)
        res.compute()

    def test_zero_length(self):
        nstep = 0
        array = np.random.rand(nstep)
        inst = bay_acor.GPStrategy()
        res = inst.compute(array)
        self.assertEqual(res.shape, ())
        self.assertIsInstance(res, da.Array)
        with self.assertRaises(Exception):
            res.compute()


class TestNp2DaskDim2(unittest.TestCase):
    def test_type(self):
        """Check variable type.
        ndim=1 is boundary test
        ndim=10 is normal test
        """
        nstep = 100
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = np.random.rand(nstep, ndim)
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertIsInstance(res, da.Array)

    def test_shape(self):
        """Check result shape
        :return:
        """
        nstep = 100
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = np.random.rand(nstep, ndim)
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))

    def test_zero_length(self):
        nstep = 0
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = np.random.rand(nstep, ndim)
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))
                self.assertIsInstance(res, da.Array)
                with self.assertRaises(Exception):
                    res.compute()


class TestNp2DaskDim3(unittest.TestCase):
    def test_type(self):
        """Check variable type.
        ndim=1 is boundary test
        ndim=10 is normal test
        """
        nstep = 100
        nwalker = 10
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = np.random.rand(nstep, nwalker, ndim)
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertIsInstance(res, da.Array)

    def test_shape(self):
        """Check result shape
        :return:
        """
        nstep = 100
        nwalker = 10
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = np.random.rand(nstep, nwalker, ndim)
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))

    def test_zero_length(self):
        nstep = 0
        nwalker = 10
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = np.random.rand(nstep, nwalker, ndim)
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))
                self.assertIsInstance(res, da.Array)
                with self.assertRaises(Exception):
                    res.compute()


class TestDask2DaskDim1(unittest.TestCase):
    def test_type(self):
        """Check variable type.
        nstep=1 is boundary test
        nstep=10 is normal test
        """
        nsteps = (1, 100)
        for nstep in nsteps:
            with self.subTest(nstep=nsteps):
                da.random.random(nstep)
                array = da.random.random(nstep)
                inst = bay_acor.GPStrategy(dask=True)
                res = inst.compute(array)
                self.assertIsInstance(res, da.Array)

    def test_shape(self):
        """Check result shape
        :return:
        """
        nsteps = (1, 100)
        for nstep in nsteps:
            with self.subTest(nstep=nsteps):
                array = np.random.rand(nstep)
                inst = bay_acor.GPStrategy(dask=True)
                res = inst.compute(array)
                self.assertEqual(res.shape, ())
                self.assertIsInstance(res, da.Array)

    def test_zero_length(self):
        nstep = 0
        array = np.random.rand(nstep)
        inst = bay_acor.GPStrategy(dask=True)
        res = inst.compute(array)
        self.assertIsInstance(res, da.Array)
        with self.assertRaises(Exception):
            res.compute()
        # self.assertEqual((), res.shape)

        # self.assertIsInstance(res, np.ndarray)
        #     res.compute()


class TestDask2DaskDim2(unittest.TestCase):
    def test_type(self):
        """Check variable type.
        ndim=1 is boundary test
        ndim=10 is normal test
        """
        nstep = 100
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = da.random.random((nstep, ndim))
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertIsInstance(res, da.Array)

    def test_shape(self):
        """Check result shape
        :return:
        """
        nstep = 100
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = da.random.random((nstep, ndim))
                inst = bay_acor.GPStrategy(dask=True)
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))

    def test_zero_length(self):
        nstep = 0
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = da.random.random((nstep, ndim))
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))
                self.assertIsInstance(res, da.Array)
                with self.assertRaises(Exception):
                    res.compute()


class TestDask2DaskDim3(unittest.TestCase):
    def test_type1(self):
        """Check variable type.
        ndim=1 is boundary test
        ndim=10 is normal test
        """
        nstep = 100
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = da.random.random((nstep, ndim))
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertIsInstance(res, da.Array)

    def test_shape(self):
        """Check result shape
        :return:
        """
        nstep = 100
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = da.random.random((nstep, ndim))
                inst = bay_acor.GPStrategy(dask=True)
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))

    def test_zero_length(self):
        nstep = 0
        ndims = (1, 10)
        for ndim in ndims:
            with self.subTest(ndim=ndims):
                array = da.random.random((nstep, ndim))
                inst = bay_acor.GPStrategy()
                res = inst.compute(array)
                self.assertEqual(res.shape, (ndim,))
                self.assertIsInstance(res, da.Array)
                with self.assertRaises(Exception):
                    res.compute()


class TestInvalidInput(unittest.TestCase):
    def test_ndim0(self):
        ndim = 0
        inst = bay_acor.GPStrategy()
        with self.assertRaises(ValueError):
            inst.compute(np.random.rand(1)[0])

    def test_ndim3(self):
        ndim = 4
        inst = bay_acor.GPStrategy()
        with self.assertRaises(ValueError):
            inst.compute(np.random.rand(1, 2, 3, 1))


if __name__ == "__main__":
    unittest.main()
