import os
import unittest

import numpy as np
import xarray as xr

from bayesian_interface.data_structure.netcdf import Attr, Array


class TestAttr(unittest.TestCase):
    def setUp(self):
        self.name = "aaa"
        self.fpath = "test.nc"
        self.gpath = "mcmc"

    def tearDown(self) -> None:
        if os.path.exists(self.fpath):
            os.remove(self.fpath)
        del self.fpath

    def test_attr_set_get(self):
        value = "aa"
        a = Attr(self.name, self.fpath, self.gpath)
        a.set(value)
        self.assertEqual(a.get(), value)

    def test_has(self):
        value = "aa"
        a = Attr(self.name, self.fpath, self.gpath)
        self.assertFalse(a.has())
        a.set(value)
        self.assertTrue(a.has())


class TestArray(unittest.TestCase):
    def setUp(self):
        self.name = "aaa"
        self.fpath = "test.nc"
        self.gpath = "mcmc"

    def tearDown(self) -> None:
        if os.path.exists(self.fpath):
            os.remove(self.fpath)

    def test_create1(self):
        shape = (10, 10)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        arr.create(shape, dtype=np.float64)
        self.assertEqual(arr.shape, shape)

    def test_create2(self):
        """multiple create 1
        :return:
        """
        shape = (10, 10)
        arr1 = Array("arr", self.fpath, self.gpath, dimensions=("draw1", "dim1"))
        arr1.create(shape, dtype=np.float64)

        arr2 = Array("arr", self.fpath, self.gpath, dimensions=("draw2", "dim2"))
        arr2.create(shape, dtype=np.float64)

    # self.assertEqual(arr.shape, shape)

    def test_set(self):
        shape = (10, 10)
        vals = np.random.rand(*shape)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        arr.create(shape, dtype=np.float64)
        arr.set(vals)
        ds = xr.load_dataset(self.fpath, group=self.gpath)
        self.assertTrue(np.allclose(ds.arr[:], vals))

    def test_get(self):
        shape = (10, 10)
        vals = np.random.rand(*shape)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        arr.create(shape, dtype=np.float64)
        arr.set(vals)
        self.assertTrue(np.allclose(arr.get(), vals))

    def test_resize(self):
        shape = (10, 10)
        reshape = (20, 10)
        # vals = np.random.rand(*shape)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        arr.create(shape, dtype=np.float64, maxshape=(None, 10))
        arr.resize(reshape)
        self.assertEqual(arr.shape, reshape)

    def test_append(self):
        shape = (10, 10)
        maxshape = (None, 10)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        arr.create(shape, dtype=np.float64, maxshape=maxshape)
        arr.append(np.zeros(shape), axis=0)
        shape_new = (20, 10)
        self.assertEqual(arr.shape, shape_new)

    def test_has(self):
        shape = (10, 10)
        maxshape = (None, 10)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        self.assertFalse(arr.has())
        arr.create(shape, dtype=np.float64, maxshape=maxshape)
        self.assertTrue(arr.has())

    def test_ndim(self):

        shape = (10, 10)
        maxshape = (None, 10)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        arr.create(shape, dtype=np.float64, maxshape=maxshape)
        self.assertTrue(arr.ndim, 2)

    def test_shape(self):

        shape = (10, 10)
        maxshape = (None, 10)
        arr = Array("arr", self.fpath, self.gpath, dimensions=("draw", "dim"))
        arr.create(shape, dtype=np.float64, maxshape=maxshape)
        self.assertTrue(arr.shape, shape)
