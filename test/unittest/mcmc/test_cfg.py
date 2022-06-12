import os
import unittest

import numpy as np

# import bayesian_interface.mcmc.cfg as bay_cfg
import bayesian_interface.data_structure as bay_data
import bayesian_interface.mcmc.mcmc as bay_mcmc


class TestSamplingResult(unittest.TestCase):
    attr = ("chain_id", "sampler_name")
    array = ("chain",)

    def setUp(self) -> None:
        self._fpath = "./mcmc.h5"

    def tearDown(self) -> None:
        if os.path.exists(self._fpath):
            os.remove(self._fpath)

    def test_memory(self):
        names = self.attr + self.array
        res = bay_mcmc.mcmc.SamplingResultFactory.create(bay_data.SaveType.memory)
        bay_mcmc.SamplingResultFactory.create(bay_data.SaveType.memory)

    def test_hdf5(self):
        names = self.attr + self.array
        res = bay_mcmc.mcmc.SamplingResultFactory.create(
            bay_data.SaveType.hdf5, fpath=self._fpath
        )
        data = bay_mcmc.SamplingResultFactory.create(
            bay_data.SaveType.hdf5, fpath=self._fpath
        )
        shape = (0, 14)
        data.chain.merge_param(shape=shape, maxshape=(None, 14), dtype=float)
        self.assertEqual(data.chain.shape, shape)
