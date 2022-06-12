import unittest
from bayesian_interface.data_structure.builder import SaveType
from bayesian_interface.data_structure.chain import (
    SamplingResultFactory,
)


class TestDirector(unittest.TestCase):
    def test_1(self):

        samp_res = SamplingResultFactory.merge_param()

        self.assertFalse(samp_res.chain.has())
        self.assertFalse(samp_res.lnprob.has())
        self.assertFalse(samp_res.chain_id.has())
        self.assertFalse(samp_res.sampler_name.has())

    def test_2(self):
        kwargs = dict(fpath="", gpath="")
        samp_res = SamplingResultFactory.merge_param(
            SaveType.hdf5,
            attr_kwargs={"chain_id": kwargs, "sampler_name": kwargs},
            array_kwargs={"chain": kwargs, "lnprob": kwargs},
        )
        self.assertFalse(samp_res.chain.has())
        self.assertFalse(samp_res.lnprob.has())
        self.assertFalse(samp_res.chain_id.has())
        self.assertFalse(samp_res.sampler_name.has())

    def test_3(self):
        kwargs = dict(fpath="", gpath="")

        samp_res = SamplingResultFactory.merge_param(
            SaveType.netcdf,
            attr_kwargs={"chain_id": kwargs, "sampler_name": kwargs},
            array_kwargs={"chain": kwargs, "lnprob": kwargs},
        )
        self.assertFalse(samp_res.chain.has())
        self.assertFalse(samp_res.lnprob.has())
        self.assertFalse(samp_res.chain_id.has())
        self.assertFalse(samp_res.sampler_name.has())
