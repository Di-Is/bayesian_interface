import os
import unittest

import numpy as np

from bayesian_interface.mcmc.autocorr.autocorrtime import AutoCorrTime
from bayesian_interface.mcmc.autocorr.normal import (
    FFTStrategy,
    GPStrategy,
)
from bayesian_interface.mcmc.autocorr.ensemble import (
    StrategyFlattenCalc,
    StrategyCalcMean,
    StrategyAssignment,
    StrategyMeanCalc,
)


class TestConvergence(unittest.TestCase):
    def _make_correlation_data(self, nstep: int, nwalkers: int, ndim: int):
        import celerite
        from celerite import terms

        kernel = terms.RealTerm(log_a=0.0, log_c=-6.0)
        kernel += terms.RealTerm(log_a=0.0, log_c=-2.0)

        true_tau = sum(2 * np.exp(t.log_a - t.log_c) for t in kernel.terms)
        true_tau /= sum(np.exp(t.log_a) for t in kernel.terms)
        # print("autocorr-time", true_tau)

        # Simulate a set of chains:
        gp = celerite.GP(kernel)
        t = np.arange(nstep)
        gp.compute(t)
        arr = []
        for i in range(ndim):
            arr.append(gp.sample(size=nwalkers).T.reshape(nstep, nwalkers, 1))
        arr = np.concatenate(arr, axis=2)
        return arr

    def test1(self):
        strategy = StrategyFlattenCalc(FFTStrategy())
        autcorr = AutoCorrTime(strategy, None)
        arr = self._make_correlation_data(100, 2, 3)
        vals = autcorr.compute(arr)

    def test2(self):
        strategy = StrategyFlattenCalc(GPStrategy())
        autcorr = AutoCorrTime(strategy, None)
        arr = self._make_correlation_data(100, 2, 3)
        vals = autcorr.compute(arr)

    def test3(self):

        strategy = StrategyCalcMean(FFTStrategy())
        autcorr = AutoCorrTime(strategy, None)

        arr = self._make_correlation_data(100, 2, 3)
        vals = autcorr.compute(arr)

    def test4(self):

        strategy = StrategyCalcMean(GPStrategy())
        autcorr = AutoCorrTime(strategy, None)

        arr = self._make_correlation_data(100, 2, 3)
        vals = autcorr.compute(arr)

    def test5(self):
        strategy = StrategyAssignment(GPStrategy())
        autcorr = AutoCorrTime(strategy, None)

        arr = self._make_correlation_data(100, 2, 3)

        vals = autcorr.compute(arr)

    def test6(self):
        strategy = StrategyAssignment(FFTStrategy())
        autcorr = AutoCorrTime(strategy, None)

        arr = self._make_correlation_data(100, 2, 3)
        with self.assertRaises(ValueError):
            vals = autcorr.compute(arr)

    def test7(self):
        strategy = StrategyMeanCalc(FFTStrategy())
        autcorr = AutoCorrTime(strategy, None)

        arr = self._make_correlation_data(100, 2, 3)

        vals = autcorr.compute(arr)

    def test8(self):
        strategy = StrategyMeanCalc(GPStrategy())
        autcorr = AutoCorrTime(strategy, None)

        arr = self._make_correlation_data(100, 2, 3)

        vals = autcorr.compute(arr)


if __name__ == "__main__":
    unittest.main()
