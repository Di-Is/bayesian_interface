import os

import numpy as np
import matplotlib.pyplot as plt
import tqdm

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


class Test:
    def __init__(self):
        pass

    def make_dat(self, nsteps: int, nwalkers: int, ndim: int) -> np.ndarray:
        import celerite
        from celerite import terms

        kernel = terms.RealTerm(log_a=0.0, log_c=-6.7)
        kernel += terms.RealTerm(log_a=0.0, log_c=-2.0)

        true_tau = sum(2 * np.exp(t.log_a - t.log_c) for t in kernel.terms)
        true_tau /= sum(np.exp(t.log_a) for t in kernel.terms)
        print("autocorr-time", true_tau)

        # Simulate a set of chains:
        gp = celerite.GP(kernel)
        t = np.arange(nsteps)
        gp.compute(t)
        arr = []
        for i in range(ndim):
            arr.append(gp.sample(size=nwalkers).T.reshape(nsteps, nwalkers, 1))
        arr = np.concatenate(arr, axis=2)
        return arr, true_tau

    def flat_fft(self, arr):
        autocorr = AutoCorrTime(strategy=StrategyFlattenCalc(FFTStrategy()))
        return autocorr.compute(arr)

    def flat_gp(self, arr):
        autocorr = AutoCorrTime(strategy=StrategyFlattenCalc(GPStrategy()))
        return autocorr.compute(arr)

    def calc_mean_fft(self, arr):
        autocorr = AutoCorrTime(strategy=StrategyCalcMean(FFTStrategy()))
        return autocorr.compute(arr)

    def calc_mean_gp(self, arr):
        autocorr = AutoCorrTime(strategy=StrategyCalcMean(GPStrategy()))
        return autocorr.compute(arr)

    def assi_gp(self, arr):
        autocorr = AutoCorrTime(strategy=StrategyAssignment(GPStrategy()))
        return autocorr.compute(arr)

    def mean_calc_fft(self, arr):
        autocorr = AutoCorrTime(strategy=StrategyMeanCalc(FFTStrategy()))
        return autocorr.compute(arr)

    def mean_calc_gp(self, arr):
        autocorr = AutoCorrTime(strategy=StrategyMeanCalc(GPStrategy()))
        return autocorr.compute(arr)

    def main(self):

        nsteps = 500000
        nwalkers = 20
        ndim = 5

        arr, true_tau = self.make_dat(nsteps, nwalkers, ndim)
        lst = range(100, nsteps, 4000)
        dct = {
            #   "flat_fft": self.flat_fft,
            # "flat_gp": self.flat_gp,
            "calc_mean_fft": self.calc_mean_fft,
            #   "calc_mean_gp": self.calc_mean_gp,
            # "assi_gp": self.assi_gp,
            #    "mean_calc_fft": self.mean_calc_fft,
            #  "mean_calc_gp": self.calc_mean_gp,
        }
        for name, method in dct.items():
            vals = []
            for i in tqdm.tqdm(lst):
                vals.append(method(arr[:i]).iats.get()[-1])
            plt.errorbar(
                lst,
                np.mean(vals, axis=1),
                yerr=np.std(vals, axis=1),
                label=name,
            )
        plt.legend()
        plt.axhline(true_tau)
        plt.show()


if __name__ == "__main__":
    Test().main()
