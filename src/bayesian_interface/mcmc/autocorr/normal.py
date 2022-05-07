import time

import numpy as np
import dask.array as da
from dask import config

from .autocorr import calc_acf
from .misc import check_dimension
from .autocorrtime import AbsStrategy


class StrategyBase(AbsStrategy):
    def compute(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def method_name(self) -> str:
        raise NotImplementedError


class FFTStrategy(StrategyBase):
    def __init__(self):
        self._dask = False

    def compute(self, array: np.ndarray | da.Array) -> float:
        expected_dims = 2
        check_dimension(array, expected_dims)

        if self._dask or isinstance(array, da.Array):
            return self.comp_dask(array)
        else:
            return self.comp_normal(array)

    def comp_dask(self, array):
        if not isinstance(array, da.Array):
            darr = da.from_array(array, chunks=(array.shape[0], None))
        else:
            darr = array

        res = darr.map_blocks(
            self._calc, drop_axis=[0], dtype=array.dtype, meta=np.array([])
        )

        res = res.compute()
        return res

    def comp_normal(self, array):
        return self._calc(array)

    import psutil

    @staticmethod
    def _calc(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            result = []
            for i in range(array.shape[-1]):
                acf = calc_acf(array[..., i])
                result.append(estimate_iat(acf))

        elif array.ndim == 1:
            acf = calc_acf(array)
            result = [estimate_iat(acf)]
        else:
            raise ValueError
        return np.asarray(result)

    @property
    def method_name(self) -> str:
        name = f"{'FFT'}"
        return name


class GPStrategy(StrategyBase):
    def compute(self, array: np.ndarray | da.Array) -> np.ndarray:
        expected_dims = (2, 3)
        check_dimension(array, expected_dims)

        # check drop axis and chunksize
        if array.ndim == 2:
            drop_axis = (0,)
            chunks = (array.shape[0], 1)
        else:
            drop_axis = (0, 1)
            chunks = (array.shape[0], array.shape[1], 1)

        if not isinstance(array, da.Array):
            darr = da.from_array(array, chunks=chunks)
        else:
            darr = array

        # compute
        res = darr.map_blocks(
            self._calc, drop_axis=drop_axis, dtype=array.dtype, meta=np.array([])
        )
        return res.compute()

    @staticmethod
    def _calc(array: np.ndarray) -> np.ndarray:

        if array.ndim == 3:
            result = []
            for i in range(array.shape[-1]):
                tau_init = estimate_iat(calc_acf(array[..., 0, i]))
                result.append([calc_iat_gp(array[..., :, i], tau_init=tau_init)])
        elif array.ndim == 2:
            result = []
            for i in range(array.shape[-1]):
                tau_init = estimate_iat(calc_acf(array[..., i]))
                result.append(calc_iat_gp(array[..., i], tau_init=tau_init))
        elif array.ndim == 1:
            tau_init = estimate_iat(calc_acf(array))
            result = [calc_iat_gp(array, tau_init=tau_init)]
        else:
            raise ValueError(f"{array.shape=}")

        return np.asarray(result)

    @property
    def method_name(self) -> str:
        name = f"{'GP'}"
        return name


def auto_window(taus: np.ndarray, c: int):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def estimate_iat(acf: np.ndarray, c: int = 5) -> float:
    iats = 2.0 * np.cumsum(acf) - 1.0
    window = auto_window(iats, c)
    iat_estimation = iats[window]
    return iat_estimation


def lnprob_1d2d(p: np.ndarray, gp, arr: np.ndarray) -> tuple[float, np.ndarray]:
    gp.set_parameter_vector(p)
    if arr.ndim == 1:
        v, g = gp.grad_log_likelihood(arr, quiet=True)
    elif arr.ndim == 2:
        v, g = [], []
        for i in range(arr.shape[-1]):
            v_, g_ = gp.grad_log_likelihood(arr[..., i], quiet=True)
            v.append(v_)
            g.append(g_)
        v, g = np.asarray(v), np.asarray(g)
        v = np.sum(v)
        g = np.sum(g, axis=0)
    else:
        raise ValueError
    return -v, -g


def lnprob_1d(p: np.ndarray, gp, arr: np.ndarray) -> tuple[float, np.ndarray]:
    gp.set_parameter_vector(p)
    v, g = gp.grad_log_likelihood(arr, quiet=True)
    return -v, -g


def calc_iat_gp(
    array: np.ndarray, thin: int = 1, tau_init: float | int = 1.0
) -> np.ndarray:
    from scipy.optimize import minimize
    from celerite import terms, GP

    # Compute the initial estimate of tau using the standard method
    arr_thin = array[::thin]

    # Build the GP model
    tau = max(1.0, tau_init / thin)

    # make kernel
    bounds = dict(log_a=(-15.0, 15.0), log_c=(-np.log(len(arr_thin)), 0.0))
    init_vals = [
        dict(
            log_a=np.log(0.9 * np.var(arr_thin)),
            log_c=min(-np.log(tau), bounds["log_c"][1]),
        ),
        dict(
            log_a=np.log(0.1 * np.var(arr_thin)),
            log_c=min(-np.log(0.5 * tau), bounds["log_c"][1]),
        ),
    ]
    kernel = terms.Term()
    for init in init_vals:
        kernel += terms.RealTerm(**init, bounds=bounds)

    gp = GP(kernel, mean=np.mean(arr_thin))
    gp.compute(np.arange(len(arr_thin)))

    # Optimize the model
    res = minimize(
        lnprob_1d2d,
        gp.get_parameter_vector(),
        jac=True,
        bounds=gp.get_parameter_bounds(),
        args=(gp, arr_thin),
    )
    gp.set_parameter_vector(res.x)

    # Compute the maximum likelihood tau
    a, c = kernel.coefficients[:2]
    tau = thin * 2 * np.sum(a / c) / np.sum(a)

    return tau


"""
100%|██████████| 6/6 [00:03<00:00,  1.53it/s]
100%|██████████| 6/6 [03:47<00:00, 37.86s/it]
100%|██████████| 6/6 [00:52<00:00,  8.71s/it]
100%|██████████| 6/6 [01:46<00:00, 17.73s/it]
100%|██████████| 6/6 [03:32<00:00, 35.43s/it]
"""
