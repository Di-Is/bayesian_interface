import typing

import numpy as np
import dask.array as da
from dask.delayed import Delayed, delayed

from .autocorr import calc_acf
from .misc import check_dimension
from .autocorrtime import AbsStrategy

# TODO: Add docs
# TODO: Separate inmpl funcs


class StrategyBase(AbsStrategy):
    def compute(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def method_name(self) -> str:
        raise NotImplementedError

    @property
    def need_chain(self) -> bool:
        raise NotImplementedError

    @property
    def need_dim(self) -> bool:
        raise NotImplementedError

    @property
    def drop_chain(self) -> bool:
        raise NotImplementedError

    @property
    def drop_dim(self) -> bool:
        raise NotImplementedError


class FFTStrategy(StrategyBase):
    @classmethod
    @property
    def expected_dim(cls) -> int:  # noqa
        return 1

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray | da.Array:
        match array:
            case np.ndarray():
                result = self._calc_iat(array)
            case da.Array() | Delayed():
                result = da.map_blocks(self._calc_iat, array)
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @staticmethod
    def _calc_iat(array: np.ndarray) -> float:
        """calculate integrated auto-correlation time
        :param array: input array
        :return: iat value
        """
        acf = calc_acf(array)
        result = estimate_iat(acf)
        return result

    @classmethod
    @property
    def method_name(cls) -> str:  # noqa
        name = f"FFT"
        return name


class GPStrategy(StrategyBase):
    def __init__(self):
        self.__tau_old = 1.0

    @property
    def need_chain(self) -> bool:
        return False

    @property
    def need_dim(self) -> bool:
        return False

    @property
    def drop_chain(self) -> bool:
        return False

    @property
    def drop_dim(self) -> bool:
        return False

    @property
    def expected_dim(self) -> tuple[int, ...]:
        return 1, 2

    def compute(self, array: np.ndarray | da.Array) -> np.ndarray | da.Array:
        check_dimension(array, self.expected_dim)
        match array:
            case np.ndarray():
                result = self._calc_iat(array)
            case da.Array() | Delayed():
                result = da.map_blocks(self._calc_iat, array)
            case _:
                raise TypeError(f"input type {type(array)} is invalid.")
        return result

    @staticmethod
    def _calc_iat(array: np.ndarray, tau_init: float = 1.0) -> np.ndarray:
        if array.ndim == 1:
            tau_init = estimate_iat(calc_acf(array))
            result = [calc_iat_gp(array, tau_init=tau_init)]
        else:
            tau_init = estimate_iat(calc_acf(array[..., 0]))
            result = calc_iat_gp(array[..., 0], tau_init=tau_init)
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


NORMAL_IAT_METHODS = (FFTStrategy.method_name, GPStrategy.method_name)
