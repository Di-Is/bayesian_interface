import time

import numpy as np


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def calc_acf(array: np.ndarray, norm: bool = True):

    n = next_pow_two(len(array))
    f = np.fft.fft(array - np.mean(array), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(array)].real
    if norm:
        acf /= acf[0]
    return acf


from scipy.fft import fft, ifft


def calc_acf(array, norm=True):
    n = next_pow_two(len(array))
    f = fft(array - np.mean(array), n=2 * n)
    acf = ifft(f * np.conjugate(f))[: len(array)].real
    acf /= 4 * n
    # Normalize
    if norm:
        acf /= acf[0]
    return acf
