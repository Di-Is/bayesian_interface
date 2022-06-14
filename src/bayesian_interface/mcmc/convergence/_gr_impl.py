import logging
from typing import Literal, Union
from abc import ABCMeta, abstractmethod
import hashlib

import numpy as np
from numba import njit, prange, objmode


BATCH_MEAN_METHOD = ["bm", "obm"]
BATCH_MEAN_METHOD_TYPE = Literal["bm", "obm"]


@njit(cache=True, fastmath=True, nogil=True)
def get_mcse(
    chain_3d: np.ndarray,
    batch_size: int,
    bm_method: BATCH_MEAN_METHOD_TYPE = "bm",
    lagsail_factor: int = 3,
    trim_method: Literal["start", "end"] = "end",
):
    """
    :param chain_3d: shape: (nprocesses, nsteps, ndim)の三次元のアレイ
    :param bm_method: バッチ平均のアルゴリズム
    :param lagsail_factor: バッチ平均を撮るときのハイパラ
    :param trim_method: チェーンをバッチの数で切り揃える時の方法
    :return: mcse
    """
    #  print(batch_size)
    nprocesses, nsteps, ndim = chain_3d.shape

    batch_size_inv = int(np.floor(batch_size / lagsail_factor))

    nbatch = int(np.floor(nsteps / batch_size))
    nsteps_trimed = int(nbatch * batch_size)

    # バッチの数に合わせてサンプリング結果をトリムする
    if trim_method == "end":
        chain_trimed = chain_3d[:, :nsteps_trimed, :]
    elif trim_method == "start":
        chain_trimed = chain_3d[:, -nsteps_trimed:, :]
    else:
        with objmode:
            logging.warning('trim_methodが不正です, trim_method = "end"として処理します')
        chain_trimed = chain_3d[:, :nsteps_trimed, :]

    # mcseを推定する
    if bm_method == "bm":
        mcse = __get_bm_mcse(chain_trimed, batch_size, batch_size_inv)
    elif bm_method == "obm":
        mcse = __get_obm_mcse(chain_trimed, batch_size, batch_size_inv)
    else:
        with objmode:
            logging.warning('bm_methodが不正です bm_method = "bm"として処理します')
        mcse = __get_bm_mcse(chain_trimed, batch_size, batch_size_inv)

    return mcse


@njit(cache=True, fastmath=True, nogil=True)
def __get_bm_mcse_part(
    chain_3d: np.ndarray, chain_mean_dir_dim: np.ndarray, batch_size: int
) -> np.ndarray:
    """バッチ平均をアルゴリズム'bm'で計算する
    :param chain_3d: shape: (nprocesses, nsteps, ndim)の三次元アレイ
    :param chain_mean_dir_dim: 上のアレイの1,2次元方向に対する平均
    :param batch_size: バッチサイズ
    :return: 計算したmcseの推定量の一部
    """

    nprocesses, nsteps_trimed, ndim = chain_3d.shape
    nbatch = int(np.floor(nsteps_trimed / batch_size))
    # calc batch mean
    chain_bm = np.zeros((nprocesses, nbatch, ndim), dtype=np.float64)
    for i in range(nprocesses):
        for j in range(nbatch):
            for k in range(ndim):
                batch_st_idx = batch_size * j
                batch_end_idx = batch_size * (j + 1)
                chain_bm[i, j, k] = np.mean(chain_3d[i, batch_st_idx:batch_end_idx, k])
    chain_flat = chain_bm.reshape(-1, ndim)
    chain_mean_shift = chain_flat - chain_mean_dir_dim
    mcse_part = (
        chain_mean_shift.T @ chain_mean_shift / (nbatch * nprocesses - 1) * batch_size
    )
    return mcse_part


@njit(cache=True, fastmath=True, nogil=True)
def __get_bm_mcse(
    chain: np.ndarray, batch_size: int, batch_size_inv: int
) -> np.ndarray:
    """
    :param chain:
    :param batch_size:
    :param batch_size_inv:
    :return:
    """

    nprocesses, nsteps_trimed, ndim = chain.shape

    # calc match mean
    chain_mean_dir_dim = np.empty(ndim, dtype=np.float64)
    for dim in range(ndim):
        chain_mean_dir_dim[dim] = np.mean(chain[:, :, dim])

    mcse_part = __get_bm_mcse_part(chain, chain_mean_dir_dim, batch_size)

    # calc match mean (correction part)
    if batch_size_inv > 1:
        mcse_part_inv = __get_bm_mcse_part(chain, chain_mean_dir_dim, batch_size_inv)
    else:
        mcse_part_inv = mcse_part

    mcse = (2.0 * mcse_part - mcse_part_inv) / (nprocesses * nsteps_trimed)
    return mcse


@njit(cache=True, fastmath=True, nogil=True)
def __get_obm_mcse_part(
    chain_3d: np.ndarray, chain_mean_dir_dim: np.ndarray, batch_size: int
) -> np.ndarray:
    """バッチ平均をアルゴリズム'obm'で計算する
    :param chain_3d: shape: (nprocesses, nsteps, ndim)の三次元アレイ
    :param chain_mean_dir_dim: 上のアレイの1,2次元方向に対する平均
    :param batch_size: バッチサイズ
    :return: 計算したmcseの推定量の一部
    """
    nprocesses, nsteps_trimed, ndim = chain_3d.shape
    nbatch = nsteps_trimed - batch_size + 1
    # calc match mean
    chain_bm = np.zeros((nprocesses, nbatch, ndim))
    for i in range(nprocesses):
        for j in range(nbatch):
            for k in range(ndim):
                batch_st_idx = j
                batch_end_idx = j + batch_size
                chain_bm[i, j, k] = np.mean(chain_3d[i, batch_st_idx:batch_end_idx, k])
    chain_flat = chain_bm.reshape(-1, ndim)
    chain_mean_shift = chain_flat - chain_mean_dir_dim
    mcse_part = (
        chain_mean_shift.T @ chain_mean_shift / (nbatch * nprocesses - 1) * batch_size
    )
    return mcse_part


@njit(cache=True, fastmath=True, nogil=True)
def __get_obm_mcse(
    chain: np.ndarray, batch_size: int, batch_size_inv: int
) -> np.ndarray:

    nprocesses, nsteps_trimed, ndim = chain.shape

    # calc match mean
    chain_mean_dir_dim = np.empty(ndim, dtype=np.float64)
    for dim in range(ndim):
        chain_mean_dir_dim[dim] = np.mean(chain[:, :, dim])

    t_b = __get_obm_mcse_part(chain, chain_mean_dir_dim, batch_size)

    if batch_size_inv > 1:
        t_b_r = __get_obm_mcse_part(chain, chain_mean_dir_dim, batch_size_inv)
    else:
        t_b_r = t_b

    mcse = (2.0 * t_b - t_b_r) / (nprocesses * nsteps_trimed)
    return mcse


@njit(cache=True, fastmath=True, nogil=True)
def get_batchsize(
    chain_3d: np.ndarray, bm_method: BATCH_MEAN_METHOD_TYPE = "bm"
) -> int:
    """stableGRで使用するバッチサイズを計算する
    :param chain_3d: MCMCのサンプリング結果, shape: (nprocesses, nsteps, ndim)
    :param bm_method: バッチ平均の手法
    :return: 算出したバッチ平均のバッチサイズ
    """

    nprocesses, nsteps, ndim = chain_3d.shape

    # calc chain length part
    if bm_method == "bm":
        b_npart = nsteps
    else:
        b_npart = 1.5 * nsteps

    b_npart = b_npart ** (1 / 3)

    b_arr = np.empty(nprocesses, dtype=np.int64)

    # calc AR coeff part
    for i in range(nprocesses):
        sigma2 = 0
        gamma2 = 0
        for j in range(ndim):
            chain_1d = chain_3d[i, :, j]
            ar_coeffs, ar_sigma2 = select_order(chain_1d)
            gamma = get_auto_covariance(chain_1d - chain_1d.mean(), len(ar_coeffs))
            fac = 0.0
            for k, coeff in enumerate(ar_coeffs):
                for m in range(k + 1):
                    gamma_ind = abs(m - k)
                    fac += coeff * (m + 1) * gamma[gamma_ind]

            sigma = ar_sigma2 / (1 - np.sum(ar_coeffs)) ** 2

            gamma = (
                2
                * (
                    fac
                    + (sigma - gamma[0])
                    / 2
                    * np.sum((np.arange(len(ar_coeffs)) + 1) * ar_coeffs)
                )
                / (1 - np.sum(ar_coeffs))
            )

            # modified sigma2 and gamma2
            sigma2 += sigma**2 / gamma**2
            gamma2 += gamma**2 / gamma**2

        b_const = (gamma2 / sigma2) ** (1 / 3)
        b_arr[i] = int(b_const * b_npart)
    return int(np.mean(b_arr))


@njit(cache=True, fastmath=False, nogil=True)
def select_order(chain: np.ndarray):
    """AICを利用して、ARモデルの最適なオーダーを選択する。"""

    # 平均を0にシフト
    chain_dm = chain - np.mean(chain)
    ndata = len(chain_dm)
    max_order = int(min(ndata - 1, 10 * np.log10(ndata)))

    aics = np.zeros(max_order, dtype=np.float64)
    ar_params_list = np.full((max_order, max_order), np.nan, dtype=np.float64)
    ar_sigmas = np.zeros(max_order, dtype=np.float64)
    acov = get_auto_covariance(chain_dm, max_order + 1)

    for order_t in range(1, max_order + 1):
        ar_params, sigma2 = levinson_durbin(acov, order=order_t)

        aic_v = ndata * (np.log(2 * np.pi) + np.log(sigma2) + 1) + 2 * (order_t + 1)
        aics[order_t - 1] = aic_v
        ar_params_list[order_t - 1, :order_t] = ar_params
        ar_sigmas[order_t - 1] = sigma2

    aic_min_order = np.argmin(aics) + 1
    ar_coeff_best = ar_params_list[aic_min_order - 1][:aic_min_order]

    ar_simga2_best = ar_sigmas[aic_min_order - 1]
    return ar_coeff_best, ar_simga2_best


@njit(cache=True, fastmath=True, nogil=True)
def get_auto_covariance(u: np.ndarray, max_lag: int) -> np.ndarray:
    """自己共分散を計算する。
    :param u: 自己共分散の計算対象
    :param max_lag: 自己共分散を計算する際のラグ
    :return: 自己共分散の値
    """
    c = np.zeros(max_lag, dtype=np.float64)
    num_tot = u.size
    for lag in range(max_lag):
        for i in range(num_tot - lag):
            c[lag] += u[i] * u[lag + i]
        c[lag] /= num_tot
    return c


@njit(cache=True, fastmath=True, nogil=True)
def levinson_durbin(r, order=None):
    """levinson durbinのアルゴリズムを利用して、yule-walker方程式を解く
    :Note:
    spectrum.levinson.LEVINSONをベースに一部追加修正
    ARの係数をマイナスに変更してあるのと、引数を複素数->実数に限定、必要のない戻り値は削除
    """

    t0 = r[0]
    t = r[1:]
    m = len(t)

    if order is None:
        m = len(t)
    else:
        assert order <= m, "order must be less than size of the input data"
        m = order

    a = np.zeros(m, dtype=np.float64)

    p = t0

    for k in range(0, m):
        save = t[k]
        if k == 0:
            temp = -save / p
        else:
            # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + a[j] * t[k - j - 1]
            temp = -save / p

        p = p * (1.0 - temp**2.0)

        if p <= 0:
            raise ValueError("singular matrix")
        a[k] = temp
        if k == 0:
            continue

        khalf = int((k + 1) / 2)
        for j in range(0, khalf):
            kj = k - j - 1
            save = a[j]
            a[j] = save + temp * a[kj]
            if j != kj:
                a[kj] += temp * save
    return -a, p


@njit(cache=True, fastmath=True, nogil=True)
def calc_psrf(chain_3d: np.ndarray) -> tuple[np.ndarray, float, float]:
    """PSRFを計算する
    :param chain_3d:
    :return:
    :Note:
    numbaを利用した高速化のためメソッドではなく関数として定義
    """

    nprocesses, nsteps, ndim = chain_3d.shape

    # chain mean
    mean_2d = np.zeros((nprocesses, ndim), dtype=np.float64)
    mean_1d = np.zeros(ndim, dtype=np.float64)
    for i in range(nprocesses):
        for j in range(nsteps):
            for k in range(ndim):
                mean_2d[i, k] += chain_3d[i, j, k]
                mean_1d[k] += chain_3d[i, j, k]
    mean_2d /= nsteps
    mean_1d /= nprocesses * nsteps

    # between chain variance
    b_var = mean_2d - mean_1d
    b_var = b_var.reshape(-1, ndim)
    b_var = b_var.T @ b_var
    b_var /= nprocesses - 1
    b_var *= nsteps

    # within chain variance
    w_var = chain_3d - mean_2d.reshape(nprocesses, 1, ndim)
    w_var = w_var.reshape(-1, ndim)
    w_var = w_var.T @ w_var
    w_var /= nprocesses - 1
    w_var /= nsteps

    tmp = np.linalg.solve(w_var.astype(np.complex128), b_var.astype(np.complex128))
    eigen, _ = np.linalg.eig(tmp)

    det = np.abs(eigen).prod() ** (1 / ndim)
    max_eigenvalue = max(np.abs(eigen))
    fac1 = (nsteps - 1) / nsteps
    fac2 = 1 / nsteps

    var = np.diag(w_var) * fac1 + fac2 * np.diag(b_var)
    psrf = np.sqrt(var / np.diag(w_var))
    mpsrf_det = np.sqrt(fac1 + fac2 * det)
    mpsrf_max = np.sqrt(fac1 + fac2 * max_eigenvalue)
    return psrf, mpsrf_det, mpsrf_max


@njit(cache=True, fastmath=True, nogil=True)
def calc_stable_psrf(
    chain_3d: np.ndarray,
    bm_method: BATCH_MEAN_METHOD_TYPE = "bm",
    lagsail_factor: int = 3,
) -> tuple[np.ndarray, float, float]:
    """
    :param chain_3d: shape: (nprocrsses, nsteps, ndim)の2次元化したサンプリング結果
    :param bm_method: バッチ平均の手法
    :param lagsail_factor: バッチ平均のハイパラ
    :return: 計算したpsrf, mpsrf二種から成るタプル
    """
    nprocesses, nsteps, ndim = chain_3d.shape

    batch_size = get_batchsize(chain_3d, bm_method)

    # with-in chain variance matrix (unbiased)
    w_arr = np.zeros((ndim, ndim), dtype=np.float64)
    for chain_met_pro in chain_3d:
        w_arr += np.cov(chain_met_pro, rowvar=False)
    w_arr /= nprocesses

    mcse = (
        get_mcse(chain_3d, batch_size, bm_method, lagsail_factor) * nprocesses * nsteps
    )

    sigma2 = (nsteps - 1) / nsteps * np.diag(w_arr) + np.diag(mcse) / nsteps

    tmp = np.linalg.solve(w_arr.astype(np.complex128), mcse.astype(np.complex128))
    eigenvalue, _ = np.linalg.eig(tmp)
    eigenvalue_max = max(np.abs(eigenvalue))
    det = np.abs(eigenvalue).prod() ** (1 / ndim)

    fac1 = (nsteps - 1) / nsteps
    fac2 = 1 / nsteps

    psrf = np.sqrt(sigma2 / np.diag(w_arr))
    mpsrf_determinant = np.sqrt(fac1 + fac2 * det)
    mpsrf_max_eigenvalue = np.sqrt(fac1 + fac2 * eigenvalue_max)

    return psrf, mpsrf_determinant, mpsrf_max_eigenvalue
