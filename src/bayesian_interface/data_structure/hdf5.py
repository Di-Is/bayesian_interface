from typing import Any, Optional
import os
from functools import wraps

import numpy as np
import h5py

import bayesian_interface.data_structure.structure_parts as parts


def make_group(func):
    """hdf5の書き出し先をチェック、グループの存在をチェックするデコレータ
    :param func: 関数
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.path.exists(args[0].fpath):
            if not args[0].gpath in h5py.File(args[0].fpath, "r"):
                create_group(args[0].fpath, args[0].gpath)
        else:
            create_group(args[0].fpath, args[0].gpath)
        return func(*args, **kwargs)

    return wrapper


def create_group(fpath: str, gpath: str) -> None:
    with h5py.File(fpath, "a") as f:
        f.create_group(gpath)


class Attr(parts.AbsAttr):
    def __init__(self, name: str, fpath: str, gpath: str, **kwargs) -> None:
        self.name = name
        self.fpath = fpath
        self.gpath = gpath
        self.kwargs = kwargs

    @parts.init_check
    def get(self) -> Any:
        with h5py.File(self.fpath, "r") as f:
            g = f[self.gpath]
            return g.attrs[self.name]

    @make_group
    def set(self, value: Any) -> None:
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            g.attrs[self.name] = value

    def has(self) -> bool:
        if not os.path.exists(self.fpath):
            return False
        return self._has_group()

    def _has_group(self) -> bool:
        return self.gpath in h5py.File(self.fpath, "r")


class Array(parts.AbsArray):
    def __init__(self, name: str, fpath: str, gpath: str, **kwargs) -> None:
        self.name = name
        self.fpath = fpath
        self.gpath = gpath
        self.kwargs = kwargs

    def has(self) -> bool:
        if not os.path.exists(self.fpath):
            return False
        if not self._has_group():
            return False
        return self._has_arr()

    def _has_group(self) -> bool:
        return self.gpath in h5py.File(self.fpath, "r")

    def _has_arr(self) -> bool:
        return self.name in h5py.File(self.fpath, "r")[self.gpath]

    @parts.init_check
    def get(self, idx=None, copy: bool = True) -> np.ndarray:
        if copy:
            with h5py.File(self.fpath, "r") as f:
                g = f[self.gpath]
                if idx is None:
                    return g[self.name][:]
                else:
                    return g[self.name][idx]
        else:
            f = h5py.File(self.fpath, "r")
            g = f[self.gpath]
            if idx is None:
                return g[self.name][:]
            else:
                return g[self.name][idx]

    @parts.init_check
    @make_group
    def set(self, value, idx=None) -> None:
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            if idx is None:
                g[self.name][:] = value
            else:
                g[self.name][idx] = value

    @make_group
    def create(
        self,
        shape: tuple[int, ...],
        maxshape: Optional[tuple[int | slice | None, ...]],
        dtype: Any,
        **kwargs
    ) -> None:
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            if self.name in g:
                del g[self.name]
            g.create_dataset(
                self.name,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                **(kwargs | self.kwargs),
            )

    @parts.init_check
    def resize(self, shape: tuple[int, ...]):
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            g[self.name].resize(shape)

    @property
    @parts.init_check
    def ndim(self) -> int:
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            return g[self.name].ndim

    @property
    @parts.init_check
    def shape(self) -> tuple[int, ...]:
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            return g[self.name].shape

    def __getitem__(self, item):
        with h5py.File(self.fpath, "r") as f:
            g = f[self.gpath]
            return g[self.name][item]


class AttrFactory(parts.AttrFactory):
    @classmethod
    def get_dataclass(cls) -> Attr.__class__:
        return Attr


class ArrayFactory(parts.AttrFactory):
    @classmethod
    def get_dataclass(cls) -> Array.__class__:
        return Array


#
#
# class WrapperArray:
#     def __init__(self, array, f):
#         self._array = array
#         self._f = f
#
#     def __array__(self, *args, **kwargs):
#         return self._array.__array__(*args, **kwargs)
#
#     def __getitem__(self, item):
#         return self._array.__getitem__(item)
#
#     def __del__(self):
#         self._f.close()
#         del self._f
#         del self._array
#
#     def close(self):
#         self._f.close()
