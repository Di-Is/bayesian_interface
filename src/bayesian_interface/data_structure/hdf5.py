from typing import Any, Optional
import os
from functools import wraps

import numpy as np
import h5py

from .structure_parts import (
    AttrBase,
    ArrayBase,
    init_check,
    AttrParamBase,
    ArrayParamBase,
)


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


class Attr(AttrBase):
    def __init__(self, name: str, fpath: str, gpath: str) -> None:
        self.name = name
        self.fpath = fpath
        self.gpath = gpath

    @init_check
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


class Array(ArrayBase):
    def __init__(
        self,
        name: str,
        dtype: Any,
        fpath: str,
        gpath: str,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self._maxshape = None
        self.fpath = fpath
        self.gpath = gpath
        self.compression = compression
        self.compression_opts = compression_opts

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

    @init_check
    def get(self, idx=None) -> np.ndarray:
        with h5py.File(self.fpath, "r") as f:
            g = f[self.gpath]
            if idx is None:
                return g[self.name][:]
            else:
                return g[self.name][idx]

    @init_check
    @make_group
    def set(self, value, idx=None) -> None:
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            if idx is None:
                g[self.name][:] = value
            else:
                g[self.name][idx] = value

    def create(self, shape: tuple, maxshape: Optional[tuple] = None) -> None:
        if maxshape is None:
            maxshape = (None,) * len(shape)
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            if self.name in g:
                del g[self.name]
            g.create_dataset(
                self.name,
                shape=shape,
                dtype=self.dtype,
                maxshape=maxshape,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    @init_check
    def resize(self, shape: tuple[int, ...]):
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            g[self.name].resize(shape)

    @init_check
    def append(self, arr: np.ndarray, axis: int) -> None:
        shape_old = self.shape
        shape_new = []
        idx = []
        for i in range(len(shape_old)):
            if i == axis:
                add_elem = shape_old[i] + arr.shape[i]
                start = shape_old[i]
            else:
                add_elem = shape_old[i]
                start = 0
            shape_new.append(add_elem)
            idx.append(slice(start, add_elem))
        shape_new = tuple(shape_new)
        idx = tuple(idx)
        self.resize(shape_new)
        self.set(arr, idx)

    @property
    def maxshape(self) -> Optional[tuple[int]]:
        with h5py.File(self.fpath, "a") as f:
            g = f[self.gpath]
            value = g[self.name].maxshape
        return value


from dataclasses import dataclass


@dataclass
class AttrParam(AttrParamBase):
    gpath: str
    fpath: str


@dataclass
class ArrayParam(ArrayParamBase):
    gpath: str
    fpath: str
    compression: Optional[str] = None
    compression_opts: Optional[int] = None


if __name__ == "__main__":
    attr = Attr("aa", "test.h5", "test")
    #  print(attr.has())
    attr.set("aa®")
    # print(attr.get())
    arr = Array("arr1", int, "test.h5", "test")
    arr.create((10, 256))
    print(arr.get((0, 0)))
    print(arr.maxshape)
    arr.resize((1, 1))
    print(arr.shape)
    arr.append(np.random.randn(*(1, 1)), 1)
    print(arr.shape)
