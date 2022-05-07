import typing
import os
from functools import wraps

import numpy as np
import h5netcdf

import bayesian_interface.data_structure.structure_parts as parts


def make_group(func):
    """xarrayの書き出し先をチェック、グループの存在をチェックするデコレータ
    :param func: 関数
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.path.exists(args[0].fpath):
            if not args[0].gpath in h5netcdf.File(args[0].fpath, "r"):
                create_group(args[0].fpath, args[0].gpath)
        else:
            create_group(args[0].fpath, args[0].gpath)
        return func(*args, **kwargs)

    return wrapper


def create_group(fpath: str, gpath: str) -> None:
    with h5netcdf.File(fpath, "a") as f:
        f.create_group(gpath)


class Attr(parts.AbsAttr):
    def __init__(self, name: str, fpath: str, gpath: str) -> None:
        self.name = name
        self.fpath = fpath
        self.gpath = gpath

    @parts.init_check
    def get(self) -> typing.Any:
        with h5netcdf.File(self.fpath, "r") as f:
            g = f[self.gpath]
            return g.attrs[self.name]

    @make_group
    def set(self, value: typing.Any) -> None:
        with h5netcdf.File(self.fpath, "a") as f:
            g = f[self.gpath]
            g.attrs[self.name] = value

    def has(self) -> bool:
        if not os.path.exists(self.fpath):
            return False
        if not self._has_group():
            return False
        return self._has_attr()

    def _has_group(self) -> bool:
        return self.gpath in h5netcdf.File(self.fpath, "r")

    def _has_attr(self) -> bool:
        with h5netcdf.File(self.fpath, "r") as f:
            flag = self.name in f[self.gpath].attrs
        return flag


class Array(parts.AbsArray):
    def __init__(
        self, name: str, fpath: str, gpath: str, dimensions: tuple, **kwargs
    ) -> None:
        self.name = name
        self.fpath = fpath
        self.gpath = gpath
        self.dimensions = dimensions
        self.kwargs = kwargs

    def has(self) -> bool:
        if not os.path.exists(self.fpath):
            return False
        if not self._has_group():
            return False
        return self._has_arr()

    def _has_group(self) -> bool:
        return self.gpath in h5netcdf.File(self.fpath, "r")

    def _has_arr(self) -> bool:
        with h5netcdf.File(self.fpath, "r") as f:
            flag = self.name in f[self.gpath]
        return flag

    @parts.init_check
    def get(self, idx=None, copy: bool = True) -> np.ndarray:
        with h5netcdf.File(self.fpath, "r") as f:
            g = f[self.gpath]
            if idx is None:
                if copy:
                    return g[self.name][:]
                else:
                    return g[self.name]
            else:
                return g[self.name][idx]

    @parts.init_check
    @make_group
    def set(self, value, idx=None) -> None:
        with h5netcdf.File(self.fpath, "a") as f:
            g = f[self.gpath]
            if idx is None:
                g[self.name][:] = value
            else:
                g[self.name][idx] = value

    @make_group
    def create(
        self,
        shape: int | tuple[int, ...],
        maxshape: typing.Optional[tuple[int | slice | None, ...]] = None,
        dtype: typing.Any = None,
        **kwargs
    ) -> None:
        if maxshape is None:
            maxshape = len(shape) * (None,)

        with h5netcdf.File(self.fpath, "a") as f:
            g = f[self.gpath]
            if self.name in g:
                raise ValueError("Not delete")

            # create netcdf dimension
            for i, dim_name in enumerate(self.dimensions):
                if maxshape[i] is None:
                    g.dimensions[dim_name] = None
                else:
                    g.dimensions[dim_name] = shape[i]
            g.create_variable(
                self.name, dtype=dtype, dimensions=self.dimensions, **kwargs
            )
        self.resize(shape)

    @parts.init_check
    def resize(self, shape: tuple[int, ...]):
        with h5netcdf.File(self.fpath, "a") as f:
            g = f[self.gpath]
            for dim_name, size in zip(self.dimensions, shape):
                if len(g.dimensions[dim_name]) != size:
                    g.resize_dimension(dim_name, size)

    @property
    @parts.init_check
    def ndim(self) -> int:
        with h5netcdf.File(self.fpath, "r") as f:
            g = f[self.gpath]
            return g[self.name].ndim

    @property
    @parts.init_check
    def shape(self) -> tuple[int, ...]:
        with h5netcdf.File(self.fpath, "r") as f:
            g = f[self.gpath]
            return g[self.name].shape


class AttrFactory(parts.AttrFactory):
    @classmethod
    def get_dataclass(cls) -> Attr.__class__:
        return Attr


class ArrayFactory(parts.AttrFactory):
    @classmethod
    def get_dataclass(cls) -> Array.__class__:
        return Array
