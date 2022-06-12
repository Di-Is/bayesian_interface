from typing import Any, Optional

import numpy as np

import bayesian_interface.data_structure.structure_parts as parts


class Attr(parts.AbsAttr):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self._kwargs = kwargs
        self.value = None

    @parts.init_check
    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value

    def has(self) -> bool:
        return self.value is not None


class Array(parts.AbsArray):
    def __init__(self, name: str, gpath=None, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs
        self.value: Optional[np.ndarray] = None

    def has(self) -> bool:
        return self.value is not None

    @parts.init_check
    def get(self, idx=None, copy: bool = True) -> np.ndarray:
        if idx is None:
            if copy:
                return self.value.copy()
            else:
                self.value.flags.writeable = False
                return self.value
        else:
            if copy:
                return self.value.copy()[idx]
            else:
                self.value.flags.writeable = False
                return self.value[idx]

    @parts.init_check
    def set(self, value, idx=None) -> None:
        if idx is None:
            self.value[:] = value
        else:
            self.value[idx] = value

    def create(
        self,
        shape: tuple[int, ...],
        maxshape: Optional[tuple[int | slice | None, ...]],
        dtype: Any,
        **kwargs
    ) -> None:
        self.value = np.empty(shape, dtype=dtype, **kwargs | self.kwargs)

    @parts.init_check
    def resize(self, shape: tuple[int, ...]):
        value_new = np.empty(shape, dtype=self.value.dtype)
        idx = tuple(slice(i) for i in self.shape)
        value_new[idx] = self.value[:]
        self.value = value_new

    #  self.value = np.resize(self.value, shape)

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def __getitem__(self, item):
        return self.value[item]


class AttrFactory(parts.AttrFactory):
    @classmethod
    def get_dataclass(cls) -> Attr.__class__:
        return Attr


class ArrayFactory(parts.AttrFactory):
    @classmethod
    def get_dataclass(cls) -> Array.__class__:
        return Array
