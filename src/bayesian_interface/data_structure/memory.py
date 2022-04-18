from typing import Any, Optional
from dataclasses import dataclass

import numpy as np

from .structure_parts import (
    AttrBase,
    ArrayBase,
    init_check,
    AttrParamBase,
    ArrayParamBase,
)


class Attr(AttrBase):
    def __init__(self, name: str):
        self.name = name
        self.value = None

    @init_check
    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value

    def has(self) -> bool:
        return self.value is not None


class Array(ArrayBase):
    def __init__(self, name: str, dtype: Any) -> None:
        self.name = name
        self.dtype = dtype
        self.value: Optional[np.ndarray] = None
        self._maxshape = None

    def has(self) -> bool:
        return self.value is not None

    @init_check
    def get(self, idx=None) -> np.ndarray:
        if idx is None:
            return self.value.copy()
        else:
            return self.value.copy()[idx]

    @init_check
    def set(self, value, idx=None) -> None:
        if idx is None:
            self.value[:] = value
        else:
            self.value[idx] = value

    def create(self, shape: tuple, maxshape: Optional[tuple] = None) -> None:
        self._maxshape = maxshape
        self.value = np.empty(shape, self.dtype)

    @init_check
    def resize(self, shape: tuple[int, ...]):
        if self._maxshape is not None:
            if len(self._maxshape) != len(shape) or self._maxshape < shape:
                raise ValueError("Exceed max shape")
        self.value.resize(shape)

    @init_check
    def append(self, arr: np.ndarray, axis: Optional[int] = None) -> None:
        self.value = np.append(self.value, arr, axis)

    @property
    def maxshape(self) -> Optional[tuple[int]]:
        return self._maxshape

    @maxshape.setter
    def maxshape(self, value: tuple[int, ...]) -> None:
        pass


@dataclass
class AttrParam(AttrParamBase):
    pass


@dataclass
class ArrayParam(ArrayParamBase):
    pass


if __name__ == "__main__":
    a = Array("a", float)
    a.create((1, 2, 3), (10, 3, 4, 2))
    a.set(np.zeros((1, 2, 3)))
