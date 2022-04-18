from typing import Any, Optional
from functools import wraps
from dataclasses import dataclass

import numpy as np


class AttrBase:
    """Attributeのデータ構造の基底クラス"""

    def set(self, value: Any) -> None:
        raise NotImplementedError

    def get(self) -> Any:
        raise NotImplementedError

    def has(self) -> bool:
        raise NotImplementedError


types = int | float | str | complex | np.ndarray
index_type = int | slice | tuple[int | slice, ...]


class ArrayBase:
    def set(self, value: types, idx: Optional[index_type] = None) -> None:
        """配列に値をセットする
        :param value: セットしたい値
        :param idx: セット対象のインデックス情報
        """
        raise NotImplementedError

    def get(self, idx: Optional[index_type] = None) -> types:
        """配列から値を取り出す
        :return:
        """
        raise NotImplementedError

    def create(
        self, shape: tuple[int, ...], maxshape: Optional[tuple[int | slice | None, ...]]
    ) -> None:
        """配列を初期化する
        :param shape: 配列のshape
        :param maxshape: 配列の取りうる最大のshape
        :return:
        """
        raise NotImplementedError

    def resize(self, shape: tuple[int, ...]) -> None:
        """配列の大きさを変更する
        :param shape: 変更後のshape
        """
        raise NotImplementedError

    def append(self, arr: np.ndarray, axis: int) -> None:
        raise NotImplementedError

    def has(self) -> bool:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        return self.get().shape

    @property
    def maxshape(self) -> Optional[tuple[int, ...]]:
        raise NotImplementedError

    @maxshape.setter
    def maxshape(self, value: tuple[int, ...]) -> None:
        pass


@dataclass
class AttrParamBase:
    name: str


@dataclass
class ArrayParamBase:
    name: str
    dtype: Any


def init_check(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args[0].has():
            raise ValueError("Not initialized")
        return func(*args, **kwargs)

    return wrapper


from abc import ABCMeta, abstractmethod


class AbsDataStructure(metaclass=ABCMeta):
    @abstractmethod
    def reset(self, *args, **kwargs):
        pass


class DataStructureBase(AbsDataStructure):
    def reset(self, *args, **kwargs):
        raise NotImplementedError
