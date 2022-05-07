from typing import Any, Optional
from functools import wraps
from abc import ABCMeta, abstractmethod

import numpy as np


def init_check(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args[0].has():
            raise ValueError("Not initialized")
        return func(*args, **kwargs)

    return wrapper


class AbsAttr(metaclass=ABCMeta):
    """Attributeのデータ構造の基底クラス"""

    @abstractmethod
    def set(self, value: Any) -> None:
        pass

    @abstractmethod
    def get(self) -> Any:
        pass

    @abstractmethod
    def has(self) -> bool:
        pass


types = int | float | str | complex | np.ndarray
Index = int | slice | tuple[int | slice, ...]
Shape = tuple[int, ...]
MaxShape = tuple[int | None, ...]


class AbsArray(metaclass=ABCMeta):
    @abstractmethod
    def set(self, value: types, idx: Optional[Index] = None) -> None:
        """配列に値をセットする
        :param value: セットしたい値
        :param idx: セット対象のインデックス情報
        """
        pass

    @abstractmethod
    def get(self, idx: Optional[Index] = None) -> types:
        """配列から値を取り出す
        :return:
        """
        pass

    @abstractmethod
    def create(self, shape: Shape, maxshape: MaxShape, dtype: Any, **kwargs) -> None:
        """配列を初期化する
        :param shape: 配列のshape
        :param maxshape: 配列の取りうる最大のshape
        :param dtype:
        :return:
        """
        pass

    @abstractmethod
    def resize(self, shape: Shape) -> None:
        """配列の大きさを変更する
        :param shape: 変更後のshape
        """
        pass

    @abstractmethod
    def has(self) -> bool:
        pass

    @property
    @abstractmethod
    def shape(self) -> Shape:
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        raise NotImplementedError

    @init_check
    def append(self, arr: np.ndarray, axis: int) -> None:
        shape = []
        indices = []
        for i, old in enumerate(self.shape):
            if i <= arr.ndim - 1:
                add = arr.shape[i]
            else:
                add = 1
            shape.append(old + add if i == axis else old)
            indices.append(slice(old, None) if i == axis else slice(None))
        self.resize(tuple(shape))
        self.set(arr, tuple(indices))


class AttrFactory(metaclass=ABCMeta):
    @classmethod
    def create(cls, **kwargs) -> AbsAttr:
        class_obj = cls.get_dataclass()
        return class_obj(**kwargs)

    @classmethod
    @abstractmethod
    def get_dataclass(cls) -> AbsAttr.__class__:
        pass


class ArrayFactory(metaclass=ABCMeta):
    @classmethod
    def create(cls, **kwargs) -> AbsArray:
        class_obj = cls.get_dataclass()
        return class_obj(**kwargs)

    @classmethod
    @abstractmethod
    def get_dataclass(cls) -> AbsArray.__class__:
        pass
