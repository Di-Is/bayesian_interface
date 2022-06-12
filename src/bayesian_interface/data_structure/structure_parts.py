import typing
from dataclasses import fields, dataclass
from typing import Any, Optional
from functools import wraps
from abc import ABCMeta, abstractmethod

import numpy as np

from bayesian_interface.data_structure.const import SaveType
from bayesian_interface.data_structure.misc import dynamically_load_class


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

    def __repr__(self):
        if self.has():
            return str(self.get())
        else:
            return "Not init"


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
    def get(self, idx: Optional[Index] = None, copy: bool = True) -> types:
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

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __repr__(self):
        if self.has():
            return f"shape: {self.shape}"
        else:
            return "Not init"


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


@dataclass
class AbsData(metaclass=ABCMeta):
    @abstractmethod
    def memory_dflt_par(self):
        pass

    @abstractmethod
    def hdf5_dflt_par(self):
        pass

    @abstractmethod
    def netcdf4_dflt_par(self):
        pass

    def __init__(
        self,
        save_type: SaveType = SaveType.memory,
        each: typing.Optional[dict] = None,
        **common,
    ):
        match save_type:
            case SaveType.memory:
                dct = self.memory_dflt_par()
            case SaveType.hdf5:
                dct = self.hdf5_dflt_par()
            case SaveType.netcdf:
                dct = self.netcdf4_dflt_par()
            case _:
                raise ValueError
        each = self.merge_param(dct, each, **common)
        for var_info in fields(self):
            name = var_info.name
            save_type_name = str(save_type).replace("SaveType.", "")
            module = f"bayesian_interface.data_structure.{save_type_name}"
            if var_info.type is AbsAttr:
                factory = dynamically_load_class(module, "AttrFactory")
            elif var_info.type is AbsArray:
                factory = dynamically_load_class(module, "ArrayFactory")
            else:
                raise TypeError(
                    f"The annotation {var_info.type} of variable {name} is an invalid."
                )
            setattr(self, name, factory.create(name=name, **each[name]))

    def __new__(cls, *args, **kwargs):
        dataclass(cls, init=False)  # noqa
        return super().__new__(cls)

    @classmethod
    def merge_param(
        cls,
        default_param: dict,
        each: typing.Optional[dict[str, dict]] = None,
        **common,
    ) -> dict:
        common = {} if common is None else common.copy()
        each = {} if each is None else each.copy()

        names = list(default_param.keys())
        result = {}
        for name in names:
            v = default_param[name]
            if name in each:
                each[name] = v | each[name]
            else:
                each[name] = v.copy()
            result[name] = common | each[name]

        return result

    @classmethod
    def get_attr_names(cls):
        return tuple(var.name for var in fields(cls) if var.type is AbsAttr)

    @classmethod
    def get_array_names(cls):
        return tuple(var.name for var in fields(cls) if var.type is AbsArray)
